import argparse
import os
import torch.nn.parallel
import torch.optim
import torch.distributed as dist
from models.TCMMAC_SNN import *
from data.augmentations import *
from data.datasets import build_imgnet
from utils.utils import *
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser(description='TCMMAC-SNN for Spiking Neural Networks - ImageNet')

parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number')

parser.add_argument('--batch_size',
                    default=64,
                    type=int,
                    metavar='N')

parser.add_argument('--learning_rate',
                    default=0.05,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')

parser.add_argument('--seed',
                    default=42,
                    type=int,
                    help='seed for initializing training')

parser.add_argument('--time_step',
                    default=6,
                    type=int,
                    metavar='N',
                    help='snn simulation time steps (default: 6)')
parser.add_argument('--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs',
                    default=250,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay',
                    default=1e-5,
                    type=float,
                    metavar='N',
                    help='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument("--no-tcmmac", dest="tcmmac_on", action="store_false", help="disable TCMMAC")
parser.add_argument("--no-na", dest="na_on", action="store_false", help="disable NA")
parser.add_argument("--no-xa", dest="xa_on", action="store_false", help="disable XA")
parser.add_argument("--no-chao", dest="chao_on", action="store_false", help="disable Chao")
parser.set_defaults(tcmmac_on=True)
parser.set_defaults(na_on=True)
parser.set_defaults(xa_on=True)
parser.set_defaults(chao_on=True)

# distributed training parameters
parser.add_argument('--world-size', default=4, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()

def train(model, train_loader, criterion, optimizer):
    running_loss = 0
    model.train()
    
    M = len(train_loader)
    total = 0
    correct = 0

    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        
        # ImageNet logic
        targets = targets.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        
        outputs = model(images)
        mean_out = outputs.mean(1)
        loss = criterion(mean_out, targets)
        
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()

        total += float(targets.size(0))
        _, predicted = mean_out.max(1)
        correct += float(predicted.eq(targets).sum().item())

    return running_loss/M, 100 * correct / total

@torch.no_grad()
def test(model, test_loader):
    correct = 0
    total = 0
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # ImageNet logic
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
            
    final_acc = 100 * correct / total
    return final_acc

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("start init_distributed_mode")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)

def is_main_process():
    return dist.get_rank() == 0

if __name__ == '__main__':

    init_distributed_mode(args)

    seed_all(args.seed)

    if is_main_process():
        print('ImageNet')
        
    num_CLS = 1000
    save_ds_name = 'ImageNet'
    train_dataset, val_dataset = build_imgnet(root="./dataset/ImageNet")
    
    # DDP: Distributed Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    # DDP: Adjust batch size per GPU
    world_size = dist.get_world_size()
    batch_size_per_gpu = args.batch_size // world_size
    if is_main_process():
        print(batch_size_per_gpu)

    # DDP: DataLoader with sampler
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_per_gpu, shuffle=False,
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_per_gpu,
                                              shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    DP_model = TCMMAC_SNN_34(num_classes=num_CLS, time_step=args.time_step, TCMMAC_ON=args.tcmmac_on, 
                           NA_ON=args.na_on, XA_ON=args.xa_on, Chao_ON=args.chao_on)
    
    # print('Total Parameters: %.2f' % (sum(p.numel() for p in DP_model.parameters())))
    
    best_acc = 0
    best_epoch = args.start_epoch
    checkpoint = None

    if args.resume:
        if os.path.isfile(args.resume):
            if is_main_process():
                print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                args.start_epoch = checkpoint['epoch']
                best_acc = checkpoint['best_acc']
                DP_model.load_state_dict(checkpoint['state_dict'])
                if is_main_process():
                    print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                DP_model.load_state_dict(checkpoint)
                if is_main_process():
                    print(f"=> loaded checkpoint '{args.resume}' (weights only)")
        else:
            if is_main_process():
                print(f"=> no checkpoint found at '{args.resume}'")

    DP_model = DP_model.cuda()
    DP_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(DP_model)
    DP_model = torch.nn.parallel.DistributedDataParallel(DP_model, device_ids=[args.gpu])

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(DP_model.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=args.weight_decay)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    if checkpoint is not None:
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            if args.start_epoch > 0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs, last_epoch=args.start_epoch - 1)
        del checkpoint

    flag_name = 'TCM-MAC-DDP'
    
    # Initialize logger only on rank 0
    if is_main_process():
        logger = get_logger(f'{save_ds_name}-S{args.seed}-B{args.batch_size}-T{args.time_step}-E{args.epochs}-LR{args.learning_rate}-WD{args.weight_decay}-{flag_name}.log')
        logger.info('start training!')
        logger.info('Total Parameters: %.2f' % (sum(p.numel() for p in DP_model.parameters())))
    else:
        logger = None

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        loss, acc = train(DP_model, train_loader, criterion, optimizer)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        if is_main_process():
            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}\t lr={:.6f}'.format(epoch +1, args.epochs, loss, acc, current_lr))
        
        scheduler.step()
        facc = test(DP_model, test_loader)
        
        if is_main_process():
            logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch+1, args.epochs, facc ))

            if best_acc < facc:
                best_acc = facc
                best_epoch = epoch + 1
                state = {
                    'epoch': epoch + 1,
                    'state_dict': DP_model.module.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(state, f'{save_ds_name}-S{args.seed}-B{args.batch_size}-T{args.time_step}-E{args.epochs}-LR{args.learning_rate}-WD{args.weight_decay}-{flag_name}.pth')

            logger.info('Epoch:[{}/{}]\t Best Test acc={:.3f}'.format(best_epoch, args.epochs, best_acc ))
            logger.info('\n')

    dist.destroy_process_group()
