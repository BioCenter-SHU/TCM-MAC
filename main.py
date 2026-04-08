import argparse
import os
import torch.nn.parallel
import torch.optim
from models.TCMMAC_SNN import *
from data.augmentations import *
from data.datasets import build_cifar
from utils.utils import *
from spikingjelly.datasets import cifar10_dvs
import numpy as np
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='TCMMAC-SNN for Spiking Neural Networks')

parser.add_argument('--DS',
                    default='',
                    type=str,
                    help='cifar10, cifar100, dvs_cifar10')

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
                    default=0.1,
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
                    default=5e-5,
                    type=float,
                    metavar='N',
                    help='weight_decay')
parser.add_argument('--beta', 
                    default=1.0, 
                    type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', 
                    default=0.5, 
                    type=float,
                    help='cutmix probability')
parser.add_argument('--mixup', 
                    type=float, 
                    default=0.5,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--mixup-prob', 
                    type=float, 
                    default=0.5,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', 
                    type=float, 
                    default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', 
                    type=str, 
                    default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--smoothing', 
                    type=float, 
                    default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument("--no-tcmmac", dest="tcmmac_on", action="store_false", help="disable TCMMAC")
parser.add_argument("--no-na", dest="na_on", action="store_false", help="disable NA")
parser.add_argument("--no-xa", dest="xa_on", action="store_false", help="disable XA")
parser.add_argument("--no-chao", dest="chao_on", action="store_false", help="disable Chao")
parser.set_defaults(tcmmac_on=True)
parser.set_defaults(na_on=True)
parser.set_defaults(xa_on=True)
parser.set_defaults(chao_on=True)
args = parser.parse_args()


def train(model, device, train_loader, criterion, optimizer, epoch, args):
    running_loss = 0
    model.train()
    imgs_seen = args.batch_size
    gpu_indices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    for idx in gpu_indices:
        torch.cuda.reset_peak_memory_stats(idx)
    M = len(train_loader)
    total = 0
    correct = 0
    r = np.random.rand(1)

    for  i,(images, targets) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        if args.DS in ['cifar10', 'cifar100']:
            # print('cifar')
            targets = targets.to(device)
            images = images.to(device)
            if args.beta > 0 and r < args.cutmix_prob:
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                    # compute output
                outputs = model(images)
                mean_out = outputs.mean(1)
                loss = criterion(mean_out, target_a) * lam + criterion(mean_out,target_b) * (1. - lam)
            else:
                # compute output
                outputs = model(images)
                mean_out = outputs.mean(1)
                loss = criterion(mean_out, targets)
        
        elif args.DS == 'dvs_cifar10':
            # print('dvs_cifar10')
            images, target = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)
            images = images.float()  
            N,T,C,H,W = images.shape
            train_aug = get_train_aug()
            trival_aug = get_trival_aug()
            mixup_fn = get_mixup_fn(args)
            
            images = torch.stack([(train_aug(images[i])) for i in range(N)])
            images = torch.stack([(trival_aug(images[i])) for i in range(N)])
            images, target = mixup_fn(images, target)
            targets = target.argmax(dim=-1)

            outputs = model(images)
            mean_out = outputs.mean(1)
            loss = criterion(mean_out, targets)
        
        else:
            raise NotImplementedError(args.DS)

        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(targets.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(targets.cpu()).sum().item())

    if gpu_indices:
        torch.cuda.synchronize()
        peak_mb_per_gpu = []
        for idx in gpu_indices:
            peak_mem = torch.cuda.max_memory_allocated(idx) / (1024 ** 2)
            peak_mb_per_gpu.append(peak_mem)
        total_peak = sum(peak_mb_per_gpu) if peak_mb_per_gpu else 0.0
        mb_per_img = total_peak / imgs_seen if imgs_seen > 0 else float("inf")
        print(f"[Epoch {epoch}] imgs={imgs_seen} peakMB={peak_mb_per_gpu} MB/img={mb_per_img:.4f}")

    return running_loss/M, 100 * correct / total

@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if args.DS in ['cifar10', 'cifar100']:
            # print('cifar')
            inputs = inputs.to(device)
            outputs = model(inputs)
            mean_out = outputs.mean(1)
            _, predicted = mean_out.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        
        elif args.DS == 'dvs_cifar10':
            # print('dvs_cifar10')
            inputs = inputs.to(device, non_blocking=True)
            target = targets.to(device, non_blocking=True)
            N,T,C,H,W = inputs.shape
            test_aug = get_test_aug()
            inputs = torch.stack([(test_aug(inputs[i])) for i in range(N)])
            inputs = inputs.float()
            outputs = model(inputs)
            mean_out = outputs.mean(1)
            _, predicted = mean_out.cpu().max(1)
            total += float(target.size(0))
            correct += float(predicted.eq(target.cpu()).sum().item())

        else:
            raise NotImplementedError(args.DS)
    final_acc = 100 * correct / total
    return final_acc

if __name__ == '__main__':

    seed_all(args.seed)

    if args.DS == 'cifar10':
        print('cifar10')
        num_CLS = 10
        save_ds_name = 'CIFAR10'
        train_dataset, val_dataset = build_cifar(use_cifar10=True)
    
    elif args.DS == 'cifar100': 
        print('cifar100')
        num_CLS = 100
        save_ds_name = 'CIFAR100'
        train_dataset, val_dataset = build_cifar(use_cifar10=False)

    elif args.DS == 'dvs_cifar10':
        print('dvs_cifar10')
        num_CLS = 10
        save_ds_name = 'DVSCIFAR10'
        origin_set = cifar10_dvs.CIFAR10DVS(root="./dataset/DVS_CIFAR10", data_type='frame', frames_number=args.time_step, split_by='number')
        train_dataset, val_dataset = split_to_train_test_set(0.9, origin_set, 10)
        
    else:
        raise NotImplementedError(args.DS)
        
    if args.DS == 'cifar10':
        print('cifar10')
        DP_model = TCMMAC_SNN_18(num_classes=num_CLS, time_step=args.time_step, TCMMAC_ON=args.tcmmac_on, dvs=False, 
                                 NA_ON=args.na_on, XA_ON=args.xa_on, Chao_ON=args.chao_on) 
    
    elif args.DS == 'cifar100':
        print('cifar100')
        DP_model = TCMMAC_SNN_18(num_classes=num_CLS, time_step=args.time_step, TCMMAC_ON=args.tcmmac_on, dvs=False, 
                                 NA_ON=args.na_on, XA_ON=args.xa_on, Chao_ON=args.chao_on)
    
    elif args.DS == 'dvs_cifar10':
        print('dvs_cifar10')
        DP_model = TCMMAC_SNN_18(num_classes=num_CLS, time_step=args.time_step, TCMMAC_ON=args.tcmmac_on, dvs=True, 
                                 NA_ON=args.na_on, XA_ON=args.xa_on, Chao_ON=args.chao_on)
    
    else:
        raise NotImplementedError(args.DS)
        
    # print('Total Parameters: %.2f' % (sum(p.numel() for p in DP_model.parameters())))
    DP_model = torch.nn.DataParallel(DP_model).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(DP_model.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=args.weight_decay)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    
    flag_name = 'TCM-MAC'
    logger = get_logger(f'{save_ds_name}-S{args.seed}-B{args.batch_size}-T{args.time_step}-E{args.epochs}-LR{args.learning_rate}-WD{args.weight_decay}-{flag_name}.log')
    logger.info('start training!')
    logger.info('Total Parameters: %.2f' % (sum(p.numel() for p in DP_model.parameters())))

    best_acc = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        loss, acc = train(DP_model, device, train_loader, criterion, optimizer, epoch, args)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch +1, args.epochs, loss, acc ))
        scheduler.step()
        facc = test(DP_model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch+1, args.epochs, facc ))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            torch.save(DP_model.module.state_dict(), f'{save_ds_name}-S{args.seed}-B{args.batch_size}-T{args.time_step}-E{args.epochs}-LR{args.learning_rate}-WD{args.weight_decay}-{flag_name}.pth')

        logger.info('Epoch:[{}/{}]\t Best Test acc={:.3f}'.format(best_epoch, args.epochs, best_acc ))
        logger.info('\n')
