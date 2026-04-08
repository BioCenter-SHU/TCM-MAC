import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import warnings
import torchvision.datasets as datasets
import albumentations as A
from PIL import Image
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torch.utils.data.distributed import DistributedSampler
import numpy as np

warnings.filterwarnings('ignore')

def build_cifar(use_cifar10=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())
    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='./dataset/CIFAR',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='./dataset/CIFAR',
                              train=False, download=download, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='./dataset/CIFAR',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='./dataset/CIFAR',
                               train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset

#for ImageNet
class Transform():
     def __init__(self,transform):
        self.transform=transform
     def __call__(self,image):
        return self.transform(image=image)["image"]
     
def open_img(img_path):
     img=Image.open(img_path).convert('RGB')
     return np.array(img)

def build_imgnet(root):

    train_transforms = A.Compose([
        A.RandomResizedCrop(size=(224, 224)),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()])
    
    val_transforms = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    imgnet_train_dataset = datasets.ImageFolder(
        root+'/train', 
        transform=Transform(train_transforms), 
        loader=open_img)
        
    imgnet_valid_dataset = datasets.ImageFolder(
        root+'/val',
        transform=Transform(val_transforms),
        loader=open_img)
    
    return imgnet_train_dataset, imgnet_valid_dataset