import torchvision
from torchvision import transforms
from .wrapper import CacheClassLabel
from .xraydataset import XrayDataset
from sklearn.model_selection import train_test_split
import pandas as pd

def MNIST(dataroot, train_aug=False):
    # Add padding to make 32x32
    #normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    val_transform = transforms.Compose([
        transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset

def CIFAR10(dataroot, train_aug=False):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
        )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR100(dataroot, train_aug=False):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset

def XRAY(dataroot, train_aug=True, df_path = "/gdrive/MyDrive/CL_notebooks/data/img_ids_filtered.csv"):
    df = pd.read_csv(df_path)
    train_df, test_df = train_test_split(df, test_size=0.2)
    transformations = transforms.Compose([transforms.ToTensor(), torchvision.transforms.RandomResizedCrop(size = 32, scale = (0.9, 1.0), ratio = (1,1)), torchvision.transforms.RandomPerspective(distortion_scale=0.1, p=0.3) ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
    transformations_test = transforms.Compose([transforms.ToTensor(), torchvision.transforms.RandomResizedCrop(size = 32, scale = (0.9, 1.0), ratio = (1,1)) ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
    if train_aug:
        train_set = XrayDataset(train_df, transform = transformations )
    else:
        train_set = XrayDataset(train_df, transform = transformations_test )
    test_set = XrayDataset(test_df, transform = transformations_test )
    return train_set, test_set
 