import os
import torch
from torchvision import datasets, transforms

import imagefolder

 
def cifar10_png_wo_norm(dataroot, batch_size):

    path_train = os.path.join(dataroot, 'train')
    train_dataset = datasets.ImageFolder(root=path_train, transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    path_test = os.path.join(dataroot, 'test')
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=path_test, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=100, shuffle=False,
        num_workers=4)


    return train_loader, test_loader


def cifar100_png_wo_norm(dataroot, batch_size):

    path_train = os.path.join(dataroot, 'train')
    train_dataset = imagefolder.ImageFolder(root=path_train, transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    path_test = os.path.join(dataroot, 'test')
    test_loader = torch.utils.data.DataLoader(
        imagefolder.ImageFolder(root=path_test, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=100, shuffle=False,
        num_workers=4)


    return train_loader, test_loader