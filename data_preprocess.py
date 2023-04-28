
from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import random

from torch.nn import DataParallel
import torchattacks

import matplotlib.pyplot as plt
import os
import time

import resnet


torch.manual_seed(2)
np.random.seed(2)
random.seed(2)
 
model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        
        layers = nn.ModuleList()

        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))

        layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

        layers.append(shortcut)
        layers.append(nn.ReLU(inplace=True))

        self.layers = layers
            
    def forward(self, x):
        fwd = self.layers[0](x) # conv layers
        fwd += self.layers[1](x) # shortcut
        fwd = self.layers[2](fwd) # activation
        return fwd


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def test(test_loader, model, device):

    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (img, label) in enumerate(test_loader):
            img, label = img.to(device), label.to(device)
            output = model(img)
            prec1, prec5 = accuracy(output, label, topk=(1, 5))

            top1.update(prec1[0], img.size(0))
            top5.update(prec5[0], img.size(0))
        
    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc



def test2(img, label, model, device):

    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():
        img, label = img.to(device), label.to(device)
        output = model(img)
        prec1, prec5 = accuracy(output, label, topk=(1, 5))

        top1.update(prec1[0], img.size(0))
        top5.update(prec5[0], img.size(0))
        
    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc



def cifar10_png_wo_norm(dataroot, batch_size):

    path_train = os.path.join(dataroot, 'train')
    train_dataset = datasets.ImageFolder(root=path_train, transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    path_test = os.path.join(dataroot, 'test')
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=path_test, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)


    return train_loader, test_loader



def cifar100_png_wo_norm(dataroot, batch_size):

    path_train = os.path.join(dataroot, 'train')
    train_dataset = datasets.ImageFolder(root=path_train, transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    path_test = os.path.join(dataroot, 'test')
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=path_test, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)


    return train_loader, test_loader



def get_model(model_params, model_path, test_loader, device):

    model = resnet.__dict__['resnet56']()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location="{}".format(device)))


    return model


def get_attacks_params():
    attacks_params = {}
    attacks_params['FGSM'] = {}
    attacks_params['FGSM']['eps'] = 8.0 / 255.0

    attacks_params['IFGSM'] = {}
    attacks_params['IFGSM']['eps'] = 8.0 / 255.0
    attacks_params['IFGSM']['alpha'] = 2.0 / 255.0
    attacks_params['IFGSM']['iters'] = 10

    attacks_params['PGD10'] = {}
    attacks_params['PGD10']['eps'] = 8.0 / 255.0
    attacks_params['PGD10']['step_size'] = 2.0 / 255.0
    attacks_params['PGD10']['iters'] = 10

    attacks_params['PGD100'] = {}
    attacks_params['PGD100']['eps'] = 8.0 / 255.0
    attacks_params['PGD100']['step_size'] = 2.0 / 255.0
    attacks_params['PGD100']['iters'] = 100

    attacks_params['DeepFool'] = {}
    attacks_params['DeepFool']['iters'] = 50

    attacks_params['CW'] = {}
    attacks_params['CW']['targeted'] = False
    attacks_params['CW']['c'] = 0.1
    attacks_params['CW']['kappa'] = 0
    attacks_params['CW']['iters'] = 100
    attacks_params['CW']['lr'] = 0.01

    model_params = {}
    model_params['block_type'] = 'basic'
    model_params['num_blocks'] = [9,9,9]
    model_params['num_classes'] = 10
    model_params['input_size'] = 32
    return attacks_params, model_params


def get_adv(model, images, labels, attacks, attacks_params):
    fgsm_img = images.clone().detach()
    ifgsm_img = images.clone().detach()
    pgd_img = images.clone().detach()
    pgd100_img = images.clone().detach()
    deepfool_img = images.clone().detach()
    cw_img = images.clone().detach()

    model.eval()
    
    if 'FGSM' in attacks:
        fgsm_attack = torchattacks.FGSM(model, eps=attacks_params['FGSM']['eps'])
        fgsm_adv = fgsm_attack(fgsm_img, labels)
    if 'IFGSM' in attacks:
        ifgsm_attack = torchattacks.IFGSM(model, eps=attacks_params['IFGSM']['eps'], alpha=attacks_params['IFGSM']['alpha'], iters=attacks_params['IFGSM']['iters'])
        ifgsm_adv = ifgsm_attack(ifgsm_img, labels)
    if 'PGD10' in attacks:
        pgd_attack = torchattacks.PGD(model, eps=attacks_params['PGD10']['eps'], alpha=attacks_params['PGD10']['step_size'], iters=attacks_params['PGD10']['iters'])
        pgd_adv = pgd_attack(pgd_img, labels)
    if 'PGD100' in attacks:
        pgd100_attack = torchattacks.PGD(model, eps=attacks_params['PGD100']['eps'], alpha=attacks_params['PGD100']['step_size'], iters=attacks_params['PGD100']['iters'])
        pgd100_adv = pgd100_attack(pgd100_img, labels)
    if 'DeepFool' in attacks:
        deepfool_attack = torchattacks.DeepFool(model, iters=attacks_params['DeepFool']['iters'])
        deepfool_adv = deepfool_attack(deepfool_img, labels)
    if 'CW' in attacks: # CW(L2)
        cw_attack = torchattacks.CW(model, targeted=attacks_params['CW']['targeted'], c=attacks_params['CW']['c'], kappa=attacks_params['CW']['kappa'], iters=attacks_params['CW']['iters'], lr=attacks_params['CW']['lr'])
        cw_adv = cw_attack(cw_img, labels)

    return fgsm_adv, ifgsm_adv, pgd_adv, pgd100_adv, deepfool_adv, cw_adv

def main():
    gpu = 0
    dataroot = 'datasets/cifar10'
    # dataroot = 'datasets/cifar100'
    batch_size = 1

    model_path = 'pytorch_resnet_cifar10/best_model.th'
    # model_path = 'pytorch_resnet_cifar100/best_model.th'
    attacks = ['FGSM', 'IFGSM', 'PGD10', 'PGD100', 'DeepFool', 'CW']
    attacks_params, model_params = get_attacks_params()

    if gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(gpu))
    

    train_loader, test_loader = cifar10_png_wo_norm(dataroot, batch_size)
    # train_loader, test_loader = cifar100_png_wo_norm(dataroot, batch_size)
    model = get_model(model_params, model_path, test_loader, device)


    for i, (images, labels) in enumerate(train_loader):       
        images, labels = images.to(device), labels.to(device)
        fgsm_adv, ifgsm_adv, pgd_adv, pgd100_adv, deepfool_adv, cw_adv = get_adv(model, images, labels, attacks, attacks_params)
        img_adv = torch.cat((images, fgsm_adv, ifgsm_adv, pgd_adv, pgd100_adv, deepfool_adv, cw_adv), 3)

        save_path_adv = 'datasets/cifar10_resnet56_w_adv'
        # save_path_adv = 'datasets/cifar100_resnet56_w_adv'

        if not os.path.exists(save_path_adv):
            os.makedirs(save_path_adv)
        if not os.path.exists('{}/train'.format(save_path_adv)):
            os.makedirs('{}/train'.format(save_path_adv))

        with torch.no_grad():
            for j in range(batch_size):
                cur_index = int(i * batch_size + j)
                cur_label = int(labels[j].squeeze().detach().cpu().numpy())
                print(cur_index)
                if not os.path.exists('{}/train/{}'.format(save_path_adv, cur_label)):
                    os.makedirs('{}/train/{}'.format(save_path_adv, cur_label))
                plt.imsave('{}/train/{}/{}.png'.format(save_path_adv, cur_label, cur_index), img_adv[j].squeeze().permute(1,2,0).cpu().numpy())
    
    for i, (images, labels) in enumerate(test_loader):
                
        images, labels = images.to(device), labels.to(device)
        fgsm_adv, ifgsm_adv, pgd_adv, pgd100_adv, deepfool_adv, cw_adv = get_adv(model, images, labels, attacks, attacks_params)
        img_adv = torch.cat((images, fgsm_adv, ifgsm_adv, pgd_adv, pgd100_adv, deepfool_adv, cw_adv), 3)
        
        save_path_adv = 'datasets/cifar10_resnet56_w_adv'
        # save_path_adv = 'datasets/cifar100_resnet56_w_adv'

        if not os.path.exists(save_path_adv):
            os.makedirs(save_path_adv)
        if not os.path.exists('{}/test'.format(save_path_adv)):
            os.makedirs('{}/test'.format(save_path_adv))

        with torch.no_grad():
            for j in range(batch_size):
                cur_index = int(i * batch_size + j)
                cur_label = int(labels[j].squeeze().detach().cpu().numpy())
                print(cur_index)
                if not os.path.exists('{}/test/{}'.format(save_path_adv, cur_label)):
                    os.makedirs('{}/test/{}'.format(save_path_adv, cur_label))
                plt.imsave('{}/test/{}/{}.png'.format(save_path_adv, cur_label, cur_index), img_adv[j].squeeze().permute(1,2,0).cpu().numpy())
    



if __name__ == '__main__':

    main()

