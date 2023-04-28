import numpy as np
import random
import torch

import torch.nn.functional as F


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


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def PGD_attack(x, y, model, device):
    K = 10
    eps = 8.0 / 255.0
    step_size = 2.0 / 255.0
    model.eval()
    
    x_adv = x + torch.FloatTensor(x.size()).uniform_(-eps, eps).to(device)

    for _ in range(K):
        x_adv.requires_grad=True
        pred = F.cross_entropy(model(x_adv)[-1], y)
        pred.backward()
        grad = x_adv.grad.clone()
        # update x_adv
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)

    return x_adv


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
    