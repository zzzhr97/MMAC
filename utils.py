import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import segmentation_models_pytorch as smp
from datetime import datetime
from datetime import timedelta
from datetime import timezone

modelDict = {
    'unet': 'Unet',
    'unet++': 'UnetPlusPlus',
    'manet': 'MAnet',
    'linknet': 'Linknet',
    'fpn': 'FPN',
    'pspnet': 'PSPNet',
    'pan': 'PAN',
    'deeplabv3': 'DeepLabV3',
    'deeplabv3+': 'DeepLabV3Plus',
}

def getLrScheduler(lrSchedulerScheme, optimizer, **kwargs):
    if lrSchedulerScheme == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=15,
            gamma=0.75
        )
    elif lrSchedulerScheme == 'multiStep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.36
        )
    elif lrSchedulerScheme == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20,
            eta_min=1e-8
        )
    elif lrSchedulerScheme == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.96
        )
    elif lrSchedulerScheme == 'cyclic':
        
        base_lr = optimizer.param_groups[0]['lr'] * 0.1
        max_lr = optimizer.param_groups[0]['lr'] * 2.5

        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=20,
            step_size_down=5
        )
    else:
        raise ValueError('Invalid lr scheduler name')

def getTime():
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    timeStr = beijing_now.strftime('%m-%d_%H:%M:%S')
    return timeStr

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getModel(args):
    kwargs = {
        'encoder_name': args.encoder,
        'encoder_weights': args.encoder_weights,
        'classes': 1,
        'in_channels': 3,
        'activation': args.activation,
    }
    if args.model in modelDict.keys():
        return eval(f"smp.{modelDict[args.model]}(**kwargs)")
    else:
        raise ValueError('Invalid model name')

def diceCoef(y_pr, y_gt, threshold=0.5, epsilon=1e-9):
    y_pr = (y_pr > threshold).type(y_pr.dtype)
    intersection = torch.sum(y_pr * y_gt)
    diceScore = (2. * intersection + epsilon) \
        / (torch.sum(y_pr) + torch.sum(y_gt) + epsilon)
    return diceScore
    
class AddLoss(nn.Module):
    def __init__(self, loss1, loss2, weight=[1.0,1.0]):
        super(AddLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight = weight

    def forward(self, y_pr, y_gt):
        return self.weight[0] * self.loss1(y_pr, y_gt) \
            + self.weight[1] * self.loss2(y_pr, y_gt)
    
def getLoss(lossName, weight=[1.0,1.0]):
    if lossName == 'dice':
        return smp.utils.losses.DiceLoss()
    elif lossName == 'focal':
        return smp.losses.FocalLoss(mode='binary')
    elif lossName == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    elif lossName == 'dice+bce':
        return AddLoss(
            smp.utils.losses.DiceLoss(), 
            torch.nn.BCEWithLogitsLoss(),
            weight
        )
    elif lossName == 'dice+focal':
        return AddLoss(
            smp.utils.losses.DiceLoss(), 
            smp.losses.FocalLoss(mode='binary'),
            weight
        )
    else:
        raise ValueError('Invalid loss name')
    
def getOptimizer(optimizerName, params, **kwargs):
    if optimizerName == 'sgd':
        return torch.optim.SGD(
            params=params,
            lr=kwargs['lr'],
            weight_decay=kwargs['wd']
        )
    elif optimizerName == 'adam':
        return torch.optim.Adam(
            params=params,
            lr=kwargs['lr'],
            weight_decay=kwargs['wd']
        )
    elif optimizerName == 'adamw':
        return torch.optim.AdamW(
            params=params,
            lr=kwargs['lr'],
            weight_decay=kwargs['wd']
        )
    else:
        raise ValueError('Invalid optimizer name')

