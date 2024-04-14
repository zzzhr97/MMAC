import albumentations as albu
import torch
import cv2

def trainAUG():
    train_transform = [
        albu.Resize(height=512,width=512),
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(
            scale_limit=0, 
            rotate_limit=(-180,180), 
            shift_limit=0, 
            p=1, 
            border_mode=cv2.BORDER_CONSTANT
        ),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),

        # albu.RandomBrightness(limit=0.4),
        # albu.RandomContrast(limit=0.4),
    ]
    return albu.Compose(train_transform)


def validAUG():
    return albu.Compose([albu.Resize(height=512,width=512)])

def to_tensor(x, **kwargs):
    return torch.tensor(x.transpose(2, 0, 1).astype('float32'))

def getPreProcess(preprocessing_fn):
    preProcess_trans = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(preProcess_trans)