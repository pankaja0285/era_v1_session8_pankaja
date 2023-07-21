import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as transforms

from utils import helper


class CIFAR10Albumentation:
    
    def __init__(self):
        pass
    
    def train_transform(self,mean,std):
        # Train Phase transformations
        train_transforms = A.Compose([A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                                      A.RandomCrop(width=32, height=32,p=1),
                                      A.Rotate(limit=5),
                                      #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.25),
                                      A.CoarseDropout(max_holes=1,min_holes = 1, max_height=16, max_width=16, p=0.5,fill_value=tuple([x * 255.0 for x in mean]),
                                      min_height=16, min_width=16),
                                      A.Normalize(mean=mean, std=std,always_apply=True),
                                      ToTensorV2()
                                    ])
        return lambda img:train_transforms(image=np.array(img))["image"]
                                
    def test_transform(self,mean,std):
        # Test Phase transformations
        test_transforms = A.Compose([A.Normalize(mean=mean, std=std, always_apply=True),
                                 ToTensorV2()])
        return lambda img:test_transforms(image=np.array(img))["image"]
        