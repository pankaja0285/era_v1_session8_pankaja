import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from utils import get_device, load_config

class Dataset():
    @staticmethod
    def download_trainset_calc_mean_std(datapath):
        transforms_train = transforms.Compose([transforms.ToTensor()])
        # train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        train_set = datasets.CIFAR10(root=datapath, train=True, download=True, transform=transforms_train)
        mean = train_set.data.mean(axis=(0,1,2))/255
        std = train_set.data.std(axis=(0,1,2))/255
        return mean, std
    
    @staticmethod
    def data_tranformations(horizontalflip_prob, rotate_limit, shiftscalerotate_prob, num_holes, cutout_prob, datapath):
        # # Calculate mean and std deviation for cifar dataset
        # # mean, std = Dataset.download_trainset_calc_mean_std(datapath)
        # train_set = datasets.CIFAR10(root=datapath, train=True, download=True, transform=transforms_train)
        # mean = train_set.data.mean(axis=(0,1,2))/255
        # # Train Phase transformations
        # train_transforms = A.Compose([A.HorizontalFlip(p=horizontalflip_prob),
        #                                                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=rotate_limit, 
        #                                                p=shiftscalerotate_prob),
        #                             A.CoarseDropout(max_holes=num_holes,min_holes = 1, max_height=16, max_width=16, 
        #                             p=cutout_prob,fill_value=tuple([x * 255.0 for x in mean]),
        #                             min_height=16, min_width=16),
        #                             A.ColorJitter(p=0.25,brightness=0.3, contrast=0.3, saturation=0.30, hue=0.2),
        #                             A.ToGray(p=0.15),
        #                             A.Normalize(mean=mean, std=std,always_apply=True),
        #                             ToTensorV2()
        #                             ])
        
        # # Test Phase transformations
        # test_transforms = A.Compose([A.Normalize(mean=mean, std=std, always_apply=True),
        #                             ToTensorV2()])

        # return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]
        # Train Phase transformations
        train_transforms = transforms.Compose([
                                            #  transforms.Resize((28, 28)),
                                            #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(15),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                            # Note the difference between (0.1307) and (0.1307,)
                                            ])

        # Test Phase transformations
        test_transforms = transforms.Compose([
                                            #  transforms.Resize((28, 28)),
                                            #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                            ])
        return train_transforms, test_transforms
    
    def __init__(self, path="../data", config_file='config.yml'):
        super(Dataset, self).__init__()
        self.datapath = path
        self.config = load_config(config_file)
        self.download()

    def download(self):
        horizontalflip_prob = self.config['data_augmentation']['args']['horizontalflip_prob']
        rotate_limit = self.config['data_augmentation']['args']['rotate_limit']
        shiftscalerotate_prob = self.config['data_augmentation']['args']['shiftscalerotate_prob']
        num_holes = self.config['data_augmentation']['args']['num_holes']
        cutout_prob = self.config['data_augmentation']['args']['cutout_prob']
        
        transform_train, transform_test = Dataset.data_tranformations(horizontalflip_prob,rotate_limit,
                                                                      shiftscalerotate_prob,num_holes,cutout_prob, 
                                                                      self.datapath)
        
        # Create train dataset and dataloader
        # dataloader arguments 
        batch_size = self.config['data_loader']['args']['batch_size']
        num_workers = self.config['data_loader']['args']['num_workers']
        use_cuda = torch.cuda.is_available()
        print("CUDA Available?", use_cuda)
        dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=64)
        
        # Create train dataset and dataloader
        self.train_dataset = torchvision.datasets.CIFAR10(root=self.datapath, train=True, download=True, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, **dataloader_args)

        # Create test dataset and dataloader
        self.test_dataset = torchvision.datasets.CIFAR10(root=self.datapath, train=False, download=True, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, **dataloader_args)
        
        # set as class variables
        return self.train_dataset, self.train_loader, self.test_dataset, self.test_loader
