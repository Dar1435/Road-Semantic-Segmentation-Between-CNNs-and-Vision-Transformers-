import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, mask_transform=None, input_size=256, is_test=False):
        """
        Dataset for EPFL ML Road Segmentation challenge.
        
        Args:
            image_dir (str): Directory containing input images (e.g., training/images or test).
            mask_dir (str, optional): Directory containing ground truth masks (e.g., training/groundtruth).
            transform (callable, optional): Transform for images.
            mask_transform (callable, optional): Transform for masks.
            input_size (int): Size to resize images and masks to (e.g., 256).
            is_test (bool): Whether this is a test dataset (images in subdirectories).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.input_size = input_size
        self.is_test = is_test
        
        if self.is_test:
            # Test dataset: images in subdirectories (e.g., test/1/test_01.png)
            self.images = []
            for subdir in sorted(os.listdir(image_dir)):
                subdir_path = os.path.join(image_dir, subdir)
                if os.path.isdir(subdir_path):
                    for img_file in os.listdir(subdir_path):
                        if img_file.endswith('.png'):
                            self.images.append(os.path.join(subdir_path, img_file))
            self.images = sorted(self.images)
            # Dummy masks (same as images for test)
            self.masks = self.images
        else:
            # Training dataset: images and masks in flat directories
            self.images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
            self.masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
            assert len(self.images) == len(self.masks), f"Number of images ({len(self.images)}) and masks ({len(self.masks)}) must match"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.images[idx]).convert('RGB')
        
        # Load mask (or dummy for test)
        mask = Image.open(self.masks[idx]).convert('L')  # Grayscale
        
        # Apply transforms with synchronized random seed
        if self.transform and self.mask_transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
        elif self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) if self.mask_transform else mask
        
        # Convert mask to binary tensor [256, 256]
        if not self.is_test:
            mask = torch.squeeze(mask, dim=0)  # [1, 256, 256] -> [256, 256]
            mask = (mask > 0.5).long()  # Threshold to {0, 1}
        else:
            # Dummy mask for test (not used)
            mask = torch.zeros((self.input_size, self.input_size), dtype=torch.long)
        
        return image, mask

def get_transform(input_size=256, is_train=True):
    """
    Transform for EPFL dataset images and masks.
    
    Args:
        input_size (int): Size to resize images and masks to (e.g., 256).
        is_train (bool): Whether to apply augmentation for training.
    
    Returns:
        image_transform (callable): Transform for images.
        mask_transform (callable): Transform for masks.
    """
    if is_train:
        image_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=90, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=90, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    else:
        image_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    return image_transform, mask_transform