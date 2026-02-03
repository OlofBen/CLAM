from functools import cache
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py

class Whole_Slide_Bag(Dataset):
    def __init__(self, file_path, img_transforms=None):
        self.roi_transforms = img_transforms
        
        # Load EVERYTHING into RAM once
        with h5py.File(file_path, 'r') as f:
            print(f"Loading {file_path} into RAM...")
            self.imgs = np.array(f['imgs'])    # The pixels move to RAM here
            self.coords = np.array(f['coords']) # The coordinates move to RAM here
            
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # This is now a pure RAM-to-CPU transfer (blazing fast)
        img = self.imgs[idx]
        coord = self.coords[idx]
        
        img = Image.fromarray(img)
        if self.roi_transforms:
            img = self.roi_transforms(img)
            
        return {'img': img, 'coord': coord}

