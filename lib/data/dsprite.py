import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import h5py

class MultiDSprites(Dataset):
    def __init__(self, root, mode):
        self.root = root
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'images/{}.png'.format(index))
        mask_path = os.path.join(self.root, 'masks/{}.npy'.format(index))
        img = io.imread(img_path)
        mask = np.load(mask_path)
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.CenterCrop(192),
            # transforms.Resize(32),
            transforms.ToTensor()
        ])
        
        # mask = torch.from_numpy(mask).float()
        
        # targets = {
        #     'mask': mask
        # }
        
        return transform(img).float()
        
        
    def __len__(self):
        return 60000
    
    
    
