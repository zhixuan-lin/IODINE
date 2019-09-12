from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Obj3D(Dataset):
    def __init__(self, root, mode):
        path = os.path.join(root, mode)
        assert os.path.exists(root), 'Path {} does not exist'.format(path)

        self.img_paths = []
        for file in os.scandir(path):
            img_path = file.path
            self.img_paths.append(img_path)

        self.img_paths.sort()

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = io.imread(img_path)[:, :, :3]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(128),
            transforms.ToTensor(),
        ])
        img = transform(img)

        return img, None # in clevr code, second return value was mask, but unused

    def __len__(self):
        return len(self.img_paths)

