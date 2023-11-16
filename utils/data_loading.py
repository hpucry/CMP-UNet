import os.path
import torch
from PIL import Image
from os import listdir
from os.path import splitext
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2

class BasicDataset(Dataset):
    def __init__(self, dir: str , mask_suffix: str = '', is_train: bool = True, W: int = 256, H: int = 256):
        self.images_dir = Path(dir+'/imgs')
        self.mask_dir = Path(dir+'/masks')
        self.mask_suffix=mask_suffix
        self.W=W
        self.H=H
        self.is_train=is_train

        self.ids = [splitext(file)[0] for file in listdir(self.images_dir)]

        if is_train:
            self.color_aug = transforms.Compose([
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.002)]), p=0.8),
                transforms.RandomChoice(
                    [transforms.RandomAdjustSharpness(2, 0.4), transforms.RandomAdjustSharpness(0, 0.4)]),
                transforms.ToTensor(),
            ])

        else:
            self.color_aug = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.ids)

    def Cv2Pil(self,img):
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def Pil2Cv(self,img):
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def random_crop(self, img, mask):
        h_ = img.shape[0]
        w_ = img.shape[1]
        w_random = np.random.randint(w_ - self.W + 1)
        h_random = np.random.randint(h_ - self.H + 1)
        img = img[h_random:(h_random + self.H), w_random:(w_random + self.W)]
        mask = mask[h_random:(h_random + self.H), w_random:(w_random + self.W)]
        return img, mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file=os.path.join(self.mask_dir,name + self.mask_suffix + '.png')
        img_file = os.path.join(self.images_dir, name + '.jpg')

        img = cv2.imread(str(img_file), -1)
        mask = cv2.imread(str(mask_file), -1) / 255

        if self.is_train:
            img,mask=self.random_crop(img,mask)

        img=self.Cv2Pil(img)
        img = self.color_aug(img)

        return {
            'image': img.float().contiguous(),
            'mask': torch.as_tensor(mask).long().contiguous()
        }