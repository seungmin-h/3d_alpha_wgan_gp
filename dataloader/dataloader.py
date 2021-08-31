#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : dataloader
# @Date : 2021-08-30-08-30
# @Project : 3d_alpha_wgan_gp
# @Author : seungmin

import glob

import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from skimage.transform import resize

cube_len = 128 #64

class IXIDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.img_paths[idx]
        img = nib.load(img_name).get_fdata()

        if self.transform:
            img = self.transform(img)

        return img

class Centering(object):

    def __init__(self):
        self.max_len = 256

    def __call__(self, img):
        min_len = img.shape[-1]
        offset = (self.max_len - min_len) // 2
        pad = np.zeros((self.max_len, self.max_len, self.max_len))
        pad[:, :, offset:self.max_len - offset] = img
        img = resize(pad, (cube_len, cube_len, cube_len))
        return img

class Normalize(object):

    def __init__(self, minv=-1, maxv=1):
        self.minv = minv
        self.maxv = maxv

    def __call__(self, img):
        mina = np.min(img)
        maxa = np.max(img)
        if mina == maxa:
            return img * self.minv
        norm = (img - mina) / (maxa - mina)
        img = (norm * (self.maxv - self.minv)) + self.minv
        return img

class ToTensor(object):
    def __call__(self, img):
        img = torch.from_numpy(img).float()
        img = torch.unsqueeze(img, 0)
        return img

class DatasetWrapper(object):

    def __init__(self, batch_size, num_workers, data_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path = data_path

    def _get_transform(self):
        return transforms.Compose([Centering(),
                                   Normalize(),
                                   ToTensor()])

    def _get_data_loader(self):
        composed = self._get_transform()

        dataset = IXIDataset(glob.glob(self.path), transform=composed)

        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=True,
                                 drop_last=True)

        return data_loader