import os
import glob
import json
import torch
import random
import torch.nn as nn
import numpy as np
import torch.utils.data
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
import albumentations as A
from sklearn.model_selection import KFold
from scipy.ndimage.interpolation import zoom
from PIL import Image


def norm01(x):
    return np.clip(x, 0, 255) / 255


class BaseDataSets(data.Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        p = 0.5
        self.transf = A.Compose([
            A.GaussNoise(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(p=p),
        ])
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            image = Image.open(os.path.join(self._base_dir, "Image", self.sample_list[idx])).resize((224, 224))
            label = Image.open(os.path.join(self._base_dir, "Label", self.sample_list[idx])).convert('L').resize((224, 224))
        else:
            image = Image.open(os.path.join(self._base_dir, "Image", self.sample_list[idx])).resize((224, 224))
            label = Image.open(os.path.join(self._base_dir, "Label", self.sample_list[idx])).convert('L').resize((224, 224))

        image = np.array(image)
        label = np.array(label)
        label = np.where(label == 255, 1, 0)
        sample = {"image": image, "label": label}

        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                tsf = self.transf(image=sample["image"].astype('uint8'), mask=sample["label"])
                sample["image"], sample["label"] = torch.from_numpy(norm01(tsf['image'])).float().permute(2, 0, 1), torch.from_numpy(tsf['mask']).float()
            else:
                tsf = self.transf(image=sample["image"].astype('uint8'), mask=sample["label"])
                sample["image"], sample["label"] = torch.from_numpy(norm01(tsf['image'])).float().permute(2, 0, 1), torch.from_numpy(tsf['mask']).float()
        else:
            sample["image"], sample["label"] = torch.from_numpy(norm01(image)).float().permute(2, 0, 1), torch.from_numpy(label)
        sample["idx"] = idx
        return sample

