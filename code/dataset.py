import os
import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split

from PIL import Image

import json


class FaceDataset(Dataset):
    def __init__(self, folder, transform=None, one=False, type_="train"):
        torch.manual_seed(2023)
        self.data = []  # 图片路径
        self.target = []  # label
        self.class_to_idx = {}  # class name 到编号的映射
        self.transform = transform
        idx = 0
        root = os.path.join("..", "data", folder, folder)
        if folder == "faces94":
            for x in os.listdir(root):
                root_ = os.path.join(root, x)
                for cls in os.listdir(root_):
                    if cls not in self.class_to_idx.keys():
                        self.class_to_idx[cls] = idx
                        idx += 1
                    class_root = os.path.join(root_, cls)
                    for pic in os.listdir(class_root):
                        self.data.append(os.path.join(class_root, pic))
                        self.target.append(cls)
                        if one:
                            break

        elif folder == "faces95" or folder == "faces96" or folder == "grimace":
            root_ = root
            for cls in os.listdir(root_):
                if cls not in self.class_to_idx.keys():
                    self.class_to_idx[cls] = idx
                    idx += 1
                class_root = os.path.join(root_, cls)
                for pic in os.listdir(class_root):
                    self.data.append(os.path.join(class_root, pic))
                    self.target.append(cls)
                    if one:
                        break
                # print(self.data, self.target)
        else:
            raise NotImplementedError
        json.dump(self.class_to_idx, open("class_to_idx.json", "w"))  # 存了一个json保留了编号映射

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, target = self.data[idx], self.class_to_idx[self.target[idx]]
        image = Image.open(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_dataset(name, transform=None):
    return random_split(FaceDataset(name, transform), [0.8, 0.1, 0.1])

def get_one_dataset(name, transform=None):
    return FaceDataset(name, transform, True)

if __name__ == '__main__':
    f = FaceDataset("faces94", None, "val")
    print(len(f))
