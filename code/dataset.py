import os
import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
from torchvision import transforms

from PIL import Image

import json


class FaceDataset(Dataset):
    def __init__(self, folders=['faces94', 'faces95', 'faces96', 'grimace'], transform=None, one=False):
        torch.manual_seed(2023)
        self.data = []  # 图片路径
        self.target = []  # label
        self.class_to_idx = {}  # class name 到编号的映射
        self.transform = transform
        idx = 0
        for folder in folders:
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


class AugmentedDataset(Dataset):  # 增加旋转和翻折，0原图，1水平翻转
    def __init__(self, folder, transform=None, type_="train"):
        self.origin_dataset = get_one_dataset(folder, transform)

    def __len__(self):
        return len(self.origin_dataset) * 2

    def __getitem__(self, idx):
        image, target = self.origin_dataset[idx // 2]
        if idx % 8 == 1:
            image = transforms.RandomHorizontalFlip(1)(image)
        # image = image + torch.randn(*image.shape)
        return image, target


class PairDataset(FaceDataset):
    def __init__(self, folder, transform=None, one=False):
        super(PairDataset, self).__init__(folder, transform, one)
        temp = []
        for i in self.target:
            temp.append(self.class_to_idx[i])
        self.target = temp

    def __getitem__(self, idx):
        img, target = self.data[idx], self.target[idx]
        # img = Image.fromarray(img)
        img = Image.open(img).convert('RGB')

        x_i = None
        x_j = None

        if self.transform is not None:
            x_i = self.transform(img)
            x_j = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return x_i, x_j, target


def get_pair_dataset(name=['faces94', 'faces95', 'faces96', 'grimace'], transform=None):
    return PairDataset(name, transform, True), PairDataset(name, transform)


def get_dataset(name=['faces94', 'faces95', 'faces96', 'grimace'], transform=None):
    return random_split(FaceDataset(name, transform), [0.8, 0.1, 0.1])


def get_one_dataset(name=['faces94', 'faces95', 'faces96', 'grimace'], transform=None):
    return FaceDataset(name, transform, True)


def get_one_aug_dataset(name=['faces94', 'faces95', 'faces96', 'grimace'], transform=None):
    return AugmentedDataset(name, transform)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from haar_pytorch import HaarForward
    transform = transforms.Compose([
        transforms.Resize(128),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.2893, 0.3374, 0.4141], [0.0378, 0.0455, 0.0619])
    ]
    )
    train_dataset = get_one_aug_dataset(transform=transform)
    # x = train_dataset[0][0].unsqueeze(0)
    # x = HaarForward()(x)
    print(len(train_dataset))

