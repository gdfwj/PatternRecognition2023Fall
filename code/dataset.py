import os
import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
from torchvision import transforms

from PIL import Image

import json
from haar_pytorch import HaarForward
from sklearn.decomposition import PCA


class FaceDataset(Dataset):
    def __init__(self, folders=['faces94', 'faces95', 'faces96', 'grimace'], transform=None, one=False, el=False,
                 haar=False, crop=False):
        self.haar = haar
        torch.manual_seed(2023)
        self.data = []  # 图片路径
        self.target = []  # label
        self.class_to_idx = {}  # class name 到编号的映射
        self.transform = transform
        idx = 0
        for folder in folders:
            if crop:
                root = os.path.join("..", "cropped_data", folder, folder)
            else:
                root = os.path.join("..", "data", folder, folder)
            if folder == "faces94":
                for x in os.listdir(root):
                    root_ = os.path.join(root, x)
                    for cls in os.listdir(root_):
                        dup = False
                        if cls not in self.class_to_idx.keys():
                            self.class_to_idx[cls] = idx
                        else:
                            dup = True
                            self.class_to_idx[cls+"1"] = idx
                        idx += 1
                        class_root = os.path.join(root_, cls)
                        flag = True
                        for pic in os.listdir(class_root):
                            if flag and el:
                                flag = False
                                continue
                            self.data.append(os.path.join(class_root, pic))
                            if dup:
                                self.target.append(cls+"1")
                            else:
                                self.target.append(cls)
                            if one:
                                break

            elif folder == "faces95" or folder == "faces96" or folder == "grimace":
                root_ = root
                for cls in os.listdir(root_):
                    dup=False
                    if cls not in self.class_to_idx.keys():
                        self.class_to_idx[cls] = idx
                    else:
                        dup=True
                        self.class_to_idx[cls + "1"] = idx
                    idx += 1
                    class_root = os.path.join(root_, cls)
                    flag = True
                    for pic in os.listdir(class_root):
                        if flag and el:
                            flag = False
                            continue
                        self.data.append(os.path.join(class_root, pic))
                        if dup:
                            self.target.append(cls+"1")
                        else:
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
        if self.haar:
            image = HaarForward()(image.unsqueeze(0)).squeeze(0)
        return image, target

    def getpath(self):
        return self.data


class AugmentedDataset(Dataset):  # 增加旋转和翻折，0原图，1水平翻转
    def __init__(self, folders=['faces94', 'faces95', 'faces96', 'grimace'], transform=None, one=False, el=False,
                 haar=False, crop=False):
        self.origin_dataset = FaceDataset(folders, transform, one, el, haar, crop)

    def __len__(self):
        return len(self.origin_dataset) * 3

    def __getitem__(self, idx):
        image, target = self.origin_dataset[idx // 3]
        if idx % 3 == 1:
            image = transforms.RandomHorizontalFlip(1)(image)
        if idx % 3 == 2:
            image = image + torch.randn(*image.shape)
        # image = image + torch.randn(*image.shape)
        return image, target


class PairDataset(FaceDataset):
    def __init__(self, folder, transform=None, one=False, haar=False):
        super(PairDataset, self).__init__(folder, transform, one, haar=haar)
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
        if self.haar:
            x_i = HaarForward()(x_i.unsqueeze(0)).squeeze(0)
            x_j = HaarForward()(x_j.unsqueeze(0)).squeeze(0)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return x_i, x_j, target


def get_pair_dataset(name=['faces94', 'faces95', 'faces96', 'grimace'], transform=None, one=False, haar=False):
    return PairDataset(name, transform, one=one, haar=haar), PairDataset(name, transform, one=False, haar=haar)


def get_dataset(name=['faces94', 'faces95', 'faces96', 'grimace'], transform=None):
    return random_split(FaceDataset(name, transform), [0.8, 0.1, 0.1])


def get_one_dataset(name=['faces94', 'faces95', 'faces96', 'grimace'], transform=None, haar=False, crop=False):
    return FaceDataset(name, transform, True, haar=haar, crop=crop), FaceDataset(name, transform, False, True,
                                                                                 haar=haar, crop=crop)


def get_all(name=['faces94', 'faces95', 'faces96', 'grimace'], transform=None, haar=False, crop=False):
    return FaceDataset(folders=name, transform=transform, haar=haar, crop=crop)


def get_one_aug_dataset(name=['faces94', 'faces95', 'faces96', 'grimace'], transform=None, haar=False, crop=False):
    return AugmentedDataset(name, transform, True, haar=haar, crop=crop), FaceDataset(name, transform, False, True,
                                                                                 haar=haar, crop=crop)


class PCADataset(Dataset):
    def __init__(self, name1, train=False):
        import numpy as np
        x = np.load(name1)
        y = np.load("y_PCA.npy")
        self.x = []
        self.y = []
        flag = np.zeros([1000])
        for img, label in zip(x, y):
            if train:
                if flag[label] == 0:
                    self.x.append(img)
                    self.y.append(label)
                    flag[label] = 1
            else:
                if flag[label] == 0:
                    flag[label] = 1
                else:
                    self.x.append(img)
                    self.y.append(label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from haar_pytorch import HaarForward, HaarInverse
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    )
    train_dataset = FaceDataset(one=True)
    print(len(train_dataset))
    import numpy as np

    count = np.zeros([400])
    for i, j in train_dataset:
        if count[j] != 0:
            print(j)
        count[j] = 1
    # for i, j in enumerate(count):
    #     if j == 0:
    #         print(i)
    # img = train_dataset[0][0].unsqueeze(0)
    # vis = img.squeeze()
    # vis = (vis - torch.min(vis)) / (torch.max(vis) - torch.min(vis))
    # # print(1)
    # vis = vis.permute(1, 2, 0)
    # # print(2)
    # plt.imshow(vis.detach().squeeze().cpu().numpy())
    # plt.show()
    # plt.close()
    #
    # vis = HaarInverse()(HaarForward()(img)).squeeze()
    # vis = (vis - torch.min(vis)) / (torch.max(vis) - torch.min(vis))
    # # print(1)
    # vis = vis.permute(1, 2, 0)
    # # print(2)
    # plt.imshow(vis.detach().squeeze().cpu().numpy())
    # plt.show()
    # plt.close()
