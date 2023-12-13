import torch
import torch.nn.functional as F
from dataset import get_dataset, get_one_dataset
from model import CNN, VGG
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch import nn
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)


if __name__ == '__main__':
    torch.manual_seed(2023)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
    )
    # x = torch.ones([32, 100])
    # y = torch.ones([32])
    # print((x.T / y).shape)
    train_dataset = get_one_dataset("faces96", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    for batch, _ in train_loader:  # calculate paras of normalization
        print(batch.shape)
        mean = torch.mean(batch, dim=[0, 2, 3])
        std = torch.var(batch, dim=[0, 2, 3])
        print(mean)
        print(std)
