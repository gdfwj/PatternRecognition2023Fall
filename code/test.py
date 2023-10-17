import torch
import torch.nn.functional as F
from dataset import get_dataset
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
    train_dataset, val_dataset, test_dataset = get_dataset("faces96", transform=transform)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=600)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = DataLoader(val_dataset, batch_size=32)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    # model = nn.Sequential(
    #     Flatten(),
    #     nn.Linear(196 * 196 * 3, 152)
    # ).to(device)
    # model = CNN(152, 3, image_size=196).to(device)
    model = VGG(3).to(device)
    model.load_state_dict(torch.load("VGG_best_model.ckpt"))
    # for _, param in enumerate(model.named_parameters()):
    #     print(param[0])
    #     print('----------------')

    with torch.no_grad():
        acc = 0.0
        for x, y in test_loader:
            x = x.to(device)
            # x = torch.flatten(x, 1)
            y = y.to(device)
            # predict = model.predict(x)
            logits = model(x)
            predict = torch.argmax(logits, 1)
            acc += torch.sum(predict == y) / x.shape[0]
        acc /= len(test_loader)
        print(f"test acc {acc.item()}")
