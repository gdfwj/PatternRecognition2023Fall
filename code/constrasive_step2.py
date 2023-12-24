import torch
import torch.nn.functional as F
from dataset import get_dataset, get_one_dataset, get_one_aug_dataset, get_pair_dataset
from model import CNN, VGG, simModel
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch import nn
import numpy as np
import os
from constrasive import *
from tqdm import tqdm


def init_normal(m):
    if type(m) == nn.Linear:
        # y = m.in_features
        # m.weight.data.normal_(0.0,1/np.sqrt(y))
        if 'weight' in m.__dict__.keys():
            m.weight.data.normal_(0.0, 1)
        m.bias.data.fill_(0)
    elif type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 1)


if __name__ == '__main__':
    folder = "simCLR_yes"
    device = "cuda:0"
    best_simModel = simModel(512, 12)
    best_simModel.load_state_dict(torch.load(f'./{folder}/trained_simclr_model.pth'))
    best_simModel.to(device)
    torch.manual_seed(2023)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.2893, 0.3374, 0.4141], [0.0378, 0.0455, 0.0619])
    ]
    )
    lr = 1e-4
    best_acc=0.0
    model = nn.Linear(2048, 392).to(device)
    model.apply(init_normal)
    fintune_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_function = nn.CrossEntropyLoss()
    train_dataset, test_dataset = get_one_dataset(transform=transform, haar=True)
    print(len(train_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=16)
    for epoch in range(1, fintune_epochs + 1):
        loss_ = 0.0
        for j, (x, y) in enumerate(train_loader):
            model.train()
            # print(x.shape)
            x = x.to(device)
            # x = HaarForward()(x)
            # x = torch.flatten(x, 1)
            y = y.to(device)
            x, _ = best_simModel(x)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()
            loss_ += loss.item() / x.shape[0]
        loss_ /= len(train_loader)
        print(f"epoch {epoch}, loss {loss_}")
        with torch.no_grad():
            model.eval()
            acc = 0.0
            acc5 = 0.0
            for x, y in val_loader:
                x = x.to(device)
                # x = HaarForward()(x)
                # x = torch.flatten(x, 1)
                x, _ = best_simModel(x)
                y = y.to(device)
                # predict = model.predict(x)
                logits = model(x)
                predict = torch.argmax(logits, 1)
                acc += torch.sum(predict == y) / x.shape[0]
                predict5 = torch.topk(logits, 5, 1).indices
                acc5 += torch.sum(predict5.T == y) / x.shape[0]
                # for i in range(predict5.shape[0]):
                #     if y[i] in predict5:
                #         acc5 += 1
            acc /= len(val_loader)
            acc5 /= len(val_loader)
            print(f"epoch {epoch}, acc {acc.item()}, top5 {acc5.item()}")
            if acc > best_acc:
                best_acc = acc
                print(f"saving model {epoch}")
                torch.save(model.state_dict(), "finetune.ckpt")
