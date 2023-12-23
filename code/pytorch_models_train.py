import torch
import torch.nn.functional as F
from dataset import get_dataset, get_one_dataset, get_one_aug_dataset
from model import CNN, VGG
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from haar_pytorch import HaarForward
from moremodel import TDiscriminator


def init_normal(m):
    if type(m) == nn.Linear:
        # y = m.in_features
        # m.weight.data.normal_(0.0,1/np.sqrt(y))
        if 'weight' in m.__dict__.keys():
            m.weight.data.normal_(0.0, 1)
        m.bias.data.fill_(0)
    elif type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 1)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)


if __name__ == '__main__':
    writer = SummaryWriter("ResNet_small")
    torch.manual_seed(2023)
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.2893, 0.3374, 0.4141], [0.0378, 0.0455, 0.0619])
    ]
    )
    val_dataset, val_dataset, test_dataset = get_dataset(transform=transform)
    train_dataset = get_one_aug_dataset(transform=transform)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(val_dataset, batch_size=32)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    # model = CNN(392, 3, image_size=128).to(device)
    # model = resnet50()
    # model.fc = nn.Linear(model.fc.in_features, 392)
    # model = model.to(device)
    model = TDiscriminator(image_size=(3, 128, 128), num_classes=392).to(device)
    model.apply(init_normal)
    # model = nn.Sequential(
    #     Flatten(),
    #     nn.Linear(196 * 196 * 3, 152)
    # ).to(device)
    # model = VGG(3)
    # model = nn.Sequential(
    #     model,
    #     nn.Linear(4096, 152)
    # ).to(device)
    # model = resnet34(392, True).to(device)
    # model.apply(init_normal)

    lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch = 1000
    loss_function = nn.CrossEntropyLoss()
    best_acc = 0.0

    for i in range(epoch):
        loss_ = 0.0
        for j, (x, y) in enumerate(train_loader):
            model.train()
            # print(x.shape)
            x = x.to(device)
            # x = HaarForward()(x)
            # x = torch.flatten(x, 1)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()
            loss_ += loss.item() / x.shape[0]
        loss_ /= len(train_loader)
        writer.add_scalar("loss", loss_, i)
        print(f"epoch {i}, loss {loss_}")
        with torch.no_grad():
            model.eval()
            acc = 0.0
            acc5 = 0.0
            for x, y in val_loader:
                x = x.to(device)
                # x = HaarForward()(x)
                # x = torch.flatten(x, 1)
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
            print(f"epoch {i}, acc {acc.item()}, top5 {acc5.item()}")
            writer.add_scalars("acc", {"top1": acc.item(), "top5": acc5.item()}, i)
            if acc > best_acc:
                best_acc = acc
                print(f"saving model {i}")
                torch.save(model.state_dict(), "ResNet_small.ckpt")
    print(f"best_acc: {best_acc}")
