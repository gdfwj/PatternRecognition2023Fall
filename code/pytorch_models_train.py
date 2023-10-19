import torch
import torch.nn.functional as F
from dataset import get_dataset, get_one_dataset
from model import CNN, VGG
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch import nn
import numpy as np


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
    torch.manual_seed(2023)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
    )
    _, val_dataset, test_dataset = get_dataset("faces96", transform=transform)
    train_dataset = get_one_dataset("faces96", transform=transform)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(val_dataset, batch_size=32)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    model = CNN(152, 3, image_size=224).to(device)
    # model = nn.Sequential(
    #     Flatten(),
    #     nn.Linear(196 * 196 * 3, 152)
    # ).to(device)
    model.apply(init_normal)
    # model = VGG(3)
    # model.load_state_dict(torch.load("vgg_face_dag.pth"))
    start = 0
    net = nn.Sequential(
        model,
        nn.Linear(2622, 152)
    ).to(device)
    if start!=0:
        net.load_state_dict(torch.load("CNN_best.ckpt"))
    lr = 1e-5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    epoch = 400
    loss_function = nn.CrossEntropyLoss()
    best_acc = 0.0

    for i in range(epoch):
        loss_ = 0.0
        for j, (x, y) in enumerate(train_loader):
            # print(x.shape)
            x = x.to(device)
            # x = torch.flatten(x, 1)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()
            loss_ += loss.item() / x.shape[0]
        loss_ /= len(train_loader)
        print(f"epoch {i}, loss {loss_}")
        with torch.no_grad():
            acc = 0.0
            for x, y in val_loader:
                x = x.to(device)
                # x = torch.flatten(x, 1)
                y = y.to(device)
                # predict = model.predict(x)
                logits = model(x)
                predict = torch.argmax(logits, 1)
                acc += torch.sum(predict == y) / x.shape[0]
            acc /= len(val_loader)
            print(f"epoch {i}, acc {acc.item()}")
            if acc > best_acc:
                best_acc = acc
                print(f"saving model {i}")
                torch.save(model.state_dict(), "CNN_best.ckpt")
