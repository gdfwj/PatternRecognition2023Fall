import torch
import torch.nn.functional as F
from dataset import *
from model import CNN, VGG
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch import nn
import numpy as np
from PIL import Image


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
        transforms.Resize((64, 64)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.2893, 0.3374, 0.4141], [0.0378, 0.0455, 0.0619])
    ]
    )
    thresholds = [0.01,
                  0.05,
                  0.1,
                  0.5,
                  1,
                  5,
                  10,
                  50,
                  100,
                  150,
                  160,
                  170,
                  180,
                  190,
                  200,
                  250,
                  300,
                  350,
                  500,
                  1000,
                  5000]
    for threshold in thresholds:
        train_dataset = get_all(name=['faces94', 'faces95', 'faces96'], transform=transform, haar=True)
        print(len(train_dataset))
        train_loader = DataLoader(train_dataset, len(train_dataset))
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = "cpu"

        # model = CNN(152, 3, image_size=196).to(device)
        # model = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(196 * 196 * 3, 152)
        # ).to(device)
        # model.apply(init_normal)
        # model = VGG(3)
        # model.load_state_dict(torch.load("vgg_face_dag.pth"))
        from moremodel import TDiscriminator

        # model = VGG(3, 392)
        # model.load_state_dict(torch.load("vgg_no/
        model = CNN(392, 12, image_size=32).to(device)
        model.load_state_dict(torch.load("cnn_big/best.ckpt"))
        model.to(device)
        # model = VGG(3)
        # model.load_state_dict(torch.load("CNN_no/best.ckpt"))
        # model.to(device)
        from register import RegisterHelper

        register = RegisterHelper(model)
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            register.register_pre_train(x, y)
        test_set = get_all(name=['grimace'], transform=transform, haar=True)
        register_dict = {}
        FAR = 0
        FRR = 0
        count = 0
        TP = 0
        FN = 0
        for x, y in test_set:
            if y not in register_dict.keys():
                register_dict[y] = x
        for now_register in list(register_dict.keys()):
            # test all
            for x, y in test_set:  # all test data
                x = x.to(device)
                # y = y.to(device)
                pred = register.registered_predict(x, threshold)
                if y in register_dict.keys():  # y not registered
                    if pred != -1:  # FAR
                        FAR += 1
                    else:
                        FN += 1
                else:  # y registered
                    if pred == -1:
                        FRR += 1
                    elif pred == y:
                        TP += 1
                    count += 1
            register.register(register_dict[now_register].to(device), now_register)
            register_dict.pop(now_register)
        print(TP, FAR, FRR, FN, count)
        print(f"threshold: {threshold}, FAR:{FAR / (FAR + FN)}, FRR:{FRR / count}")
