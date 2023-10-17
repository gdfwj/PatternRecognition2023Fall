import torch
import torch.nn.functional as F
from dataset import get_dataset
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
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
    )
    train_dataset, val_dataset, test_dataset = get_dataset("faces96", transform=transform)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, len(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(val_dataset, batch_size=32)
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
    model = VGG(3)
    model.load_state_dict(torch.load("vgg_face_dag.pth"))
    from register import RegisterHelper
    register = RegisterHelper(model)
    for x, y in train_loader:
        register.register_pre_train(x, y)
        break
    print("finished register train")
    picture = transform(Image.open("jer_2.1.jpg").convert('RGB'))
    cls = "jer"
    register.register(picture, cls)
    test = transform(Image.open("jer_2.13.jpg").convert('RGB'))
    print(register.registered_predict(test))


