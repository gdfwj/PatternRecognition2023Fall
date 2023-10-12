import torch
from dataset import get_dataset
from model import GaussianDistribution, Perception
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import numpy as np

if __name__ == '__main__':
    torch.manual_seed(2023)
    transforms = transforms.Compose([
        transforms.ToTensor(),
    ]
    )
    train_dataset, val_dataset, test_dataset = get_dataset("faces96", transform=transforms)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = DataLoader(val_dataset, batch_size=len(test_dataset))
    model = Perception(152, 3*196*196, 1e-5)
    for x, y in train_loader:
        # count=[]
        # for i in range(152):
        #     count.append(0)
        # for i in y:
        #     i = int(i)
        #     count[i] += 1
        # print(count)
        print(x.shape)
        x = np.array(x)
        y = np.array(y)
        model.train(x, y)
    for x, y in val_loader:
        x = np.array(x)
        y = np.array(y)
        y_pred = model.predict(x)
        acc = np.mean(y_pred == y)
        print(acc)
