import torch
from dataset import get_dataset, get_one_dataset, FaceDataset
from model import GaussianDistribution, Perception, SVM
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import numpy as np
from haar_pytorch import HaarForward

if __name__ == '__main__':
    torch.manual_seed(2023)
    transforms = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ]
    )
    _, val_dataset, test_dataset = get_dataset(transform=transforms)
    train_dataset = get_one_dataset(transform=transforms)
    # all_dataset = FaceDataset("faces96", transforms)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = DataLoader(val_dataset, batch_size=len(test_dataset))
    model = SVM(394)
    # model = Perception(392, 12*64*64, 1e-5)
    # model = GaussianDistribution(392)
    acc_all = []
    for svm in range(5):

        for x, y in train_loader:
            # x = HaarForward()(x)
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
            # x = HaarForward()(x)
            x = np.array(x)
            y = np.array(y)
            y_pred = model.predict(x, y)
            acc = np.mean(y_pred == y)
            print(acc)
            acc_all.append(acc)
            # y_top5 = model.predict_top5(x)
            # # print(y_top5.shape)
            # acc5 = 0.0
            # for i in range(y_top5.shape[0]):
            #     if y[i] in y_top5[i]:
            #         acc5 += 1
            # print(acc5 / len(y_top5))
    print(acc_all)