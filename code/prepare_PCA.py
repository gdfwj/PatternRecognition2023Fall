import torch
from dataset import *
from model import GaussianDistribution, Perception
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import numpy as np
from haar_pytorch import HaarForward

if __name__ == '__main__':
    torch.manual_seed(2023)
    transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.2893, 0.3374, 0.4141], [0.0378, 0.0455, 0.0619])
    ]
    )
    train_dataset= get_all(transform=transforms, haar=False, crop=False)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    for x, y in train_loader:
        from sklearn.decomposition import PCA
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
        x = x.reshape(x.shape[0], -1)
        pca = PCA(n_components=0.9)
        x = pca.fit_transform(x)
        x = pca.inverse_transform(x)
        print(x.shape)
        np.save("x_PCA_crop.npy", x)
        np.save("y_PCA.npy", y)
