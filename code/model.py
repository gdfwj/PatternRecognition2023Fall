import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class GaussianDistribution:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.class_pos = None

    def train(self, x, y):
        self.class_pos = np.zeros([self.num_classes, *x[0].shape])
        for i in range(y.shape[0]):
            self.class_pos[y[i]] += x[i]

    def predict(self, x):
        out = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            dis = np.mean((x[i] - self.class_pos) ** 2, axis=(1, 2, 3))
            out[i] = np.argmin(dis)
        return out


class Perception:
    def __init__(self, num_classes, dim, lr):
        self.w = np.zeros([dim, num_classes])
        self.b = np.zeros([num_classes])
        self.lr = lr

    def train(self, x, y):
        x = x.reshape(x.shape[0], -1)
        count = 0
        while True:
            flag = True
            count += 1
            err = 0.0
            for i, j in zip(x, y):
                logit = i @ self.w + self.b  # n, cls
                for k in range(len(logit)):
                    if k == j:
                        if logit[k] <= 0:
                            flag = False
                            err += 1
                            self.w[:, k] += i
                            self.b[k] += 1
                    else:
                        if logit[k] > 0:
                            flag = False
                            self.w[:, k] -= i
                            self.b[k] -= 1
            if flag or count > 100:
                break
            print(f"cycle{count}, err{err / (x.shape[0])}")

    def predict(self, x):
        x = x.reshape(x.shape[0], -1)
        return np.argmax(x @ self.w + self.b, axis=1)


class CNN(nn.Module):
    def __init__(self, num_classes, in_channels=3, image_size=128):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear((image_size//8)**2*64, 120),  # 这里把第三个卷积当作是全连接层了
            nn.Linear(120, 84),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # 输出 16*5*5 特征图
        # print(x.shape)
        x = torch.flatten(x, 1)  # 展平 （1， 16*5*5）
        logits = self.classifier(x)  # 输出 10
        return logits

    def predict(self, x):
        x = self.features(x)  # 输出 16*5*5 特征图
        x = torch.flatten(x, 1)  # 展平 （1， 16*5*5）
        logits = self.classifier(x)  # 输出 10
        out = torch.argmax(logits, 1)
        return out


class VGG(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channel, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.max_pool = nn.MaxPool2d(2, 2, 0)
        self.relu = nn.ReLU()
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        return x