import json

from time import time
import matplotlib.pyplot as plt
from scipy.stats import loguniform
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SVM:
    def __init__(self, num_classes, h=128, w=128):
        self.num_classes = num_classes
        self.class_pos = None
        self.eigenfaces = None
        self.pca = None
        self.clf = None
        self.h = h
        self.w = w

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.reshape(gray.shape[0], -1)

    def fit4clf_cross_validation(self, X, y, dup_times=5):
        X_dup = []
        y_dup = []
        for idx in range(X.shape[0]):
            for i in range(dup_times):
                X_dup.append(X[idx])
                y_dup.append(y[idx])
        return torch.FloatTensor(np.array(X_dup)), y_dup

    def train(self, X_train, y_train, n_components=150):
        # introspect the images arrays to find the shapes (for plotting)
        X_train = self.rgb2gray(X_train)
        print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver="randomized", whiten=False).fit(X_train)
        print("done in %0.3fs" % (time() - t0))
        self.pca = pca
        self.eigenfaces = pca.components_.reshape((n_components, self.h, self.w))

        X_train_pca = pca.transform(X_train)
        # X_test_pca = pca.transform(X_test)
        # Train a SVM classification model

        print("Fitting the classifier to the training set")
        t0 = time()
        param_grid = {
            "C": loguniform(1e3, 1e4),
            "gamma": loguniform(1e-4, 1e-2),
        }
        clf = RandomizedSearchCV(
            SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=50
        )
        X_train_pca_dup, y_train_dup = self.fit4clf_cross_validation(X_train_pca, y_train)
        self.clf = clf.fit(X_train_pca_dup, y_train_dup)
        print("done in %0.3fs" % (time() - t0))
        print("Best estimator found by grid search:")
        print(self.clf.best_estimator_)

    def predict(self, X_test, y_test):
        X_test_pca = self.pca.transform(self.rgb2gray(X_test))
        y_pred = self.clf.predict(X_test_pca)
        ConfusionMatrixDisplay.from_estimator(
            self.clf, X_test_pca, y_test, xticks_rotation="vertical"
        )
        plt.tight_layout()
        plt.show()
        return y_pred

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    def show_eigenfaces(self):
        eigenface_titles = ["eigenface %d" % i for i in range(self.eigenfaces.shape[0])]
        _, h, w = self.eigenfaces.shape
        self.plot_gallery(self.eigenfaces, eigenface_titles, h, w)
        plt.show()


class GaussianDistribution:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.class_pos = None

    def train(self, x, y):
        self.class_pos = np.zeros([self.num_classes, *x[0].shape])
        count = np.zeros(self.num_classes)
        for i in range(y.shape[0]):
            self.class_pos[y[i]] += x[i]
            count[y[i]] += 1
        for i in range(self.num_classes):
            if count[i] > 0:
                self.class_pos[i] = self.class_pos[i] / count[i]
            else:
                print(i)

    def predict(self, x):
        out = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            # print(i)
            dis = np.sum((x[i] - self.class_pos) ** 2, axis=(1, 2, 3))
            out[i] = np.argmin(dis)
        return out

    def predict_top5(self, x):
        out = np.zeros([x.shape[0], 5])
        for i in range(x.shape[0]):
            # print(i)
            dis = np.sum((x[i] - self.class_pos) ** 2, axis=(1, 2, 3))
            tdis = torch.tensor(dis)
            top5 = torch.topk(-tdis, 5)
            out[i] = top5.indices.detach().numpy()
        return out


class Perception:
    def __init__(self, num_classes, dim, lr):
        self.w = np.zeros([dim, num_classes])
        self.b = np.zeros([num_classes])
        self.lr = lr

    def train(self, x, y):
        # print(x.shape)
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
            # print(f"cycle{count}, err{err / (x.shape[0])}")

    def predict(self, x):
        x = x.reshape(x.shape[0], -1)
        return np.argmax(x @ self.w + self.b, axis=1)

    def predict_top5(self, x):
        x = x.reshape(x.shape[0], -1)
        logits = torch.tensor(x @ self.w + self.b)
        top5 = torch.topk(logits, 5)
        return top5.indices.detach().numpy()


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
            nn.Linear((image_size // 8) ** 2 * 64, 128)
        )

        self.out = nn.Sequential(
            nn.Linear(128, 84),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # 输出 16*5*5 特征图
        # print(x.shape)
        x = torch.flatten(x, 1)  # 展平 （1， 16*5*5）
        x = self.classifier(x)
        logits = self.out(x)  # 输出 num_classes
        return logits


    def represent(self, x):
        x = self.features(x)  # 输出 16*5*5 特征图
        # print(x.shape)
        x = torch.flatten(x, 1)  # 展平 （1， 16*5*5）
        x = self.classifier(x)
        return x


class VGG(nn.Module):
    def __init__(self, in_channel, num_classes):
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
        self.fc9 = nn.Linear(in_features=4096, out_features=128, bias=True)
        self.fc10 = nn.Linear(in_features=128, out_features=num_classes)

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
        x = self.relu(x)
        x = self.fc9(x)
        x = self.relu(x)
        x = self.fc10(x)
        return x

    def represent(self, x):
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
        x = self.relu(x)
        x = self.fc9(x)
        return x


class simModel(nn.Module):
    def __init__(self, feature_dim=128, channel=3):
        super(simModel, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        # print("feature:", feature.shape)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


if __name__ == '__main__':
    vgg = VGG(3)
    vgg.register_train(1, 2)
