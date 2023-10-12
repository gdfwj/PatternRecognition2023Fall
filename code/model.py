import torch
import numpy as np


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
        count=0
        while True:
            flag = True
            count+=1
            err = 0.0
            for i, j in zip(x, y):
                logit = i @ self.w + self.b  # n, cls
                for k in range(len(logit)):
                    if k==j:
                        if logit[k]<=0:
                            flag = False
                            err += 1
                            self.w[:,k] += i
                            self.b[k] += 1
                    else:
                        if logit[k]>0:
                            flag = False
                            self.w[:, k] -= i
                            self.b[k] -= 1
            if flag or count>100:
                break
            print(f"cycle{count}, err{err/(x.shape[0])}")

    def predict(self, x):
        x = x.reshape(x.shape[0], -1)
        return np.argmax(x @ self.w + self.b, axis=1)