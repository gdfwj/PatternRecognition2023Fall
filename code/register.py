import json

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class RegisterHelper:
    def __init__(self, model, class_dict_path="class_to_idx.json"):
        self.model = model
        self.model.eval()
        self.model.train(False)
        self.class_dict = json.load(open(class_dict_path))
        self.idx2name = {}
        for i, j in self.class_dict.items():
            self.idx2name[j] = i
        self.registered = None

    def register_pre_train(self, x, y, latent_dim=128):
        if x.get_device() > 0:
            self.registered = torch.zeros([len(self.class_dict), latent_dim]).to(x.get_device())
        else:
            self.registered = torch.zeros([len(self.class_dict), latent_dim])
        count = torch.zeros([len(self.class_dict)])
        count_all = 0
        for i in range(x.shape[0]):
            if count[y[i]] == 0:
                count[y[i]] += 1
                count_all += 1
                self.registered[y[i]] += self.model.represent(x[i].unsqueeze(0)).squeeze(0)
                if count_all == len(self.class_dict):  # choose one picture in per class
                    break

    def register(self, x, y):
        if self.registered is None:
            raise NotImplementedError
        latent = self.model.represent(x.unsqueeze(0))
        # print(latent.shape)
        self.registered = torch.cat((self.registered, latent), 0)
        # print(len(self.class_dict))
        self.class_dict[y] = len(self.class_dict)
        # print(len(self.class_dict))
        self.idx2name[len(self.class_dict) - 1] = y

    def registered_predict(self, x, threshold):
        # print(self.idx2name)
        latent = self.model.represent(x.unsqueeze(0)).squeeze(0)
        dis = torch.sum((self.registered - latent) ** 2, dim=1)
        # print(dis.shape)
        dis, idx = torch.min(dis, 0)
        # print(dis)
        if dis > threshold:
            return -1
        # print(f"face is class: {self.idx2name[idx]}")
        return self.idx2name[idx.item()]
