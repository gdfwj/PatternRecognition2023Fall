
import torch
import torch.nn as nn
import torch.optim as optim
from haar_pytorch import HaarForward

x = torch.zeros([64, 1, 128, 128])
h = HaarForward()
y = h(x)
print(y.shape)