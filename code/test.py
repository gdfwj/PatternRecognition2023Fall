
import torch
import torch.nn as nn
import torch.optim as optim

class SVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

#定义训练数据
x_train = torch.tensor([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
y_train = torch.tensor([3., 1., 2., -1.])

# 定义SVM模型
svm = SVM(input_size=2, num_classes=1)
criterion = nn.HingeEmbeddingLoss()
optimizer = optim.SGD(svm.parameters(), lr=0.01)

#训练模型
num_epochs = 1000
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     outputs = svm(x_train)
#     loss = criterion(outputs.squeeze(), y_train)
#     loss.backward()
#     optimizer.step()

#测试模型
x_test = torch.tensor([[2., 2.], [-2., 2.], [-2., -2.], [2., -2.]])
outputs = svm(x_test)
print(outputs.shape)
predicted = torch.sign(outputs).squeeze()
print(predicted)
