import torch
import torch.nn.functional as F
from dataset import get_dataset, get_one_dataset, get_one_aug_dataset, get_pair_dataset
from model import CNN, VGG, simModel
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch import nn
import numpy as np
import os
from constrasive import *
from tqdm import tqdm


def init_normal(m):
    if type(m) == nn.Linear:
        # y = m.in_features
        # m.weight.data.normal_(0.0,1/np.sqrt(y))
        if 'weight' in m.__dict__.keys():
            m.weight.data.normal_(0.0, 1)
        m.bias.data.fill_(0)
    elif type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 1)


def train(model, data_loader, train_optimizer, epoch, epochs, batch_size=32, temperature=0.5, device='cuda'):
    """Trains the model defined in ./model.py with one epoch.

    Inputs:
    - model: Model class object as defined in ./model.py.
    - data_loader: torch.utils.data.DataLoader object; loads in training data. You can assume the loaded data has been augmented.
    - train_optimizer: torch.optim.Optimizer object; applies an optimizer to training.
    - epoch: integer; current epoch number.
    - epochs: integer; total number of epochs.
    - batch_size: Number of training samples per batch.
    - temperature: float; temperature (tau) parameter used in simclr_loss_vectorized.
    - device: the device name to define torch tensors.

    Returns:
    - The average loss.
    """
    model.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data_pair in train_bar:
        x_i, x_j, target = data_pair
        x_i, x_j = x_i.to(device), x_j.to(device)

        out_left, out_right, loss = None, None, None
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Take a look at the model.py file to understand the model's input and output.
        # Run x_i and x_j through the model to get out_left, out_right.              #
        # Then compute the loss using simclr_loss_vectorized.                        #
        ##############################################################################
        _, out_left = model(x_i)
        _, out_right = model(x_j)
        loss = simclr_loss_vectorized(out_left, out_right, temperature)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


def test(model, memory_data_loader, test_data_loader, epoch, epochs, c, temperature=0.5, k=200, device='cuda'):
    model.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = model(data.to(device))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.target, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature, out = model(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)

            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    torch.manual_seed(2023)
    img_size = 56
    batch_size = 151
    transform = transforms.Compose([
        transforms.Resize(img_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.2893, 0.3374, 0.4141], [0.0378, 0.0455, 0.0619])
    ]
    )
    _, val_dataset, test_dataset = get_dataset("faces96", transform=transform)
    train_dataset, val_dataset = get_pair_dataset("faces96", transform=transform)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(val_dataset, batch_size=batch_size)
    if torch.cuda.is_available():
        device = torch.device("cuda:6")
    else:
        device = "cpu"

    model = simModel(512).to(device)
    # model.apply(init_normal)
    lr = 1e-5
    epochs = 20
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}  # << -- output
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    temperature = 0.5
    k = 200
    best_acc = 0.0
    c = 152
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('pretrained_model'):
        os.mkdir('pretrained_model')

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, epochs, batch_size=batch_size,
                           temperature=temperature, device=device)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, val_loader, test_loader, epoch, epochs, c, k=k, temperature=temperature,
                                      device=device)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), './pretrained_model/trained_simclr_model.pth')
    torch.save(results, f'./results/batch_{batch_size}_size_{img_size}.txt')
