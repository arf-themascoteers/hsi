import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cnn


def test():
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

    model = cnn.SimpleCNN()
    model.eval()
    model.load_state_dict(torch.load("models/cnn.h5"))
    correct = 0
    total = 0

    for (x, y) in data_loader:
        y_hat = model(x)
        pred = torch.argmax(y_hat, dim=1, keepdim=True)
        correct += pred.eq(y.data.view_as(pred)).sum()
        total += x.shape[0]
        break

    print(f'Total:{total}, Correct:{correct}')