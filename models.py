import torch
import torch.nn as nn
import torch.nn.functional as F

class CONV(nn.Module):
    def __init__(self):
        super(CONV, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=5)
        self.fc1 = nn.Linear(7680, 1000)
        self.fc2 = nn.Linear(1000, 784)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7680)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)*255
        return x