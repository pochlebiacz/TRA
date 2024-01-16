import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models import *

def data_prep(noise_lambda=64, batch_size_train=512, batch_size_test=512):
    trainset = torchvision.datasets.MNIST('./data/', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor()]))
    testset = torchvision.datasets.MNIST('./data/', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor()]))

    # -------------- Adding gaussian noise to the training set ---------------
    trainset_norm = trainset.data 
    testset_norm = testset.data

    trainset_noise = trainset_norm + torch.randn(trainset_norm.size()) * noise_lambda
    testset_noise = testset_norm + torch.randn(testset_norm.size()) * noise_lambda

    # -------------- Squeezing augmented data to [0, 255] ---------------
    trainset_noise = torch.clamp(trainset_noise, 0, 255)
    testset_noise = torch.clamp(testset_noise, 0, 255)

    train_dataset_full = torch.utils.data.TensorDataset(trainset_noise.float(), trainset_norm.data.float())
    test_dataset_full = torch.utils.data.TensorDataset(testset_noise.float(), testset_norm.data.float())

    print(f"Trainset size: {len(train_dataset_full)}")
    print(f"Testset size: {len(test_dataset_full)}")

    train_loader = torch.utils.data.DataLoader(train_dataset_full, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset_full, batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader