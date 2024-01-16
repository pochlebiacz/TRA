import torch
import torchvision
from model import *

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

def data_prep_3_loaders(noise_lambda=64, batch_size_train=512, batch_size_test=512):
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
    testset_noise_32 = testset_norm + torch.randn(testset_norm.size()) * 32
    testset_noise_64 = testset_norm + torch.randn(testset_norm.size()) * 64
    testset_noise_128 = testset_norm + torch.randn(testset_norm.size()) * 128

    # -------------- Squeezing augmented data to [0, 255] ---------------
    testset_noise_32 = torch.clamp(testset_noise_32, 0, 255)
    testset_noise_64 = torch.clamp(testset_noise_64, 0, 255)
    testset_noise_128 = torch.clamp(testset_noise_128, 0, 255)

    train_dataset_full = torch.utils.data.TensorDataset(trainset_noise.float(), trainset_norm.data.float())
    test_dataset_full_32 = torch.utils.data.TensorDataset(testset_noise_32.float(), testset_norm.data.float())
    test_dataset_full_64 = torch.utils.data.TensorDataset(testset_noise_64.float(), testset_norm.data.float())
    test_dataset_full_128 = torch.utils.data.TensorDataset(testset_noise_128.float(), testset_norm.data.float())

    print(f"Trainset size: {len(train_dataset_full)}")
    print(f"Testset size: {len(test_dataset_full_32)}")
    print(f"Testset size: {len(test_dataset_full_64)}")
    print(f"Testset size: {len(test_dataset_full_128)}")

    train_loader = torch.utils.data.DataLoader(train_dataset_full, batch_size=batch_size_train, shuffle=True)
    test_loader_32 = torch.utils.data.DataLoader(test_dataset_full_32, batch_size=batch_size_test, shuffle=False)
    test_loader_64 = torch.utils.data.DataLoader(test_dataset_full_64, batch_size=batch_size_test, shuffle=False)
    test_loader_128 = torch.utils.data.DataLoader(test_dataset_full_128, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader_32, test_loader_64, test_loader_128