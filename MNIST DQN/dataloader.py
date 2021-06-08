import torch
import torchvision
import torchvision.transforms as transforms
from augmentation_functions import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(0)
np.random.seed(0)

# Augmentation dictionary: connecting aug_array to used augmentations
aug_dict = {
            0: transforms.RandomRotation((-180,180)),
            1: transforms.RandomHorizontalFlip(p=0.5),
            2: transforms.RandomVerticalFlip(p=0.5),
            3: RandomHorizontal(),
            4: RandomVertical(),
            5: RandomScramble1(),
            6: RandomScramble2(),
            7: RandomScramble3(),
            8: RandomScramble4(),
            9: RandomScramble5(),
            10: RandomScramble6(),
            11: RandomScramble7()
            }

#######################################################################################################################
# MNIST
#######################################################################################################################

# Composing the augmentation array from aug_array using aug_dict
def mnist_transform_array(aug_dict, aug_array=torch.zeros(12)):
  # Outputs the array of transformations from aug_array
  transforms_array = []
  for i in range(len(aug_array)):
    if i == 3:
      transforms_array.append(transforms.ToTensor())
    if aug_array[i] > 0:
      transforms_array.append(aug_dict[i])
  transforms_array.append(transforms.Normalize((0.1307,), (0.3081,)))
  return transforms_array

# Get trainloader with augmentation of choice
def get_mnist_trainloader(aug_array, train_size=2000, batch_size=4, shuffle=True, num_workers=2,aug_dict=aug_dict):
  traintransform=transforms.Compose(mnist_transform_array(aug_dict, aug_array))
  trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=traintransform)
  trainset_mini, _= torch.utils.data.random_split(trainset, [train_size, len(trainset) - train_size])
  trainloader = torch.utils.data.DataLoader(trainset_mini, batch_size=batch_size, shuffle=True, num_workers=2)
  return trainloader

# Get trainset with augmentation of choice
def get_mnist_trainset(aug_array, train_size=2000,aug_dict=aug_dict):
  traintransform=transforms.Compose(mnist_transform_array(aug_dict, aug_array))
  trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=traintransform)
  trainset_mini, _= torch.utils.data.random_split(trainset, [train_size, len(trainset) - train_size])
  return trainset_mini

# Get testloader with augmentation of choice
def get_mnist_testloader(aug_array, test_size=1000, batch_size=4, shuffle=True, num_workers=2,aug_dict=aug_dict):
  testtransform=transforms.Compose(mnist_transform_array(aug_dict, aug_array))
  testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=testtransform)
  testset_mini, _= torch.utils.data.random_split(testset, [test_size, len(testset) - test_size])
  testloader = torch.utils.data.DataLoader(testset_mini, batch_size=batch_size, shuffle=False, num_workers=2)
  return testloader

# Get testset with augmentation of choice
def get_mnist_testset(aug_array, batch_size=4, shuffle=True, num_workers=2,aug_dict=aug_dict):
  testtransform=transforms.Compose(mnist_transform_array(aug_dict, aug_array))
  testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=testtransform)
  testset_mini, _= torch.utils.data.random_split(testset, [test_size, len(testset) - test_size])
  return testset_mini

classes_mnist = ('0','1','2','3','4','5','6','7','8','9')

#######################################################################################################################
# CIFAR-10
#######################################################################################################################

def cifar10_transform_array(aug_dict, aug_array=torch.zeros(12)):
  # Outputs the array of transformations from aug_array
  transforms_array = []
  for i in range(len(aug_array)):
    if i == 3:
      transforms_array.append(transforms.ToTensor())
      transforms_array.append(CIFAR_dimension_3_to_1())
    if aug_array[i] > 0:
      transforms_array.append(aug_dict[i])
  transforms_array.append(transforms.Normalize((0.7,), (0.5,)))
  return transforms_array

# Get trainloader with augmentation of choice
def get_cifar10_trainloader(aug_array, train_size=2000, batch_size=4, shuffle=True, num_workers=2, aug_dict=aug_dict):
  traintransform=transforms.Compose(cifar10_transform_array(aug_dict, aug_array))
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=traintransform)
  trainset_mini, _= torch.utils.data.random_split(trainset, [train_size, len(trainset) - train_size])
  trainloader = torch.utils.data.DataLoader(trainset_mini, batch_size=batch_size, shuffle=True, num_workers=2)
  return trainloader

# Get testloader with augmentation of choice
def get_cifar10_testloader(aug_array, test_size=1000, batch_size=4, shuffle=True, num_workers=2,aug_dict=aug_dict):
  testtransform=transforms.Compose(cifar10_transform_array(aug_dict, aug_array))
  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=testtransform)
  testset_mini, _= torch.utils.data.random_split(testset, [test_size, len(testset) - test_size])
  testloader = torch.utils.data.DataLoader(testset_mini, batch_size=batch_size, shuffle=False, num_workers=2)
  return testloader

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

