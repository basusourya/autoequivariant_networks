# Section Experiments -> Augmented MNIST -> Classification -> Table 1
import torch
import torchvision
import torchvision.transforms as transforms
from equivariance_functions import *
from augmentation_functions import *
from equivariance_search_utilities import *
from dataloader import *
from test_equivariance import *
import numpy as np
import argparse
import math
from equivariance_functions import *
from torch import nn

class EQNet(nn.Module):
    def __init__(self, network_dimensions, eq_indices):
      # hidden_sizes and eq_indices are both lists of size 2
        super(EQNet, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(network_dimensions[0], network_dimensions[1])  # 
        self.fc2 = nn.Linear(network_dimensions[1], network_dimensions[2]) # 
        self.fc3 = nn.Linear(network_dimensions[2], network_dimensions[3]) # 
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.eq_indices = eq_indices
        self.network_dimensions = network_dimensions
        
    def forward(self, x):
        out = self.relu(x.mm(self.fc1.weight.view(-1)[self.eq_indices[0]].view(self.network_dimensions[0], self.network_dimensions[1]))) + self.fc1.bias
        out = self.relu(out.mm(self.fc2.weight.view(-1)[self.eq_indices[1]].view(self.network_dimensions[1], self.network_dimensions[2]))) + self.fc2.bias
        out = out.mm(torch.transpose(self.fc3.weight, 0 ,1)) + self.fc3.bias
        return out


def train(net, device, train_loader, criterion, optimizer, epoch, num_epochs=20, use_cuda=True, train_size=60000, batch_size=64):
  for i, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
      images = Variable(images.view(-1, 28*28))         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
      labels = Variable(labels)
      
      if use_cuda and torch.cuda.is_available():
          images = images.cuda()
          labels = labels.cuda()
      
      optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
      outputs = net(images)                             # Forward pass: compute the output class given a image
      loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
      loss.backward()                                   # Backward pass: compute the weight
      optimizer.step()                                  # Optimizer: update the weights of hidden nodes
      
      if (i+1) % 40 == 0:                              # Logging
          print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
              %(epoch+1, num_epochs, i+1, (train_size/batch_size), loss.item()))
  return net

def test(net, device, test_loader, use_cuda=True):         
  correct = 0
  total = 0

  net.eval()
  for images, labels in test_loader:
      images = Variable(images.view(-1, 28*28))
      #if torch.randn(1)>0.0:
      #  images = torchvision.transforms.functional.hflip(images)
      if use_cuda and torch.cuda.is_available():
          images = images.cuda()
          labels = labels.cuda()
      
      
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
      total += labels.size(0)                    # Increment the total count
      correct += (predicted == labels).sum()     # Increment the correct count
      
  print('Accuracy of the network on the 10K test images:', (100.0 * correct / total))

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='Augmented MNIST hypothesis testing')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--train-size', type=int, default=60000, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-size', type=int, default=10000, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--epochs', type=int, default=20, metavar='N',
						help='number of epochs to train (default: 14)')
	parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
						help='learning rate (default: 1.0)')
	parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
						help='Learning rate step gamma (default: 0.7)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--dry-run', action='store_true', default=False,
						help='quickly check a single pass')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=False,
						help='For Saving the current Model')
	parser.add_argument('--aug-array-id', type=int, default=0,
						help='augmentation index to be used from aug_array_list')
	args = parser.parse_args()

	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu")

	train_kwargs = {'batch_size': args.batch_size}
	test_kwargs = {'batch_size': args.test_batch_size}

	if use_cuda:
		cuda_kwargs = {'num_workers': 1,
					   'pin_memory': True,
					   'shuffle': True}
		train_kwargs.update(cuda_kwargs)
		test_kwargs.update(cuda_kwargs)

	aug_array_list = [
				  [0,0,0,0,0,0,0,0,0,0,0,0],
				  [0,1,1,0,0,0,1,0,0,0,0,1],
				  [1,1,0,0,0,0,0,0,1,1,0,0],
				  [1,0,0,1,0,0,0,1,0,0,1,0],
				  [1,1,1,1,1,1,0,0,0,0,0,0],
				  [1,1,1,1,1,1,1,1,1,1,1,1]
				  ]

	eq_array_list = [
				 [0,0,0,0,0,0,0,0,0,0,0,0],
				 [1,0,0,0,0,0,0,0,0,0,0,0],
				 [0,1,0,0,0,0,0,0,0,0,0,0],
				 [0,0,1,0,0,0,0,0,0,0,0,0],
				 [0,0,0,1,0,0,0,0,0,0,0,0],
				 [0,0,0,0,1,0,0,0,0,0,0,0],
				 [0,0,0,0,0,1,0,0,0,0,0,0],
				 [0,0,0,0,0,0,1,0,0,0,0,0],
				 [0,0,0,0,0,0,0,1,0,0,0,0],
				 [0,0,0,0,0,0,0,0,1,0,0,0],
				 [0,0,0,0,0,0,0,0,0,1,0,0],
				 [0,0,0,0,0,0,0,0,0,0,1,0],
				 [0,0,0,0,0,0,0,0,0,0,0,1]   
				 ]

	aug_array = aug_array_list[args.aug_array_id]
	print("Augmentation array",aug_array)
	# load data with appropriate equivariances
	trainloader = get_mnist_trainloader(aug_dict=aug_dict,aug_array=aug_array, train_size=args.train_size, batch_size=args.batch_size, shuffle=True, num_workers=2)
	testloader = get_mnist_testloader(aug_dict=aug_dict,aug_array=aug_array, test_size=args.test_size, batch_size=args.test_batch_size, shuffle=True, num_workers=2)

	network_dimensions=[28*28, 20*20, 20*20, 10]

	for i in range(len(eq_array_list)):
		eq_array = eq_array_list[i]
		print("Equivariance array",eq_array)
		model = get_equivariant_network(network_dimensions, eq_array).to(device)
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		for epoch in range(args.epochs):
		  import time
		  start_time = time.time()
		  model = train(model, device, trainloader, criterion, optimizer, epoch, num_epochs=args.epochs, train_size=args.train_size)
		  time_elapsed = time.time() - start_time
		  print("Time elapsed",time_elapsed,"secs")
		# test
		test(model, device, testloader)


if __name__ == '__main__':
	main()






