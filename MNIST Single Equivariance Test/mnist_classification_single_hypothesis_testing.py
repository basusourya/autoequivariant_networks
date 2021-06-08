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


def train(net, device, trainimageset, trainlabelset, criterion, optimizer, epoch, num_epochs=20, use_cuda=True, train_size=60000, batch_size=64):
  
  for i in range(len(trainimageset)):   # Load a batch of images with its (index, data, class)
      images = trainimageset[i]         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
      labels = trainlabelset[i]
      
      optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
      outputs = net(images)                             # Forward pass: compute the output class given a image
      loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
      loss.backward()                                   # Backward pass: compute the weight
      optimizer.step()                                  # Optimizer: update the weights of hidden nodes
      
      if (i+1) % 40 == 0:                              # Logging
          print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
              %(epoch+1, num_epochs, i+1, (train_size/batch_size), loss.item()))
  return net

def test(net, device, testimageset, testlabelset, use_cuda=True):         
  correct = 0
  total = 0

  net.eval()
  for i in range(len(testimageset)):
      images = testimageset[i]         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
      labels = testlabelset[i]

      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
      total += labels.size(0)                    # Increment the total count
      correct += (predicted == labels).sum()     # Increment the correct count
      
  print('Accuracy of the network on the 1K test images:', (100.0 * correct / total))
  return (100.0 * correct / total)

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

	trainimageset = []
	trainlabelset = []
	for i, (images, labels) in enumerate(trainloader):   # Load a batch of images with its (index, data, class)
		trainimageset.append(Variable(images.view(-1, 28*28)).to(device))         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
		trainlabelset.append(Variable(labels).to(device))

	testimageset = []
	testlabelset = []
	for i, (images, labels) in enumerate(testloader):   # Load a batch of images with its (index, data, class)
		testimageset.append(Variable(images.view(-1, 28*28)).to(device))         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
		testlabelset.append(Variable(labels).to(device))

	network_dimensions=[28*28, 20*20, 20*20, 10]
	test_accuracy_list = [] # Stores the maximum test accuracy for each of the equivariant networks

	for i in range(len(eq_array_list)):
		eq_array = eq_array_list[i]
		print("Equivariance array",eq_array)
		model = get_equivariant_network(network_dimensions, eq_array).to(device)
		print("No. of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		test_accuracy_temp = 0
		for epoch in range(args.epochs):
		  import time
		  start_time = time.time()
		  model = train(model, device, trainimageset, trainlabelset, criterion, optimizer, epoch, num_epochs=args.epochs, train_size=args.train_size)
		  time_elapsed = time.time() - start_time
		  print("Time elapsed",time_elapsed,"secs")
		# test
		  test_accuracy_temp = max(test_accuracy_temp, test(model, device, testimageset, testlabelset))
		test_accuracy_list.append(test_accuracy_temp)
		print("Max accuracy:", test_accuracy_temp)
	print("Test accuracies",test_accuracy_list)


if __name__ == '__main__':
	main()






