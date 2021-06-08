# Group Neural Architecture Search
# Example use: 

#=======================================================Imports===========================================================================================================
import time
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch.optim as optim

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms


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
#=====================================================================Quick fix===================================================================================================
# Quick fix for an issue with downloading FashionMNIST
from six.moves import urllib    
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

input_size = 12 #12 by default
hidden_sizes = [400, 400, 400]
output_size = input_size
n_actions = output_size

#=====================================================================Q-Network=========================================================================================================

class QNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(QNet, self).__init__()                   # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])  # 1st Full-Connected Layer: k (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2]) # 2nd Full-Connected Layer: 500 (hidden node) -> k (output class)
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        x = x/10
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out
        
#define the policy network and target network
device = 'cuda'
policy_net = QNet(input_size, hidden_sizes, output_size).to(device)
target_net = QNet(input_size, hidden_sizes, output_size).to(device)
#=====================================================================Train and test for child network============================================================================
def train(net, device, trainimageset, trainlabelset, criterion, optimizer, epoch, num_epochs=20, use_cuda=True, train_size=60000, batch_size=64):
  
  for i in range(len(trainimageset)):   # Load a batch of images with its (index, data, class)
      images = trainimageset[i]         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
      labels = trainlabelset[i]
      
      optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
      outputs = net(images)                             # Forward pass: compute the output class given a image
      loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
      loss.backward()                                   # Backward pass: compute the weight
      optimizer.step()                                  # Optimizer: update the weights of hidden nodes
      
      #if (i+1) % 40 == 0:                              # Logging
          #print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
              #%(epoch+1, num_epochs, i+1, (train_size/batch_size), loss.item()))
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
      
  #print('Accuracy of the network on the 1K test images:', (100.0 * correct / total))
  return (100.0 * correct / total)
#=====================================================================Environment (keep it clean :p)============================================================================
class Environment(object):
  #describes the environment for our Q-learning problem
  def __init__(self, device, trainimageset, trainlabelset, testimageset, testlabelset, child_network_dimensions, child_train_size, child_test_size, child_batch_size, child_test_batch_size, child_lr=1e-3, child_epochs=4, g_size=12):
    # Start with all zero vector
    # Initialize only once, reset as many times as needed, to avoid recomputation of base accuracies.
    print("Group Neural Architectural Search using Group Decomposition and Reinforcement Learning!")
    self.k = g_size # k = g_size for fully connected and k = g_size + 4*h_size for convolutional neural networks
    self.current_state = torch.zeros(g_size) # g_size represents the size of the array of groups for equivariance
    self.next_state = torch.zeros(g_size)
    self.models_trained = 1 # 1 corresponds to the base case
    self.time = 0                                                    # Further, base_reward should be equal to zero.
    self.next_state_string = ''.join(str(e) for e in self.next_state)
    self.device = device
    self.trainimageset = trainimageset
    self.trainlabelset = trainlabelset
    self.testimageset = testimageset
    self.testlabelset = testlabelset
    self.child_network_dimensions = child_network_dimensions
    self.child_epochs = child_epochs
    self.child_lr = child_lr
    self.child_train_size = child_train_size
    self.child_test_size = child_test_size
    self.child_batch_size = child_batch_size
    self.child_test_batch_size = child_test_batch_size
    self.base_accuracy = self.get_state_accuracy(self.current_state)
    self.base_reward = self.get_state_reward(self.base_accuracy, self.base_accuracy)
    current_state_string = ''.join(str(e) for e in self.current_state)
    self.visited_model_rewards = {current_state_string: self.base_reward} # Dictionary of models visited and their rewards. Models are saved in the form of a binary string of length g_size
    self.visited_model_accuracies = {current_state_string: self.base_accuracy} # Dictionary of models visited and their accuracies. Models are saved in the form of a binary string of length g_size

  def step(self, action):
    #returns from QNN for the data augmentation problem; for this toy example it is going to return +100 for state = all 1s, and -1 for anything else
    reward = 0
    accuracy = 0
    new_models_trained = False
    done = False
    self.time += 1

    # Make action
    action = action.item()

    # Update states
    for i in range(self.k):
      self.next_state[i] = self.current_state[i]
    self.next_state[action] = (self.current_state[action] + 1)%2 #remove for multiple adversaries
    self.update_current_state()
    self.next_state_string = ''.join(str(e) for e in self.next_state)

    # Update reward and accuracy
    if self.next_state_string not in self.visited_model_rewards:
      new_models_trained = True
      self.models_trained += 1
      accuracy = self.get_state_accuracy(self.current_state)
      reward = self.get_state_reward(accuracy, self.base_accuracy)
      self.visited_model_accuracies[self.next_state_string] = accuracy
      self.visited_model_rewards[self.next_state_string] = reward
    else:
      accuracy = self.visited_model_accuracies[self.next_state_string]
      reward = self.visited_model_rewards[self.next_state_string]

    if self.time > 100:
      done = True
    return accuracy, reward, new_models_trained, done

  def get_state_reward(self, state_accuracy, base_accuracy):
    return (state_accuracy - base_accuracy)*math.exp(abs(state_accuracy - base_accuracy))

  def get_state_accuracy(self, state):
    # Basic Hamming distance
    # goal_state = [0,1]*6
    # accuracy = sum([abs(x-y) for (x,y) in zip(goal_state,state)])
    torch.manual_seed(1)
    eq_array = state
    #print("Equivariance array",eq_array)
    child_model = get_equivariant_network(self.child_network_dimensions, eq_array.tolist()).to(device)
    #print("No. of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    child_criterion = nn.CrossEntropyLoss()
    child_optimizer = torch.optim.Adam(child_model.parameters(), lr=self.child_lr)
    test_accuracy = 0
    for epoch in range(self.child_epochs):
      #start_time = time.time()
      child_model = train(child_model, self.device, self.trainimageset, self.trainlabelset, child_criterion, child_optimizer, epoch, num_epochs=self.child_epochs, train_size=self.child_train_size)
      #time_elapsed = time.time() - start_time
      #print("Time elapsed",time_elapsed,"secs")
      # test
      test_accuracy = max(test_accuracy, test(child_model, self.device, self.testimageset, self.testlabelset))
    return test_accuracy

  def update_avg_reward(self,reward,time):
    self.avg_reward = (self.avg_reward*(time - 1) + reward)/self.time

  def update_avg_test_acc(self,test_acc,time):
    self.avg_test_acc = (self.avg_test_acc*(time - 1) + test_acc)/self.time

  def update_current_state(self):
    for i in range(self.k):
      self.current_state[i] = self.next_state[i]

  def reset(self):
    #reset to state 0 w.p. 0.5, rest of the time set to an uniformly random vector of length k
    if random.random() > 0.0:
      self.current_state = torch.zeros(self.k)
    else:
      a = [0]*self.k + [1]*self.k
      self.current_state = torch.tensor(random.sample(a, self.k))
    self.next_state = torch.zeros(self.k)
    print("Starting state:",self.current_state)
    self.time = 0
    print("reset!")
#=====================================================================Replay Memory===================================================================================================

# Replay memory
Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def push(self,*args):
    "Saves a transition"
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = Transition(*args)
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)


#=====================================================================Select action=========================================================================================================

steps_done = 0
def select_action(state, EPS, device, n_actions, policy_net):
  global steps_done
  sample = random.random()
  steps_done += 1
  if sample > EPS:
    with torch.no_grad():
      return policy_net(state).max(0)[1].view(1,1).to(device) #returns the index instead of the value
  else:
    return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

#=====================================================================Plots===============================================================================================================

def plot_update(x_models_trained, y_accuracy):
  fig = plt.figure()
  plt.title('Deep Q-learning FashionMNIST')
  plt.xlabel('Models trained')
  plt.ylabel('Accuracy')
  plt.plot(x_models_trained, y_accuracy, label="Accuray")
  plt.pause(0.001)  #pause a bit so that plots are updated
  fig.savefig('GNAS_FCNN_FashionMNIST.eps', format='eps', dpi=1000)

def plot_overall(x_models_trained, y_accuracy, aug_array_id=0):
  # average windowed plot
  # averaged epsilon plot
  window_size = 60 # window_size < 200
  EPS_MODELS_TRAINED_LIST = [0,200,300,400,500,600,700,800,900,950,1000,1200]
  EPS_LIST = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01]

  y_accuracy_window = [sum(y_accuracy[i:i+window_size])/window_size for i in range(len(y_accuracy)-window_size)]
  y_epsilon_accuracy = []
  v_lines = []
  x_models_trained_window = x_models_trained[window_size:]

  for i in range(len(EPS_MODELS_TRAINED_LIST)-1):
    array = y_accuracy[EPS_MODELS_TRAINED_LIST[i]:EPS_MODELS_TRAINED_LIST[i+1]]
    y_epsilon_accuracy += [sum(array)/(len(array))]*(len(array))
    v_lines.append(sum(array)/(len(array)))

  y_epsilon_accuracy = y_epsilon_accuracy[window_size:len(x_models_trained)]
  fig = plt.figure()
  plt.title('Deep Q-learning FashionMNIST'+ ' Aug' + str(aug_array_id))
  plt.xlabel('Models trained')
  plt.ylabel('Accuracy')
  plt.plot(x_models_trained_window, y_accuracy_window, label="Rolling mean Accuray")
  plt.fill_between(x_models_trained_window, y_epsilon_accuracy, label="Average Accuracy Per Epsilon",alpha=0.5)
  plt.legend()
  x1,x2,y1,y2 = plt.axis()
  plt.axis((x1,x2,5,y2))

  for i in range(len(v_lines)):
    plt.vlines(EPS_MODELS_TRAINED_LIST[i+1],5,v_lines[i], alpha=0.3)

  fig.text(0.18,0.14,'$\epsilon=1$')
  fig.text(0.275,0.14,'$.9$')
  fig.text(0.35,0.14,'$.8$')
  fig.text(0.428,0.14,'$.7$')
  fig.text(0.490,0.14,'$.6$')
  fig.text(0.568,0.14,'$.5$')
  fig.text(0.637,0.14,'$.4$')
  fig.text(0.705,0.14,'$.3$')
  fig.text(0.762,0.14,'$.2$')
  fig.text(0.795,0.14,'$.1$')
  fig.text(0.83,0.14,'$.05$')
  plt.pause(0.001)  #pause a bit so that plots are updated
  fig.savefig('GNAS_FCNN_MNIST_Aug'+str(aug_array_id)+'.eps', format='eps', dpi=1000)

#plot_overall(x_models_trained, y_accuracy)

#=====================================================================Optimize model===============================================================================================================

def optimize_model(k, device, memory, q_optimizer, Q_BATCH_SIZE, Q_GAMMA):
  if len(memory.memory) < Q_BATCH_SIZE:
    return
  transitions = memory.sample(Q_BATCH_SIZE)
  # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
  # detailed explanation). This converts batch-array of Transitions
  # to Transition of batch-arrays.
  batch = Transition(*zip(*transitions))

  # Compute a mask of non-final states and concatenate the batch elements
  # (a final state would've been the one after which simulation ended)

  state_batch = torch.cat(batch.state).view(Q_BATCH_SIZE,k).to(device)
  action_batch = torch.cat(batch.action).to(device)
  reward_batch = torch.cat(batch.reward).to(device)
  next_state_batch = torch.cat(batch.next_state).view(Q_BATCH_SIZE,k).to(device)


  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  #print("State batch:",state_batch)
  state_action_values = policy_net(state_batch).gather(1, action_batch).to(device)
  #print("state_action_values:",state_action_values)

  # Compute V(s_{t+1}) for all next states.
  # Expected values of actions for non_final_next_states are computed based
  # on the "older" target_net; selecting their best reward with max(1)[0].
  # This is merged based on the mask, such that we'll have either the expected
  # state value or 0 in case the state was final.

  next_state_values = target_net(next_state_batch).max(1)[0].detach().to(device)
  #print("next_state_values:",next_state_values)
  # Compute the expected Q values
  expected_state_action_values = (next_state_values * Q_GAMMA) + reward_batch
  #print("expected_state_action_values:",expected_state_action_values)
  # Compute Huber loss
  loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

  # Optimize the model
  q_optimizer.zero_grad()
  loss.backward()
  #for param in policy_net.parameters():
    #param.grad.data.clamp_(-1, 1)
  q_optimizer.step()

def main():
  # Training settings
  # For multiple augmentations set the flag --multiple-augmentations to true
  parser = argparse.ArgumentParser(description='Deep Q-learning FashionMNIST')
  parser.add_argument('--child-batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
  parser.add_argument('--child-test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
  parser.add_argument('--child-train-size', type=int, default=60000, metavar='N',
            help='input batch size for training (default: 64)')
  parser.add_argument('--child-test-size', type=int, default=10000, metavar='N',
            help='input batch size for training (default: 64)')
  parser.add_argument('--child-epochs', type=int, default=20, metavar='N',
            help='number of epochs to train (default: 14)')
  parser.add_argument('--child-lr', type=float, default=1e-3, metavar='LR',
            help='learning rate (default: 1.0)')
  parser.add_argument('--child-gamma', type=float, default=0.7, metavar='M',
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
  parser.add_argument('--g-size', type=int, default=12,
            help='Size of the group array')
  parser.add_argument('--max_models', type=int, default=1000,
            help='Maximum number of models to be trained')
  parser.add_argument('--max_episodes', type=int, default=60,
            help='Maximum number of episodes')
  args = parser.parse_args()
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  torch.manual_seed(args.seed)
  device = torch.device("cuda" if use_cuda else "cpu")

  train_kwargs = {'batch_size': args.child_batch_size}
  test_kwargs = {'batch_size': args.child_test_batch_size}
  if use_cuda:
    cuda_kwargs = {'num_workers': 1,
             'pin_memory': True,
             'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

  multiple_aug_array_list = [
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,1,1,0,0,0,1,0,0,0,0,1],
          [1,1,0,0,0,0,0,0,1,1,0,0],
          [1,0,0,1,0,0,0,1,0,0,1,0],
          [1,1,1,1,1,1,0,0,0,0,0,0],
          [1,1,1,1,1,1,1,1,1,1,1,1]
          ]

  k = args.g_size
  aug_array = multiple_aug_array_list[args.aug_array_id]
  print("Augmentation array",aug_array)
  # Setup the dataset for child network
  trainloader = get_fashionmnist_trainloader(aug_dict=aug_dict,aug_array=aug_array, train_size=args.child_train_size, batch_size=args.child_batch_size, shuffle=True, num_workers=2)
  testloader = get_fashionmnist_testloader(aug_dict=aug_dict,aug_array=aug_array, test_size=args.child_test_size, batch_size=args.child_test_batch_size, shuffle=True, num_workers=2)

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

  child_network_dimensions = [28*28, 20*20, 20*20, 10]

  # Setup the Q-network and its hyperparameters
  input_size = args.g_size #12 by default
  hidden_sizes = [400, 400, 400]
  output_size = input_size
  n_actions = output_size
  memory = ReplayMemory(10000)
  #define the policy network and target network

  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()

  q_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
  memory = ReplayMemory(10000)

  #=======================================================================Setup for Q-learning=============================================================================
  Q_BATCH_SIZE = 128
  Q_GAMMA = 0.5 # Dependency on the future
  Q_EPS_LIST = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01]
  Q_EPS_INDEX = 0
  Q_EPS = Q_EPS_LIST[Q_EPS_INDEX]
  Q_MODELS_TRAINED_LIST = [200,300,400,500,600,700,800,900,950,1000,1200]
  Q_TARGET_UPDATE = 2
  env = Environment(device=device, trainimageset=trainimageset, trainlabelset=trainlabelset, testimageset=testimageset, testlabelset=testlabelset, child_network_dimensions=child_network_dimensions, child_train_size=args.child_train_size, child_test_size=args.child_test_size, child_batch_size=args.child_batch_size, child_test_batch_size=args.child_test_batch_size, child_lr=args.child_lr, child_epochs=args.child_epochs, g_size=args.g_size) # (self, device, trainimageset, trainlabelset, testimageset, testlabelset, child_network_dimensions, child_train_size, child_test_size, child_batch_size, child_test_batch_size, child_lr=1e-3, child_epochs=4, g_size=12)
  y_accuracy = [env.base_accuracy] # Compute the rolling mean accuracy and average per epsilon accuracy from here
  x_models_trained = [env.models_trained] # Should be an enumeration from 1,...,total models to be trained.
  steps_per_model_list = []
  average_model_accuracy = env.base_accuracy
  steps_per_model = 1 # Considering the base accuracies and steps
  num_episodes = 0

  #==========================================================================Iteration loop===============================================================================
  while env.models_trained < args.max_models and num_episodes < args.max_episodes:
    # Select and perform an action
    state = torch.tensor([env.current_state[i] for i in range(env.k)]).to(device)
    action = select_action(state, Q_EPS, device, n_actions, policy_net)
    accuracy, reward, new_models_trained, done = env.step(action) # done = True when 1 episode completes, new_models_trained = True only when a new model has been trained in the step
    reward = torch.tensor([reward], device=device)
    next_state = torch.tensor([env.next_state[i] for i in range(env.k)])
    memory.push(state, action, next_state, reward)
    
    steps_per_model += 1
    average_model_accuracy = (average_model_accuracy*(steps_per_model-1) + accuracy)/steps_per_model

    #================================================Perform one step of the optimization (on the target network)===========================================================
    optimize_model(env.k, device, memory, q_optimizer, Q_BATCH_SIZE, Q_GAMMA)
    #=======================Check if new models are trained===========================================
    if new_models_trained:
      print("Number of models trained:", env.models_trained)
      y_accuracy.append(average_model_accuracy)
      x_models_trained.append(env.models_trained)
      steps_per_model_list.append(steps_per_model)
      average_model_accuracy = 0
      steps_per_model = 0
      if env.models_trained in Q_MODELS_TRAINED_LIST:
        Q_EPS_INDEX += 1
        Q_EPS = Q_EPS_LIST[Q_EPS_INDEX]

    if done:
      num_episodes += 1
      print("Number of episodes:", num_episodes)
      print("Current state:", env.current_state)
      print("Current epsilon value:", Q_EPS)
      plot_update(x_models_trained, y_accuracy)
      env.reset()

    # Update the target network, copying all weights and biases in DQN
    if env.models_trained % Q_TARGET_UPDATE == 0:
      target_net.load_state_dict(policy_net.state_dict())
      torch.save(target_net.state_dict(), "Target_Net_FashionMNIST_DQN_AUG_"+str(args.aug_array_id))

  #plot_overall(x_models_trained, y_accuracy, args.aug_array_id)

  # Save network and plot data
  np.save("y_accuracy"+str(args.aug_array_id),y_accuracy)
  np.save("x_models_trained"+str(args.aug_array_id),x_models_trained)
  np.save("steps_per_model_list"+str(args.aug_array_id),steps_per_model_list)
  np.save("model_accuracies"+str(args.aug_array_id),env.visited_model_accuracies)
  np.save("model_rewards"+str(args.aug_array_id),env.visited_model_rewards)
  torch.save(policy_net.state_dict(), "FashionMNIST_DQN_AUG_"+str(args.aug_array_id))
  print('Complete')


if __name__ == '__main__':
  main()