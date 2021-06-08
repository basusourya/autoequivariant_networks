import math
from equivariance_functions import *
import random
import numpy as np

from torch import nn
'''
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
'''
class EQNet(nn.Module):
    def __init__(self, network_dimensions, eq_indices, n_orbits):
      # hidden_sizes and eq_indices are both lists of size 2
        super(EQNet, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(n_orbits[0], 1)  # 
        self.fc2 = nn.Linear(n_orbits[1], 1) # 
        self.fc3 = nn.Linear(network_dimensions[2], network_dimensions[3]) # 
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.eq_indices = eq_indices
        self.network_dimensions = network_dimensions

        
    def forward(self, x):
        self.w1 = self.fc1.weight.view(-1)[self.eq_indices[0]].view(self.network_dimensions[0], self.network_dimensions[1])
        self.w2 = self.fc2.weight.view(-1)[self.eq_indices[1]].view(self.network_dimensions[1], self.network_dimensions[2])
        out = self.relu(x.mm(self.w1)) + self.fc1.bias
        out = self.relu(out.mm(self.w2)) + self.fc2.bias
        out = out.mm(torch.transpose(self.fc3.weight, 0 ,1)) + self.fc3.bias
        return out



def get_total_parameters(network_dimensions,n_orbits):
  total = n_orbits[0] + network_dimensions[1] + n_orbits[1] + network_dimensions[2] + network_dimensions[2]*network_dimensions[3] + network_dimensions[3]
  return total

def get_equivariance_indices(eq_array,I_prev_orbits,I_prev_index_to_orbit,input_size,hidden_size):
  I_orbits, I_index_to_orbit = I_prev_orbits, I_prev_index_to_orbit
  G_list = ["rotation","hflip","vflip","htrans","vtrans","synthetic1","synthetic2","synthetic3",
            "synthetic4","synthetic5","synthetic6","synthetic7"] 
  G = []

  for i in range(len(eq_array)):
    if eq_array[i] == 1:
      G.append(G_list[i])

  for g in G:
    I_orbits, I_index_to_orbit = get_g_eq_index(g,int(math.sqrt(input_size)),int(math.sqrt(hidden_size)),I_prev_orbits,I_prev_index_to_orbit)
    I_prev_orbits,I_prev_index_to_orbit = I_orbits, I_index_to_orbit

  return I_orbits, I_index_to_orbit

def get_equivariant_network(network_dimensions=[28*28, 20*20, 20*20, 10], eq_array=[0,0,0,0,0,0,0,0,0,0,0,0]):
  # get the equivariance indices
  I_prev_orbits = []
  I_prev_index_to_orbit = []
  I_orbits = []
  I_index_to_orbit = []
  n_orbits = []

  for it in range(len(network_dimensions)-2):
    #I_prev_orbits.append([{i} for i in range(network_dimensions[it]*network_dimensions[it+1])])

    m,h = int(math.sqrt(network_dimensions[it])),int(math.sqrt(network_dimensions[it+1]))
    I_prev_index_to_orbit.append([i for i in range(network_dimensions[it]*network_dimensions[it+1])])
    I_index_to_orbit.append(squeeze_orbits(get_G_list_equivariance_indices_graph(m,h,eq_array,I_prev_index_to_orbit[it],d=4)))
    #I_orbits.append(temp_1)
    #I_index_to_orbit.append(temp_2)
    n_orbits.append(len(set(I_index_to_orbit[it])))

  eq_indices = I_index_to_orbit

  # define the network model
  eqnet = EQNet(network_dimensions, eq_indices,n_orbits)

  # count the number of parameters
  pytorch_total_params_orbits = get_total_parameters(network_dimensions,n_orbits)
  print("net parameters orbits:",pytorch_total_params_orbits)

  #pytorch_total_params = sum(p.numel() for p in eqnet.parameters() if p.requires_grad)
  #print("net parameters:",pytorch_total_params)
  return eqnet

def squeeze_orbits(orbits):

  sorted_orbits, arg_sorted_orbits = np.sort(orbits), np.argsort(orbits)
  rank_orbits = np.argsort(arg_sorted_orbits)
  sorted_ranked_orbits = []

  sorted_orbits = sorted_orbits - min(sorted_orbits)
  sorted_orbits = sorted_orbits.tolist()

  current_rank = 0
  current_orbit = 0
  for i in range(len(sorted_orbits)):
    if sorted_orbits[i] > current_orbit:
      current_orbit = sorted_orbits[i]
      sorted_orbits[i] = current_rank+1
      current_rank += 1
    else:
      sorted_orbits[i] = current_rank
      
  orbits_new = [sorted_orbits[rank_orbits[i]] for i in range(len(sorted_orbits))]
  return orbits_new