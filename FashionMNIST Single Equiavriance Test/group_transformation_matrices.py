# Custom matrices
import torch
import numpy as np
#Augmentation functions

#Rotation matrix (used only for test cases). Use transforms.rotation for augmentation
def rotation_matrix(w):
  "Rotates W (square matrix) by 90"
  m,m = w.size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[m-1-j,i]

  return w_new

#Horizontal flip
def hflip_matrix(w):
  "Hflip (square matrix)"
  m,m = w.size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[i,m-1-j]

  return w_new

#Vertical flip
def vflip_matrix(w):
  "Vflip (square matrix)"
  m,m = w.size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[m-1-i,j]

  return w_new


#Synthetic 1
def get_next_indices_sc1(i,j,m,h):
    "works for even m,h for now."
    i_next, j_next = i,j

    if i<int(m/2) and j<int(h/2):
      i_next,j_next = i,j+int(h/2)
    elif i<int(m/2) and j>=int(h/2):
      i_next,j_next = i+int(m/2),j
    elif i>=int(m/2) and j>=int(h/2):
      i_next,j_next = i,j-int(h/2)
    else:
      i_next,j_next = i-int(m/2),j

    return (i_next,j_next)

def synthetic1_matrix(w):
  "Rotates W (square matrix) by one quadrant"
  m,m = w[0].size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[get_next_indices_sc1(i,j,m,m)]
  return w_new.view(1,m,m)

#Synthetic 2
def get_next_indices_sc2(i,j,m,h):
    "works for even m,h for now."
    i_next, j_next = i,(j+int(h/2))%h

    return (i_next,j_next)

def synthetic2_matrix(w):
  "Rotates W (square matrix) by one quadrant"
  m,m = w[0].size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[get_next_indices_sc2(i,j,m,m)]

  return w_new.view(1,m,m)

#Synthetic 3
def get_next_indices_sc3(i,j,m,h):
    "works for even m,h for now."
    i_next, j_next = (i+int(m/2))%m,j

    return (i_next,j_next)

def synthetic3_matrix(w):
  "Rotates W (square matrix) by one quadrant"
  m,m = w[0].size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[get_next_indices_sc3(i,j,m,m)]

  return w_new.view(1,m,m)

#Synthetic 4
def get_next_indices_sc4(i,j,m,h):
    "works for even m,h for now."
    if j<int(h/2):
      i_next, j_next = (i+int(m/2))%m,j
    else:
      i_next, j_next = i,j

    return (i_next,j_next)

def synthetic4_matrix(w):
  "Rotates W (square matrix) by one quadrant"
  m,m = w[0].size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[get_next_indices_sc4(i,j,m,m)]

  return w_new.view(1,m,m)

#Synthetic 5
def get_next_indices_sc5(i,j,m,h):
    "works for even m,h for now."
    if j>=int(h/2):
      i_next, j_next = (i+int(m/2))%m,j
    else:
      i_next, j_next = i,j

    return (i_next,j_next)

def synthetic5_matrix(w):
  "Rotates W (square matrix) by one quadrant"
  m,m = w[0].size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[get_next_indices_sc5(i,j,m,m)]

  return w_new.view(1,m,m)

#Synthetic 6
def get_next_indices_sc6(i,j,m,h):
    "works for even m,h for now."
    if i<int(m/2):
      i_next, j_next = i,(j+int(h/2))%h
    else:
      i_next, j_next = i,j

    return (i_next,j_next)

def synthetic6_matrix(w):
  "Rotates W (square matrix) by one quadrant"
  m,m = w[0].size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[get_next_indices_sc6(i,j,m,m)]

  return w_new.view(1,m,m)

#Synthetic 7
def get_next_indices_sc7(i,j,m,h):
    "works for even m,h for now."
    if i>=int(m/2):
      i_next, j_next = i,(j+int(h/2))%h
    else:
      i_next, j_next = i,j

    return (i_next,j_next)

def synthetic7_matrix(w):
  "Rotates W (square matrix) by one quadrant"
  m,m = w[0].size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[get_next_indices_sc7(i,j,m,m)]

  return w_new.view(1,m,m)

#Synthetic 8
def get_next_indices_sc8(i,j,m,h,d):
    "one step vertical translation"
    i_next, j_next = (i+d)%m,j

    return (i_next,j_next)

def vtrans_matrix(w,d=4):
  "translate vertical"
  m,m = w[0].size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[get_next_indices_sc8(i,j,m,m,d)]

  return w_new.view(1,m,m)

#Synthetic 9
def get_next_indices_sc9(i,j,m,h,d):
    "one step vertical translation"
    i_next, j_next = i,(j+d)%h

    return (i_next,j_next)

def htrans_matrix(w,d=4):
  "translate vertical"
  m,m = w[0].size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[get_next_indices_sc9(i,j,m,m,d)]

  return w_new.view(1,m,m)