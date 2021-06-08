# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].
# Contains classes to add various forms of augmentation to input
# torchvision.transforms.RandomVerticalFlip(p=0.5),transforms.RandomHorizontalFlip()
import torch 

class RandomScramble1(object):
    def __init__(self, q=0.5):
        self.p = torch.tensor([1-q,q])
        self.q = q
        
    def __call__(self, tensor):
        if torch.multinomial(self.p,1)>0:
          tensor = self.synthetic1_matrix(tensor)
        return tensor

    def synthetic1_matrix(self,w):
      "Rotates W (square matrix) by one quadrant"
      m,m = w[0].size()
      w_new = torch.rand((m,m))
      for i in range(m):
        for j in range(m):
          w_new[i,j] = w[self.get_next_indices_sc1(i,j,m,m)]
      return w_new.view(1,m,m)

    def get_next_indices_sc1(self,i,j,m,h):
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

        return (0,i_next,j_next)

class RandomScramble2(object):
    def __init__(self, q=0.5):
        self.p = torch.tensor([1-q,q])
        self.q = q
        
    def __call__(self, tensor):
        if torch.multinomial(self.p,1)>0:
          tensor = self.synthetic2_matrix(tensor)
        return tensor

    def synthetic2_matrix(self,w):
      "Rotates W (square matrix) by one quadrant"
      m,m = w[0].size()
      w_new = torch.rand((m,m))
      for i in range(m):
        for j in range(m):
          w_new[i,j] = w[self.get_next_indices_sc2(i,j,m,m)]
      return w_new.view(1,m,m)

    def get_next_indices_sc2(self,i,j,m,h):
        "works for even m,h for now."
        i_next, j_next = i,(j+int(h/2))%h
        return (0,i_next,j_next)

class RandomScramble3(object):
    def __init__(self, q=0.5):
        self.p = torch.tensor([1-q,q])
        self.q = q
        
    def __call__(self, tensor):
        if torch.multinomial(self.p,1)>0:
          tensor = self.synthetic3_matrix(tensor)
        return tensor

    def synthetic3_matrix(self,w):
      "Rotates W (square matrix) by one quadrant"
      m,m = w[0].size()
      w_new = torch.rand((m,m))

      for i in range(m):
        for j in range(m):
          w_new[i,j] = w[self.get_next_indices_sc3(i,j,m,m)]

      return w_new.view(1,m,m)

    def get_next_indices_sc3(self,i,j,m,h):
        "works for even m,h for now."
        i_next, j_next = (i+int(m/2))%m,j
        return (0,i_next,j_next)

class RandomScramble4(object):
    def __init__(self, q=0.5):
        self.p = torch.tensor([1-q,q])
        self.q = q
        
    def __call__(self, tensor):
        if torch.multinomial(self.p,1)>0:
          tensor = self.synthetic4_matrix(tensor)
        return tensor

    def synthetic4_matrix(self,w):
      "Rotates W (square matrix) by one quadrant"
      m,m = w[0].size()
      w_new = torch.rand((m,m))

      for i in range(m):
        for j in range(m):
          w_new[i,j] = w[self.get_next_indices_sc4(i,j,m,m)]

      return w_new.view(1,m,m)

    def get_next_indices_sc4(self,i,j,m,h):
        "works for even m,h for now."
        if j<int(h/2):
          i_next, j_next = (i+int(m/2))%m,j
        else:
          i_next, j_next = i,j

        return (0,i_next,j_next)

class RandomScramble5(object):
    def __init__(self, q=0.5):
        self.p = torch.tensor([1-q,q])
        self.q = q
        
    def __call__(self, tensor):
        if torch.multinomial(self.p,1)>0:
          tensor = self.synthetic5_matrix(tensor)
        return tensor

    def synthetic5_matrix(self,w):
      "Rotates W (square matrix) by one quadrant"
      m,m = w[0].size()
      w_new = torch.rand((m,m))

      for i in range(m):
        for j in range(m):
          w_new[i,j] = w[self.get_next_indices_sc5(i,j,m,m)]

      return w_new.view(1,m,m)

    def get_next_indices_sc5(self,i,j,m,h):
        "works for even m,h for now."
        if j>=int(h/2):
          i_next, j_next = (i+int(m/2))%m,j
        else:
          i_next, j_next = i,j

        return (0,i_next,j_next)

class RandomScramble6(object):
    def __init__(self, q=0.5):
        self.p = torch.tensor([1-q,q])
        self.q = q
        
    def __call__(self, tensor):
        if torch.multinomial(self.p,1)>0:
          tensor = self.synthetic6_matrix(tensor)
        return tensor

    def synthetic6_matrix(self,w):
      "Rotates W (square matrix) by one quadrant"
      m,m = w[0].size()
      w_new = torch.rand((m,m))

      for i in range(m):
        for j in range(m):
          w_new[i,j] = w[self.get_next_indices_sc6(i,j,m,m)]

      return w_new.view(1,m,m)

    def get_next_indices_sc6(self,i,j,m,h):
        "works for even m,h for now."
        if i<int(m/2):
          i_next, j_next = i,(j+int(h/2))%h
        else:
          i_next, j_next = i,j

        return (0,i_next,j_next)


class RandomScramble7(object):
    def __init__(self, q=0.5):
        self.p = torch.tensor([1-q,q])
        self.q = q
        
    def __call__(self, tensor):
        if torch.multinomial(self.p,1)>0:
          tensor = self.synthetic7_matrix(tensor)
        return tensor

    def synthetic7_matrix(self,w):
      "Rotates W (square matrix) by one quadrant"
      m,m = w[0].size()
      w_new = torch.rand((m,m))

      for i in range(m):
        for j in range(m):
          w_new[i,j] = w[self.get_next_indices_sc7(i,j,m,m)]

      return w_new.view(1,m,m)

    def get_next_indices_sc7(self,i,j,m,h):
        "works for even m,h for now."
        if i>=int(m/2):
          i_next, j_next = i,(j+int(h/2))%h
        else:
          i_next, j_next = i,j

        return (0,i_next,j_next)

class RandomVertical(object):
    def __init__(self, q=0.5, d=1):
        self.p = torch.tensor([1-q,q])
        self.d = d #displacement magnitude
        self.q = q
        
    def __call__(self, tensor):
        if torch.multinomial(self.p,1)>0:
          tensor = self.vtrans_matrix(tensor,self.d)
        return tensor

    def vtrans_matrix(self,w,d=1):
      "translate vertical"
      m,m = w[0].size()
      w_new = torch.rand((m,m))

      for i in range(m):
        for j in range(m):
          w_new[i,j] = w[self.get_next_indices_sc8(i,j,m,m,d)]

      return w_new.view(1,m,m)

    def get_next_indices_sc8(self,i,j,m,h,d):
        "one step vertical translation"
        i_next, j_next = (i+d)%m,j

        return (0,i_next,j_next)

class RandomHorizontal(object):
    def __init__(self, q=0.5, d=1):
        self.p = torch.tensor([1-q,q])
        self.d = d #displacement magnitude
        self.q = q
        
    def __call__(self, tensor):
        if torch.multinomial(self.p,1)>0:
          tensor = self.htrans_matrix(tensor,self.d)
        return tensor

    def htrans_matrix(self,w,d=1):
      "translate vertical"
      m,m = w[0].size()
      w_new = torch.rand((m,m))

      for i in range(m):
        for j in range(m):
          w_new[i,j] = w[self.get_next_indices_sc9(i,j,m,m,d)]

      return w_new.view(1,m,m)

    def get_next_indices_sc9(self,i,j,m,h,d):
        "one step vertical translation"
        i_next, j_next = i,(j+d)%h

        return (0,i_next,j_next)

class CIFAR_dimension_3_to_1(object):
    def __call__(self, tensor):
        tensor = self.dimension_3_to_1(tensor)
        return tensor

    def dimension_3_to_1(self,w):
      "translate vertical"
      m,m = w[0].size()
      w_new = torch.rand((m,m))
      for i in range(m):
        for j in range(m):
          w_new[i,j] = (w[0,i,j] + w[1,i,j] + w[2,i,j])
      return w_new.view(1,m,m)





# Custom matrices

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
  m,m = w.size()
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
  m,m = w.size()
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
  m,m = w.size()
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
  m,m = w.size()
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
  m,m = w.size()
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
  m,m = w.size()
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
  m,m = w.size()
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

def vtrans_matrix(w,d=1):
  "translate vertical"
  m,m = w.size()
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

def htrans_matrix(w,d=1):
  "translate vertical"
  m,m = w.size()
  w_new = torch.rand((m,m))

  for i in range(m):
    for j in range(m):
      w_new[i,j] = w[get_next_indices_sc9(i,j,m,m,d)]

  return w_new.view(1,m,m)