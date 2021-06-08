# graph search based fast equivariance algorithm
import torch
from torch.autograd import Variable
import numpy as np

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

def get_next_indices_sc2(i,j,m,h):
    "works for even m,h for now."
    i_next, j_next = i,(j+int(h/2))%h

    return (i_next,j_next)

def get_next_indices_sc3(i,j,m,h):
    "works for even m,h for now."
    i_next, j_next = (i+int(m/2))%m,j

    return (i_next,j_next)

def get_next_indices_sc4(i,j,m,h):
    "works for even m,h for now."
    if j<int(h/2):
      i_next, j_next = (i+int(m/2))%m,j
    else:
      i_next, j_next = i,j

    return (i_next,j_next)

def get_next_indices_sc5(i,j,m,h):
    "works for even m,h for now."
    if j>=int(h/2):
      i_next, j_next = (i+int(m/2))%m,j
    else:
      i_next, j_next = i,j

    return (i_next,j_next)

def get_next_indices_sc6(i,j,m,h):
    "works for even m,h for now."
    if i<int(m/2):
      i_next, j_next = i,(j+int(h/2))%h
    else:
      i_next, j_next = i,j

    return (i_next,j_next)

def get_next_indices_sc7(i,j,m,h):
    "works for even m,h for now."
    if i>=int(m/2):
      i_next, j_next = i,(j+int(h/2))%h
    else:
      i_next, j_next = i,j

    return (i_next,j_next)

def get_next_indices_vert_trans(i,j,m,h,d):
    "one step vertical translation"
    i_next, j_next = (i+d)%m,j

    return (i_next,j_next)

def get_next_indices_hori_trans(i,j,m,h,d):
    "one step vertical translation"
    i_next, j_next = i,(j+d)%h

    return (i_next,j_next)

def get_G_list_equivariance_indices_graph(m,h,eq_array,I_prev,d=4):
  "Size of input layer = m*m; size of hidden layer = h*h; I permuted indices"
  "d=4 is the default translation step used"

  nx = m*m #size of X
  nh = h*h #size of hidden layer
  I = I_prev #records the permutation of indices of W, which is of size n*n. For no previous symmetry, I_prev is given as the original indices
  V = np.zeros(nx*nh)-1 #-1 if not visited, else 1
  current_orbit = -1
  indices_queue = []
  indices_queue_pair = []

  for i in range(nx):
    for j in range(nh):
      i_0 = i
      j_0 = j
      if V[i_0*nh+j_0] < 0: 
        V[i_0*nh+j_0] = 1
        index  = i_0*nh+j_0
        indices_queue.append(index)
        indices_queue_pair.append([i_0,j_0])
        current_orbit += 1
        while len(indices_queue)>0:
          index = indices_queue.pop(0)
          i_0, j_0 = indices_queue_pair.pop(0)
          I[index] = current_orbit

          if eq_array[0]==1:      
            # rotations  
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h

            #rotate by 90
            alpha_r, beta_r = beta, m-1 - alpha
            gamma_r, delta_r = delta, h-1 - gamma

            i_r = alpha_r*m + beta_r
            j_r = gamma_r*h + delta_r

            index_r = i_r*nh+j_r
            if V[index_r] < 0:
              V[index_r] = 1
              I[index_r] = current_orbit
              indices_queue.append(index_r)
              indices_queue_pair.append([i_r,j_r])

            #rotate by 180
            alpha_rr, beta_rr = beta_r, m-1 - alpha_r
            gamma_rr, delta_rr = delta_r, h-1 - gamma_r

            i_rr = alpha_rr*m + beta_rr
            j_rr = gamma_rr*h + delta_rr

            index_rr = i_rr*nh+j_rr
            if V[index_rr] < 0:
              V[index_rr] = 1
              I[index_rr] = current_orbit
              indices_queue.append(index_rr)
              indices_queue_pair.append([i_rr,j_rr])

            #rotate by 270
            alpha_rrr, beta_rrr = beta_rr, m-1 - alpha_rr
            gamma_rrr, delta_rrr = delta_rr, h-1 - gamma_rr

            i_rrr = alpha_rrr*m + beta_rrr
            j_rrr = gamma_rrr*h + delta_rrr

            index_rrr = i_rrr*nh+j_rrr
            if V[index_rrr] < 0:
              V[index_rrr] = 1
              I[index_rrr] = current_orbit
              indices_queue.append(index_rrr)
              indices_queue_pair.append([i_rrr,j_rrr])

          if eq_array[1]==1:      
            # horizontal flip
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h

            #hflip
            alpha_h, beta_h = alpha, m-1 - beta
            gamma_h, delta_h = gamma, h-1 - delta

            i_h = alpha_h*m + beta_h
            j_h = gamma_h*h + delta_h

            index_h = i_h*nh+j_h
            if V[index_h] < 0:
              V[index_h] = 1
              I[index_h] = current_orbit
              indices_queue.append(index_h)
              indices_queue_pair.append([i_h,j_h])
          
          if eq_array[2]==1:     
            # vertical flip
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h

            #vflip
            alpha_h, beta_h = m-1-alpha, beta
            gamma_h, delta_h = h-1-gamma, delta

            i_h = alpha_h*m + beta_h
            j_h = gamma_h*h + delta_h

            index_h = i_h*nh+j_h
            if V[index_h] < 0:
              V[index_h] = 1
              I[index_h] = current_orbit
              indices_queue.append(index_h)
              indices_queue_pair.append([i_h,j_h])

          if eq_array[10]==1:      
            # synthetic 1
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h

            #move to quadrant 2
            alpha_q, beta_q = get_next_indices_sc1(alpha,beta,m,m)
            gamma_q, delta_q = get_next_indices_sc1(gamma,delta,h,h)

            i_q = alpha_q*m + beta_q
            j_q = gamma_q*h + delta_q
            
            index_q = i_q*nh+j_q
            if V[index_q] < 0:
              V[index_q] = 1
              I[index_q] = current_orbit
              indices_queue.append(index_q)
              indices_queue_pair.append([i_q,j_q])

            #move to quadrant 3
            alpha_qq, beta_qq = get_next_indices_sc1(alpha_q,beta_q,m,m)
            gamma_qq, delta_qq = get_next_indices_sc1(gamma_q,delta_q,h,h)

            i_qq = alpha_qq*m + beta_qq
            j_qq = gamma_qq*h + delta_qq

            index_qq = i_qq*nh+j_qq
            if V[index_qq] < 0:
              V[index_qq] = 1
              I[index_qq] = current_orbit
              indices_queue.append(index_qq)
              indices_queue_pair.append([i_qq,j_qq])

            #move to quadrant 4
            alpha_qqq, beta_qqq = get_next_indices_sc1(alpha_qq,beta_qq,m,m)
            gamma_qqq, delta_qqq = get_next_indices_sc1(gamma_qq,delta_qq,h,h)

            i_qqq = alpha_qqq*m + beta_qqq
            j_qqq = gamma_qqq*h + delta_qqq

            index_qqq = i_qqq*nh+j_qqq
            if V[index_qqq] < 0:
              V[index_qqq] = 1
              I[index_qqq] = current_orbit
              indices_queue.append(index_qqq)
              indices_queue_pair.append([i_qqq,j_qqq])

          if eq_array[11]==1:      
            # synthetic 2
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h

            #move to next half
            alpha_q, beta_q = get_next_indices_sc2(alpha,beta,m,m)
            gamma_q, delta_q = get_next_indices_sc2(gamma,delta,h,h)

            i_q = alpha_q*m + beta_q
            j_q = gamma_q*h + delta_q
            
            index_q = i_q*nh+j_q
            if V[index_q] < 0:
              V[index_q] = 1
              I[index_q] = current_orbit
              indices_queue.append(index_q)
              indices_queue_pair.append([i_q,j_q])
          
          if eq_array[5]==1:
            # synthetic 3
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h

            #move to next half
            alpha_q, beta_q = get_next_indices_sc3(alpha,beta,m,m)
            gamma_q, delta_q = get_next_indices_sc3(gamma,delta,h,h)

            i_q = alpha_q*m + beta_q
            j_q = gamma_q*h + delta_q

            index_q = i_q*nh+j_q
            if V[index_q] < 0:
              V[index_q] = 1
              I[index_q] = current_orbit
              indices_queue.append(index_q)
              indices_queue_pair.append([i_q,j_q])

          if eq_array[6]==1:
            # synthetic 4
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h

            #move to next half
            alpha_q, beta_q = get_next_indices_sc4(alpha,beta,m,m)
            gamma_q, delta_q = get_next_indices_sc4(gamma,delta,h,h)

            i_q = alpha_q*m + beta_q
            j_q = gamma_q*h + delta_q

            index_q = i_q*nh+j_q
            if V[index_q] < 0:
              V[index_q] = 1
              I[index_q] = current_orbit
              indices_queue.append(index_q)
              indices_queue_pair.append([i_q,j_q])
          
          if eq_array[7]==1:
            # synthetic 5
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h

            #move to next half
            alpha_q, beta_q = get_next_indices_sc5(alpha,beta,m,m)
            gamma_q, delta_q = get_next_indices_sc5(gamma,delta,h,h)

            i_q = alpha_q*m + beta_q
            j_q = gamma_q*h + delta_q

            index_q = i_q*nh+j_q
            if V[index_q] < 0:
              V[index_q] = 1
              I[index_q] = current_orbit
              indices_queue.append(index_q)
              indices_queue_pair.append([i_q,j_q])

          if eq_array[8]==1:
            # synthetic 6
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h

            #move to next half
            alpha_q, beta_q = get_next_indices_sc6(alpha,beta,m,m)
            gamma_q, delta_q = get_next_indices_sc6(gamma,delta,h,h)

            i_q = alpha_q*m + beta_q
            j_q = gamma_q*h + delta_q

            index_q = i_q*nh+j_q
            if V[index_q] < 0:
              V[index_q] = 1
              I[index_q] = current_orbit
              indices_queue.append(index_q)
              indices_queue_pair.append([i_q,j_q])

          if eq_array[9]==1:
            # synthetic 7
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h

            #move to next half
            alpha_q, beta_q = get_next_indices_sc7(alpha,beta,m,m)
            gamma_q, delta_q = get_next_indices_sc7(gamma,delta,h,h)

            i_q = alpha_q*m + beta_q
            j_q = gamma_q*h + delta_q

            index_q = i_q*nh+j_q
            if V[index_q] < 0:
              V[index_q] = 1
              I[index_q] = current_orbit
              indices_queue.append(index_q)
              indices_queue_pair.append([i_q,j_q])

          if eq_array[4]==1:
            # vertical translation 
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h
            for it in range(m):
              #translate one step
              alpha, beta = get_next_indices_vert_trans(alpha,beta,m,m,d)
              gamma, delta = get_next_indices_vert_trans(gamma,delta,h,h,d)

              i_q = alpha*m + beta
              j_q = gamma*h + delta

              index_q = i_q*nh+j_q
              if V[index_q] < 0:
                V[index_q] = 1
                I[index_q] = current_orbit
                indices_queue.append(index_q)
                indices_queue_pair.append([i_q,j_q])

              it += d-1

          if eq_array[3]==1:
            # horizontal translation
            alpha, beta = i_0//m, i_0%m
            gamma, delta = j_0//h, j_0%h
            for it in range(h):
              #translate one step
              alpha, beta = get_next_indices_hori_trans(alpha,beta,m,m,d)
              gamma, delta = get_next_indices_hori_trans(gamma,delta,h,h,d)

              i_q = alpha*m + beta
              j_q = gamma*h + delta

              index_q = i_q*nh+j_q
              if V[index_q] < 0:
                V[index_q] = 1
                I[index_q] = current_orbit
                indices_queue.append(index_q)
                indices_queue_pair.append([i_q,j_q])

              it += d-1
  print("current_orbit",current_orbit)
  return I