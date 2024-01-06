import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math
# Loss functions


def loss_cross_entropy(epoch, y, t,class_list, ind, noise_or_not,loss_all,loss_div_all):
    ##Record loss and loss_div for further analysis
    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_all[ind,epoch] = loss_numpy
    return torch.sum(loss)/num_batch



        

def loss_cores(epoch, y, t,class_list, ind, noise_or_not,loss_all,loss_div_all, noise_prior = None):
    beta = f_beta(epoch)
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    loss = F.cross_entropy(y, t, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(y) + 1e-8)
    # sel metric
    loss_sel =  loss - torch.mean(loss_,1) 
    if noise_prior is None:
        loss =  loss - beta*torch.mean(loss_,1)
    else:
        loss =  loss - beta*torch.sum(torch.mul(noise_prior, loss_),1)
    
    loss_div_numpy = loss_sel.data.cpu().numpy()
    loss_all[ind,epoch] = loss_numpy
    loss_div_all[ind,epoch] = loss_div_numpy
    for i in range(len(loss_numpy)):
        if epoch <=30:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000
    else:
        return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)

def loss_my_cores(epoch, probs,y,device='cuda',noise_prior = None):
    """
    probs: after log softmax, query num * (negative num + positive num)
    return: (batch loss, loss_v) [loss_v dimension: query num * (negative num + positive num)]
    """
    beta = f_beta(epoch)
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    tag=torch.where(y==-1,1,0)
    loss = probs.flatten().mul(tag.long()).reshape(probs.shape)
    #loss = F.nll_loss(probs.flatten(), tag.long(),reduction='none') # NLL loss of negative passages
    loss_numpy = loss.flatten().data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)# 
    loss_div_numpy = float(np.array(0))
    loss_re = torch.mean(probs,dim=1) # confidence regularizer
    # sel metric
    loss_sel =  loss-loss_re.unsqueeze(dim=1) # this is for sample sieve
    if noise_prior is None:
        loss =  loss - beta*loss_re.unsqueeze(dim=1)
    else:
        loss =  loss - beta*loss_re.unsqueeze(dim=1)
    
    loss_div_numpy = loss_sel.flatten().data.cpu().numpy()
    for i in range(len(loss_numpy)):
        if epoch <=0 or y[i]==1:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).to(device)
    loss_final = loss_v_var * loss.flatten()
    if sum(loss_v) == 0.0:
        return torch.mean(loss_final)/100000000
    else:
        return torch.sum(loss_final)/sum(loss_v), loss_v.reshape(probs.shape).astype(int),loss_div_numpy
def loss_my_cores_pos(epoch, probs,y,device='cuda',noise_prior = None):
    """
    probs: after log softmax, query num * (negative num + positive num)
    return: (batch loss, loss_v) [loss_v dimension: query num * (negative num + positive num)]
    """
    beta = f_beta(epoch)
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    tag=torch.where(y==1,1,0)
    loss = (-1)*probs.flatten().mul(tag.long()).reshape(probs.shape)
    #loss = F.nll_loss(probs.flatten(), tag.long(),reduction='none') # NLL loss of negative passages
    loss_numpy = loss.flatten().data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)# 
    loss_div_numpy = float(np.array(0))
    loss_re = (-1)*torch.mean(probs,dim=1) # confidence regularizer
    # sel metric
    loss_sel =  (-1)*probs-loss_re.unsqueeze(dim=1) # this is for sample sieve
    if noise_prior is None:
        loss =  loss - beta*loss_re.unsqueeze(dim=1)
    else:
        loss =  loss - beta*loss_re.unsqueeze(dim=1)
    
    loss_div_numpy = loss_sel.flatten().data.cpu().numpy()
    for i in range(len(loss_numpy)):
        if epoch <=1 or y[i]==1:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).to(device)
    loss_final = loss_v_var * loss.flatten()
    if sum(loss_v) == 0.0:
        return torch.mean(loss_final)/100000000
    else:
        return torch.sum(loss_final)/sum(loss_v), loss_v.reshape(probs.shape).astype(int),probs.flatten().data.cpu().numpy()
    
def loss_my_cores_positive_v2(epoch, probs,y,device='cuda',noise_prior = None):
    """
    probs: after log softmax, query num * (negative num + positive num)
    return: (batch loss, loss_v) [loss_v dimension: query num * (negative num + positive num)]
    """
    beta = f_beta(epoch)
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    tag=torch.where(y==1,1,0).long().flatten()
    loss = (-1)*probs.flatten().mul(tag)[tag.to(torch.bool)]
    #loss = F.nll_loss(probs.flatten(), tag.long(),reduction='none') # NLL loss of negative passages
    num_batch = len(tag)
    loss_v = np.ones(num_batch)# 
    loss_re = (-1)*torch.mean(probs,dim=1) # confidence regularizer
    if noise_prior is None:
        loss =  loss - beta*loss_re
    else:
        loss =  loss - beta*loss_re
    loss_v = loss_v.astype(np.float32)
    loss_final = loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_final)/100000000
    else:
        return torch.sum(loss_final)/sum(loss_v), loss_v.reshape(probs.shape).astype(int),probs.flatten().data.cpu().numpy()

def f_beta(epoch):
    beta1 = np.linspace(0.1, 0.1, num=10)
    beta2 = np.linspace(0.25, 0.25, num=2)
    beta3 = np.linspace(0.25, 1.0, num=5)
 
    beta = np.concatenate((beta1,beta2,beta3),axis=0) 
    return beta[epoch]