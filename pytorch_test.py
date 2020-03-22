# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 06:27:03 2020

@author: Simas
"""

import os,sys

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



from random import choice
import networkx as nx
import matplotlib.pyplot as plt
from functools import *

from FinSet import *

torch.set_default_dtype(torch.float32)


def create_fc_layers(input_size,width,depth,out_size,hidden_nonlin=nn.ReLU,out_nonlin=nn.ReLU):
    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

    
    
    layers = []
    layers.append(Flatten())
    
    layers.append(nn.Linear(input_size,width))
    nn.init.xavier_uniform_(layers[-1].weight)
    layers.append(hidden_nonlin())
    layers.append(nn.BatchNorm1d(width))
    
    if (depth>=2):
        for i in range(depth-1):
            layers.append(nn.Linear(width,width))
            nn.init.xavier_uniform_(layers[-1].weight)
            layers.append(hidden_nonlin())
            layers.append(nn.BatchNorm1d(width))
        
    layers.append(nn.Linear(width,out_size))
    nn.init.xavier_uniform_(layers[-1].weight)
    layers.append(out_nonlin())
    
    return layers

def create_fc_network(input_size,width,depth,out_size,hidden_nonlin=nn.ReLU,out_nonlin=nn.ReLU):
    return nn.Sequential(*create_fc_layers(input_size,width,depth,out_size,hidden_nonlin,out_nonlin))

def plot_data(datapoints):
    
    x=np.arange(len(datapoints))
    y=[]
    for i in range(len(datapoints[0])):
        y.append(x)
        y.append(list(map(lambda p:p[i],datapoints)))
    plt.plot(*y)
    plt.show()
    
def create_discrete_network(input_size,width,depth,out_size,hidden_nonlin=nn.ReLU):
    net = create_fc_network(input_size,width,depth,out_size,hidden_nonlin=nn.ReLU,out_nonlin=nn.Softmax)
    
    return net

torch.set_default_dtype(torch.double)
size = 10


i = create_discrete_network(1,2,2,size)
iM = Morphism("i",lambda x:torch.argmax(i(x),dim=1),G,G)
m = create_discrete_network(2,2,2,size)
mM = Morphism("m",lambda x:torch.argmax(m(x),dim=1),GxG,G)


C=Category()
G = Object("G",[1],metric=nn.CrossEntropyLoss)
GxG = Object("GxG",[2],metric = lambda x,y:map(lambda pair:nn.CrossEntropyLoss()(pair[0],pair[1]),zip(x,y)))

C.add_morphism(iM)
C.add_morphism(mM)


t = torch.tensor(np.random.rand(10,2))
print(t)
print(m(t))
print(mM(t))

C.draw()