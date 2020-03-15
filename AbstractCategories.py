# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:29:20 2020

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

def choose_composite(mors,f,g):
    if g[1][0]!=f[1][1]:
        return None
    
    mors2 = list(filter(lambda h_:(h_[1][0]==f[1][0]) and (h_[1][1]==g[1][1]),mors))
    h = choice(mors2)
    
        
    return h

    

def rand_category(n_obj,n_mors):
    objs = range(n_obj)
    mors = []
    
    
    for m in range(n_mors):
        mor = [choice(objs),choice(objs)]
        mors.append((m,mor))
    
    
    comps2 = {}
    for f in mors:
        comps = {}
        for g in mors:
            try:
                h=choose_composite(mors,f,g)
                if h==None:
                    continue
                comps[g[0]]=h[0]
            except:
                mors.append((len(mors),[f[1][0],g[1][1]]))
                comps[g[0]]=mors[-1][0]

        
        comps2[f[0]]=comps

    return list(zip(mors,map(lambda f:comps2[f[0]],mors)))


def compose(cat,expr):
    c = expr[0]
    for g in expr[1:]:
        try:
            c=cat[c[1][g[0][0]]]
        except:
            return None
    
           
    return c

def draw_cat(n_ob,C):
    G=nx.DiGraph()
    
    map(lambda n:G.add_node(n),range(n_ob))
    
    for mor in C:
        G.add_edge(mor[0][1][0],mor[0][1][1])
    
    nx.draw(G, with_labels=True, font_weight='bold')
    return

def cat2tensor(C):
    out = []
    for mor in C:
        out+=[0,mor[0][0],1,mor[0][1][0],1,mor[0][1][1]]
        
        for k,v in mor[1].items():
            out+=[0,k,0,v]
    
    out+=[2,0]
    return out

def expr2tensor(expr):
    out = []
    for f in expr:
        out+=[0,f]
        
    return out
    

def rand_expr(C):
    l=choice(range(1,20))
    expr=[]
    for i in range(l):
        expr.append(choice(C)[0][0])
    
    return expr
n_ob,n_mo=2,15


C=rand_category(n_ob,n_mo)
for mor in C:
    print(mor)

t=cat2tensor(C)
draw_cat(n_ob,C)

#---------------
def plot_data(datapoints):
    
    x=np.arange(len(datapoints))
    y=[]
    for i in range(len(datapoints[0])):
        y.append(x)
        y.append(list(map(lambda p:p[i],datapoints)))
    plt.plot(*y)
    plt.show()
    
    return


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


def one_hot_compare(x,y):
    o=[]
    for i in range(len(x)):
        out = [0,0]
        out[int(x==y)]=1
        o.append(out)
    
    return torch.tensor(np.array(o),dtype=torch.float32)

torch.set_default_dtype(torch.float32)

rnn = nn.LSTM(2,16,3)
net = create_fc_network(16*3,30,3,1)
qann=lambda x:net(x)-1

opt = optim.Adam(list(rnn.parameters())+list(net.parameters()))

loss=nn.MSELoss()

err_plot = []



batchsize=25

numornone=lambda x:-1 if x==None else x
try:
    for e in range(1000):
        C=rand_category(2,15)
        Ct=cat2tensor(C)
        batch = []
        for q in range(batchsize):
            ex=rand_expr(C)
            item=[np.array(Ct+expr2tensor(ex)),np.array([numornone(compose(C,ex))])]
            #print(item)
            batch.append([torch.tensor(item[0],dtype=torch.float32),torch.tensor(item[1],dtype=torch.float32)])
        
        opt.zero_grad()
        
        arr=nn.utils.rnn.pad_sequence(list(map(lambda i:i[0],batch)))
        
        inp=arr.view(-1,batchsize,2)
        nno,hidden=rnn(inp)[-1]
        
        
        out=qann(nno.view(batchsize,3*16))
        l = loss(out,torch.tensor(list(map(lambda b:b[1],batch)),dtype=torch.float32)).mean()
        
        err_plot.append([l.item()])
        print(e,err_plot[-1])
        l.backward()
        opt.step()
except KeyboardInterrupt:
    pass        
plot_data(err_plot)  
        




            