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
            c=cat[c][1][g]
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
    div=len(C)
    for mor in C:
        out+=[0,mor[0][0]/div,1,mor[0][1][0],1,mor[0][1][1]/div]
        
        for k,v in mor[1].items():
            out+=[0,k/div,0,v/div]
    
    out+=[2,0]
    return out

def expr2tensor(expr,div=0):
    out = []
    
    for f in expr:
        out+=[1,f/div]
        
    return out
    

def rand_expr(C):
    l=choice(range(1,10))
    expr=[choice(C)[0][0]]
    for i in range(l):
        try:
            chosen = choice(list(C[expr[-1]][1].keys()))
            expr.append(chosen)
        except:
            return expr
        
    return expr
n_ob,n_mo=2,5


C=rand_category(n_ob,n_mo)
for mor in C:
    print(mor)

t=cat2tensor(C)
draw_cat(n_ob,C)
plt.show()

for i in range(10):
    ex=rand_expr(C)
    print(ex,"=",compose(C,ex))
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
batchsize=25
embedding=2

encoder_layer = nn.TransformerEncoderLayer(d_model=embedding, nhead=2)
rnn = nn.TransformerEncoder(encoder_layer,3)

encdecnet = nn.Transformer(2,2,2,2,32)#nn.Sequential(*[nn.LSTM(2,32,3),nn.LSTM(32,2)])#
encdec = lambda x,y:encdecnet(x,y)-1

net = create_fc_network(embedding,30,3,1)
qann=lambda x:net(x)-1

opt = optim.Adam(encdecnet.parameters())

loss=nn.MSELoss()

err_plot = []





numornone=lambda x,div:[1,-1] if x==None else [1,x/div]
try:
    for e in range(10000):
        
        C=rand_category(choice(range(n_ob,n_ob+7)),choice(range(n_mo,n_mo+7)))
        #for m in C:
        #    print(m)
            
        
        Ct=cat2tensor(C)
        batch = []#torch.tensor([[Ct]])
        
        
        for q in range(batchsize):
            ex=rand_expr(C)
            ext=expr2tensor(ex,float(len(C)))
            item=[np.array(Ct+ext),np.array([numornone(compose(C,ex),len(C))])]
            #item[1][0][1]=item[1][0][1]/float(len(C))
            #item[0]=item[0].reshape(-1,embedding)
            
            batch.append([torch.tensor(item[0],dtype=torch.float32),torch.tensor(item[1],dtype=torch.float32)])
        
        opt.zero_grad()
        
        #inp=list(map(lambda i:i[0],batch))
        
        #inp=inp.view(-1,25,2)
        #print(inp)
        #inp=torch.stack(inp)
        #nno=rnn(inp)[-1]
        
        #inp=batch[0][0]
        #batch=batch.view(-1,1,2)
        
        inp=nn.utils.rnn.pad_sequence(list(map(lambda i:i[0],batch)))
        inp=inp.view(-1,batchsize,embedding)
        
        #inp = torch.stack(inp)
        outp = torch.stack(list(map(lambda i:i[1],batch)))
        outp = outp.view(-1,batchsize,embedding)
        
        out=encdec(inp,outp)#qann(nno)
        
        l = loss(out,outp).mean()#loss(out,torch.tensor(list(map(lambda b:b[1],batch)),dtype=torch.float32)).mean()
        
        err_plot.append([l.item()])
        print(e,err_plot[-1])
        l.backward()
        opt.step()
except KeyboardInterrupt:
    pass        
plot_data(err_plot)  
        




            