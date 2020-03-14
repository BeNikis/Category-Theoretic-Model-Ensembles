# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:29:20 2020

@author: Simas
"""
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
    nn = m
    
    
    comps2 = {}
    new = []
    for f in mors:
        print(f)
        comps = {}
        for g in mors:
            try:
                h=choose_composite(mors,f,g)
                if h==None:
                    continue
                comps[g[0]]=h[0]
            except:
                mors.append((len(mors),[choice(objs),choice(objs)]))

        comps2[f[0]]=comps

    return list(zip(mors,map(lambda f:comps2[f[0]],mors)))


def compose(cat,expr):
    c = expr[0]
    for g in expr[1:]:
        try:
            c=cat[c[1][g[0][0]]]
        except:
            return None
            
    return c[0][0]

def draw_cat(n_ob,C):
    G=nx.DiGraph()
    
    map(lambda n:G.add_node(n),range(n_ob))
    
    for mor in C:
        G.add_edge(mor[0][1][0],mor[0][1][1])
    
    nx.draw(G, with_labels=True, font_weight='bold')
    return






n_ob,n_mo=2,5


C=rand_category(n_ob,n_mo)
for mor in C:
    print(mor)

draw_cat(n_ob,C)
# =============================================================================
# 
# for f in C:
#     for g in C:
#         print(f[0][0],g[0][0],compose(C,[f,g]))
# =============================================================================




            