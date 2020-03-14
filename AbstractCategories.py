# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:29:20 2020

@author: Simas
"""
from random import choice
import networkx as nx
import matplotlib.pyplot as plt
from functools import *




        

def rand_category(n_obj,n_mors):
    def choose_composite(mors,f,g):
        if g[1][0]!=f[1][1]:
            return None
                
        h = choice(list(filter(lambda h_:(h_[1][0]==f[1][0]) and (h_[1][1]==g[1][1]),mors)))
        
        return h
        
        
    
    
    objs = range(n_obj)
    mors = []
    
    
    for m in range(n_mors):
        mor = [choice(objs),choice(objs)]
        mors.append((m,mor))
    nn = m
    
    
    comps2 = {}
    new = []
    for f in mors:
        comps = {}
        for g in mors:
            try:
                h=choose_composite(mors,f,g)
                if h==None:
                    continue
                comps[g[0]]=h[0]
            except:
                nn+=1
                comps[g[0]]=nn
                new.append((nn,[f[1][0],g[1][1]]))
        print(comps)
        comps2[f[0]]=comps
          #TODO: add new morphisms and their compositions to the category 

            
        

    
    while new!=[]:
        print(nn)
        f=new.pop()
        mors.append(f)
        for g in mors:
            try:
                h=choose_composite(mors,f,g)
                if h==None:
                    continue
                comps2[g[0]][f[0]]=h[0]
            except:
                nn+=1
                comps2[g[0]][f[0]]=nn
                new.append((nn,[f[1][0],g[1][1]]))
                for h2 in mors:
                    h3 = choose_composite(mors,h2,f)
                    if h3==None:
                        continue
                    comps2[g[0]][f[0]]=h3[0]
    
    return list(zip(mors,comps2.values()))


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






n_ob,n_mo=5,4


C=rand_category(n_ob,n_mo)
for mor in C:
    print(mor)

draw_cat(n_ob,C)

for f in C:
    for g in C:
        print(f[0][0],g[0][0],compose(C,[f,g]))




            