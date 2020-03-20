# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:08:51 2020

@author: Simas
"""
from random import choice
import copy
import itl

import networkx as nx
import matplotlib.pyplot as plt

class Object:

    def __init__(self,name,shape,metric=lambda x,y:abs(x-y)):
        self.name = name
        self.shape=shape
        self.metric=metric

        self.special={}
        
    def named_morphisms(self):
        return self.special
    
    def __str__(self):
        return self.name
    
    def __call__(self,x,y):
        return self.metric(x,y)
    
    def __eq__(self,other):
        return self.name==other.name
    
    def __hash__(self):
        return hash(self.name)

class Morphism:
    i=0
    def __init__(self,name,f,src,tgt):
        self.name = name
        self.f=f
        self.src=src
        self.tgt=tgt
        
        self.special = {}
        
    def __call__(self,x):
        y=x
        if type(self.f)==list:
            for m in self.f:
                y=m(y)
        else:
            y=self.f(x)
            
        return y
    
    def __str__(self):
        self.i+=1
        #print(self.i)
        if type(self.f)==list:
            return "*".join(map(str,self.f))
        return self.name
    
    def compose(self,*expr):
        #print(list(map(str,expr)))
        expr=(self if type(self.f)==list else [self])+list(expr)
        
        return Morphism(str(expr),expr,self.src,expr[-1].tgt)
    
    def named_morphisms(self):
        return self.special
    
    def add_named_morphism(self,m):
        self.special[m.name]=m

class DataMorphism(Morphism):
    def __init__(self,name,data,tgt):
        super(DataMorphism,self).__init__(name,data,None,tgt)
        
    def __call__(self,index=None):
        if not index:
            return []
        return self.f[index]
    
class Category:
    def __init__(self):
        self.os = set()
        self.ms = {}
        
        self.input = []
        return
    
    def add_morphism(self,f):
        self.os.add(f.src)
        self.os.add(f.tgt)
        
        if f.src not in self.ms.keys():
            self.ms[f.src]={}
            
        if f.tgt not in self.ms[f.src].keys():
            self.ms[f.src][f.tgt]=[f]
        else:
            self.ms[f.src][f.tgt].append(f)
            
        if f is DataMorphism:
            self.input.append(f)
            
            
        return
    
    def add_object(self,o):
        self.os.add(o)
            
        
        for m in o.special.values():
            self.add_morphism(m)
        self.ms[o]={}
        
    def subcategory(self,*morphisms):
        C=Category()
        map(C.add_morphism,morphisms)
        return C
    
    def commute_value(self,f,g,x):
        if (f.src!=g.src) or (f.tgt!=g.tgt):
            return None
        else:
            return f.tgt.eq(f(x),g(x))
    
    #get all morpisms going from src
    def one_step(self,src):
        out = []
    
        if src not in self.ms.keys():
            return []
    
        for tgt in self.ms[src].keys():
            out+=self.ms[src][tgt]
        return out
    
    #returns a dict of lists where the key is the target of the morphisms in the list,going from src
    def all_paths_from(self,src,visited=[]):
        def combine_dicts(d1,d2):
            
            comb = {}
            
            for k1,v1 in d1.items():
                comb[k1]=v1
                
            for k2,v2 in d2.items():
                if k2 not in comb.keys():
                    comb[k2]=[]
                    
                comb[k2]+=v2
            
            #print("COMB",print_dict(d1),print_dict(d2),print_dict(comb))
            return comb
            
        paths={}
        step=self.one_step(src)
#        visited.append(src)
        #print(list(map(str,visited)))
        for f in step:
            #print(str(f),{str(k):list(map(str,v)) for k,v in paths})
            if f.tgt not in paths.keys():
                paths[f.tgt]=[]

            #if f in paths[f.tgt]:
            #    continue
            
            paths[f.tgt].append(f)
            
            if f.tgt in visited:
                continue
            
           
            paths=combine_dicts(paths,{k:[f.compose(g) for g in v] for k,v in self.all_paths_from(f.tgt,visited+[f.tgt]).items()})
            
        
        #print("VIS",list(map(str,visited)))
        return paths
            
            
                
    def diagram_commute_scores(self):
        paths = []
        score = 0 
        
        for d in self.input:
            
            paths += map(lambda f:d.compose(f),self.one_step(d.tgt))
            
    def draw(self):
        G=nx.DiGraph()
        
        map(lambda n:G.add_node(n),self.os)
        
        for src in self.ms.keys():
            for tgt in self.ms[src].keys():
                G.add_edge(src,tgt)
        
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()
        return          
    
def print_dict(d):
    return {str(k) : list(map(str,v)) for k,v in d.items()}
            
def rand_fin_f(srcS,tgtS):
    f=[choice(range(tgtS)) for i in range(srcS)]
    #print(f)
    return lambda x:f[x]

def I(f,x):
    yM=itl.entM(f(x))
    
    if type(f)==DataMorphism:
        return itl.MatrixEntropy.apply(yM)
    xM=itl.entM(x)
    
    
    return itl.mutual_information(yM,xM)

if __name__=="__main__":
    n_obs=5
    n_mors=15
    C=Category()
    
    obs = [Object(str(i),[5]) for i in range(n_obs)]
    
    
    for i in range(n_obs-1):
        C.add_morphism(Morphism(str(i)+"->"+str(i+1),rand_fin_f(5,5),obs[i],obs[i+1]))
    
    C.add_morphism(Morphism(str(n_obs-1)+"->"+str(0),rand_fin_f(5,5),obs[-1],obs[0]))
    C.add_morphism(Morphism(str(0)+"->A",rand_fin_f(5,5),obs[0],Object("A",[5])))
# =============================================================================
#     
#     
#     objs = [Object(str(i),[5]) for i in range(n_obs)]
#     
#     for i in range(n_obs):
#         C.add_object(objs[i])
#     
#     for i in range(n_mors):
#         o1,o2=choice(objs),choice(objs)
#         C.add_morphism(Morphism(str(o1)+"->"+str(o2),rand_fin_f(5,5),o1,o2))
#         
#     
# =============================================================================
    C.draw()
    print(print_dict(C.ms[obs[0]]))
    for ob in obs:
        print(str(ob)+" : " ,print_dict(C.all_paths_from(ob)))




    