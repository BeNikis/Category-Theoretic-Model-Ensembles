# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:54:58 2020

@author: Simas
"""

from random import *
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
        if type(x)==list:
            ret = list(map(lambda pair:self.metric(pair[0],pair[1]),zip(x,y)))
        else:
            ret = self.metric(x,y)
        return ret
    
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
        
    def __call__(self,x=None):
        y=x
        if type(self)==DataMorphism:
            y=self.data
        
        if type(self.f)==list:
            for m in self.f:
                if type(y)!=list:
                    y=m(y)
                else:
                    y=list(map(m,y))
        else:
            y=self.f(x)
            
        return y
    
    def __str__(self):
        self.i+=1
        #print(self.i)
        if type(self.f)==list:
            return "*".join(map(str,self.f))
        return self.name
    
    def __repr__(self):
        return self.__str__()
    
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
        super(DataMorphism,self).__init__(name,None,None,tgt)
        self.data=data
        
    def __call__(self,index=None):
        if not index:
            return self.data
        return self.data[index]
    

        

    
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
            
        if type(f)==DataMorphism:
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
    #exclude is a list of morphisms/objects to not use
    def all_paths_from(self,src,exclude=[],visited=[]):
        
            
        paths={}
        step=self.one_step(src)
#        visited.append(src)
        #print(list(map(str,visited)))
        for f in step:
            if (f in exclude) or (f.tgt in exclude):
                continue
            #print(str(f),{str(k):list(map(str,v)) for k,v in paths})
            if f.tgt not in paths.keys():
                paths[f.tgt]=[]

            #if f in paths[f.tgt]:
            #    continue
            
            paths[f.tgt].append(f)
            
            if f.tgt in visited:
                continue
            
           
            paths=combine_dicts(paths,{k:[f.compose(g) for g in v] for k,v in self.all_paths_from(f.tgt,exclude,visited+[f.tgt]).items()})
            
        
        #print("VIS",list(map(str,visited)))
        return paths
            
            
                
    def diagram_commute_scores(self):
        paths = {d.tgt:{} for d in self.input}

        for d in self.input: 
            
            paths[d.tgt] = {k:list(map(lambda f:d.compose(f),v)) for k,v in self.all_paths_from(d.tgt).items()}
        
        scores = {}
        visited = []
        for src in paths.keys():
            for tgt in paths[src].keys():
                for p1 in paths[src][tgt]:

                    for p2 in paths[src][tgt]:
                        
                        if (p2,p1) in visited:
                            continue
                        visited.append((p1,p2))
                        
                        if p1!=p2:
                            if p1 not in scores.keys():
                                scores[p1]={}
                                
                            scores[p1][p2]=p1.tgt(p1(),p2())
        
        return scores

                            
            
    def draw(self):
        G=nx.DiGraph()
        
        map(lambda n:G.add_node(n),self.os)
        
        for src in self.ms.keys():
            for tgt in self.ms[src].keys():
                G.add_edge(src,tgt)
        
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()
        return          

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
   
def print_dict(d):
    return {str(k) : list(map(str,v)) for k,v in d.items()}


#main interface - could represent a categoryy of states/functors/categories etc.
class MetricStateCategory:
    def __init__(self,equality_metric,init_state=None):
        self.state = init_state
        self.eq = equality_metric
        return
    
    def update(self,delta,save=True):
        if save:
            self.state=delta(self.state)
        return self.state if save else delta(self.state)
    
    
class Adjunction:
    def __init__(self,agent_state,environment_state,percieve_functor,guess_functor):
        self.a = agent_state
        self.e = environment_state
        
        #e -> a
        self.P = percieve_functor
        
        #a -> e
        self.G = guess_functor
        
    def agent_error(self):
        return self.a.eq(self.a,self.P(self.G(self.a)))
    
    def perception_error(self):
        return self.e.eq(self.e,self.G(self.P(self.e)))
    
    
    #agent_delta_creator makes an update morphism from the current data
    def optimize(self,delta_creator,criterion,steps=1):
        for i in range(steps):
            delta=delta_creator(self)
            
            if criterion(self,delta):
                new = delta(self)
                self.a = new.a
                self.e = new.e
                self.P = new.P
                self.G = new.G
            
    
            
        
#test
      

def list_dist(x,y):
    d=0
    for i in range(min(len(x.state),len(y.state))):
        d+=abs(x.state[i]-y.state[i])

    return d 

def opt_criterion(adj,delta):
    new = delta(adj)
    
    return adj.perception_error()>new.perception_error()

def list_list_delta(adj):

    
    new_P=NumberListCat([randint(0,11)-5 for i in range(min(len(adj.a.state),len(adj.e.state)))])
    
    return lambda adj:Adjunction(new_P(adj.e),adj.e,new_P,lambda x:x)
    
               

class NumberListCat(MetricStateCategory):
    def __init__(self,init_state=None):
        super(NumberListCat,self).__init__(list_dist,init_state if init_state else [randint(0,100) for i in range(3)])
        
    def __call__(self,x):
        return NumberListCat(list(map (lambda pair:pair[0]+pair[1],zip(x.state,self.state))))

        
truestate = NumberListCat()
guesstate = NumberListCat()

P = NumberListCat()
G = lambda x:x

adj = Adjunction(guesstate,truestate,P,id)

for i in range(300):
    print(adj.e.state,adj.a.state,list_dist(adj.e,adj.a))
    adj.optimize(list_list_delta,opt_criterion)
    
        
        
    
    