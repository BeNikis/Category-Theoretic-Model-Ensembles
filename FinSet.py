# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:08:51 2020

@author: Simas
"""
from random import choice
import copy
import itl



class Object:

    def __init__(self,name,shape,eq=lambda x,y:abs(x-y)):
        self.name = name
        self.shape=shape
        self.eq=eq


        self.special={}
        
    def named_morphisms(self):
        return self.special
    
    def __str__(self):
        return self.name
    
    def __call__(self,x,y):
        return self.eq(x,y)

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
        
        try:
            self.ms[f.src][f.tgt].append(f)
        except:
            self.ms[f.src]={}
            self.ms[f.src][f.tgt]=[f]
        
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
        
    def all_paths_from(self,src,visited=[]):

        paths=[]
        step=self.one_step(src)
        for f in step:
            if f in paths:
                continue
            paths.append(f)
            
            if f.tgt in visited:
                continue
            
            paths+=map(lambda g:f.compose(g),self.one_step(f.tgt))
        
        return paths
            
            
                
    def diagram_commute_scores(self):
        paths = []
        score = 0 
        
        for d in self.input:
            
            paths += map(lambda f:d.compose(f),self.one_step(d.tgt))
            
                
            
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



C=Category()

E,N=Object("A",10),Object("B",10)

f=Morphism("f",rand_fin_f(E.shape,N.shape),E,N)
g=Morphism("g",rand_fin_f(E.shape,N.shape),N,E)
d = DataMorphism("data",[],N)

C.add_morphism(f)
C.add_morphism(g)
C.add_morphism(Morphism("h",rand_fin_f(E.shape,N.shape),E,N))

print(f.compose(g)(2))

print(list(map(str,C.one_step(E))))
print(list(map(str,C.all_paths_from(E))))


    