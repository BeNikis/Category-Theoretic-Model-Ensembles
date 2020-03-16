# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:08:51 2020

@author: Simas
"""
from random import choice


class Object:
    def __init__(self,name,size,eq=lambda x,y:abs(x-y)):
        self.name = name
        self.size=size
        self.eq=eq

class Morphism:
    def __init__(self,f,src,tgt):
        self.f=f
        self.src=src
        self.tgt=tgt
        
    def __call__(self,x):
        y=x
        if type(self.f[0])!=int:
            for m in self.f:
                y=m.f[y]
        else:
            y=self.f[x]
            
        return y
    
    def compose(self,*expr):
        return Morphism([self] + list(expr),self.src,expr[-1].tgt)
    
    def mutate(self):
        if type(self.f[0])!=int:
            for m in self.f:
                m.mutate()
        else:
            for i in range(len(self.f)):
                if choice(range(5))==0:
                    self.f[i]=choice(range(self.tgt.size))
            

class Category:
    def __init__(self):
        self.os = set()
        self.ms = {}
        return
    
    def add_morphism(self,f):
        self.os.add(f.src)
        self.os.add(f.tgt)
        
        
            
        try:
            self.ms[f.src][f.tgt].append(f)
        except:
            self.ms[f.src]={}
            self.ms[f.src][f.tgt]=[f]
        
        return
    
    def add_object(self,o):
        self.os.add(o)
        
    def compose(self,*expr):
        return Morphism(expr,expr[0].src,expr[-1].tgt)
    

def rand_fin_f(srcS,tgtS):
    f=[choice(range(tgtS)) for i in range(srcS)]
    #print(f)
    return f

C=Category()

E,N=Object("A",10),Object("B",10)

f=Morphism(rand_fin_f(E.size,N.size),E,N)
g=Morphism(rand_fin_f(E.size,N.size),E,N)

C.add_morphism(f)
C.add_morphism(g)

f(1)
f.mutate()
f(1)
g(1)
g.mutate()
g(1)
f.compose(g)(1)
g.compose(f)(1)
    
    