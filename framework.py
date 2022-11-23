# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:54:58 2020

@author: Simas
"""

from random import *
import copy

from torch import Tensor
import networkx as nx
import matplotlib.pyplot as plt

class Object:

    def __init__(self,name,contents=None,metric=lambda x,y:x==y):
        self.name = name
        self.type=None if not contents else type(contents)
        self.metric=metric

        
        
    def named_morphisms(self):
        return self.special
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()
    
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

#class for distinguishing all special objects (like products) in the class hierarchy
class SpecialObject(Object):
    def __init__(self,name,contents=None,metric=lambda x,y:x==y):
        super().__init__(name,contents,metric)
        self.obs = []
        self.special=[]

class ProductObject(SpecialObject):
    def __init__(self,*OBS):
        super().__init__("x".join(map(lambda o:o.name,OBS)))
        self.obs = OBS
        self.type = list(map(type,self.obs))
        self.metric = lambda x,y:list(map(lambda o:o.metric(x,y),self.obs))
        
        
        
        
        for i in range(len(self.obs)):
            self.special.append(Morphism("p"+str(i),self,self.obs[i],lambda x:self.obs[self.obs[i](x)]))
    


class Morphism:
    i=0
    def __init__(self,name,src,tgt,f=None):
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
        self.data=data
        super(DataMorphism,self).__init__(name,'*',tgt,lambda x=None:self.data)
        
    
    def __index__(self,index):
        return self(index)
    
    def __call__(self,index=None):
        if not index:
            return self.data
        return self.data[index]
    

#Morphisms which can take input ang give output from/into any object of the given types
class KernelMorphism(Morphism):
    def __init__(self,name,src_t,tgt_t,f=None):
        self.name = name
        self.f=f
        
        if type(src_t)==torch.Tensor:
            self.src=src_t.shape
        else:
            self.src=src_t
        
        
        if type(tgt_t)==torch.Tensor:
            self.tgt=tgt_t.shape
        else:
            self.tgt=tgt_t
        
        self.special = {}
        
    def __call__(self,x=None):
  
        if type(x)==self.src_t:
            y=self.f(x)
            if y==self.tgt_t:
                return y
            else:
                raise TypeError("Kernel morphism "+self.name+" takes type " + self.src+" but was given " + type(x))
        else:
            raise TypeError("Kernel morphism "+self.name+" takes type " + self.src+" but was given " + type(x))
    

class HiddenMorphism(Morphism):
    pass

class OptimisableMorphism(Morphism):
    pass
    
class Category:

    
    def __init__(self,ob_list:list=[],mor_list=[]):
        self.os = set() 
        self.ms = {}
        
        
        self.input = []
        self.kernel_ms = []
        self.optimizable_ms=[]
        for o in ob_list:
            self.add_object(o)
        for m in mor_list:
            self.add_morphism(m)
        
        return
    
    def add_morphism(self,f):
        if f.src not in self.os:
            self.add_object(f.src)
        if f.tgt not in self.os:
            self.add_object(f.tgt)
        
        if f.src not in self.ms.keys():
            self.ms[f.src]={}
            
        if f.tgt not in self.ms[f.src].keys():
            self.ms[f.src][f.tgt]=[f]
        else:
            self.ms[f.src][f.tgt].append(f)
            
        if type(f)==DataMorphism:
            self.input.append(f)
        elif type(f)==KernelMorphism:
            self.kernel_ms.append(f)
        elif type(f)==OptimisableMorphism:
            self.optimizable_ms.append(f)
            
        return
    
    def all_morphisms(self):
        ms = [] 
        for src in self.ms.keys():
            for tgt in self.ms[src].keys():
                ms+=self.ms[src][tgt]
        return ms+self.input+self.optimizable_ms
    
    def add_object(self,o):
        if o not in self.os:            
            self.os.add(o)
            
            if isinstance(o, SpecialObject):
                for so in o.obs:
                    self.add_object(so)                
            
                for m in o.special:
                    self.add_morphism(m)
            
            
            
        
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
    def mors_from(self,src,kernel_only=False):
        out = []
    
        
        if not kernel_only:
            for tgt in self.ms[src].keys():
                out+=self.ms[src][tgt]
        
        
        for kf in self.kernel_ms:
            if type(src)==kf.src:
                out.append(kf)
        return out
    
    #returns a dict of lists where the key is the target of the morphisms in the list,going from src
    #exclude is a list of morphisms/objects to not use
    def all_paths_from(self,src,exclude=[],visited=[]):
        
            
        paths={}
        step=self.mors_from(src)
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
        color_key={Morphism:'black',
                   KernelMorphism:'black',
                   HiddenMorphism:'gray',
                   DataMorphism:'red',
                   OptimisableMorphism:'purple'}
        
        map(lambda n:G.add_node(n),self.os)
        
        for src in self.ms.keys():
            for tgt in self.ms[src].keys():
                if (src is not None) and (tgt is not None) and not (src==tgt):
                    for mor in self.ms[src][tgt]:
                        G.add_edge(src,tgt,color=color_key[type(mor)])
        
        
        nx.draw(G, with_labels=True, font_weight='bold',edge_color=nx.get_edge_attributes(G,'color').values())
        plt.show()
        return          
    
    def __index__(self,src,tgt):
        k_mors = list(filter(lambda km:src.type==km.src and tgt.type==km.tgt,self.kernel_ms))
        return self.ms[src,tgt]+k_mors
#a fixed diagram defined by the shape of a category that takes morphisms and obejcts to replace the placeholders with
class Diagram(Category):
    def __init__(self,c:Category,ob_list:list= None,mor_list:list = None):
        
        super().__init__()
        for o in c.os:
            self.add_object(o)
        
        
        for m in c.all_morphisms():
            self.add_morphism(HiddenMorphism(m.name, m.src, m.tgt))
        
        if ob_list or mor_list:
            self._instantiate(ob_list,mor_list)
        
    def _instantiate(ob_list,mor_list):
        if ob_list is not None:
            for ob in ob_list:
                for holder_ob in self.os:
                    if holder_ob.name==ob.name:
                        self.os.remove(holder_ob)
                        self.os.add(ob)
        if mor_list is not None:
            for mor in mor_list:
                for holder_mor in self.ms:
                    if holder_mor.name==mor.name:
                        h_i=self.ms[holder_mor.src][holder_mor.tgt].index(holder_mor)
                        self.ms[holder_mor.src][holder_mor.tgt][h_i]=mor
    
    def instantiate(ob_list:list = None,mor_list:list = None):
        return Diagram(self,ob_list,mor_list)
    
    
    def commute_scores(self):
        pass #implement later
        


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

    
        
def gen_abstract_category(n_obs,n_mors,mors_between_all_objects=True,max_name_length=3) -> Category:
    from random import choice
    names = []
    def gen_name(l,upper=False):
       from string import ascii_lowercase as letters
       
       def gen():
           name = "".join([choice(letters) for i in range(choice(range(1,l+1)))])
           if upper:
               name = name.upper()
        
           return name 
       name = gen()
       while (name in names):
           name = gen()
       
       names.append(name)
       return name
       
     
    obs = [Object(gen_name(max_name_length,True)) for n in range(n_obs) ]       
    mors = [Morphism(gen_name(max_name_length), choice(obs), choice(obs)) for n in range(n_mors) ]       
    
    if mors_between_all_objects:
        for o1 in obs:
            for o2 in obs:
                mors.append(Morphism("".join([o1.name.lower(),"to",o2.name.lower()]), o1, o2))
                mors.append(Morphism("".join([o2.name.lower(),"to",o1.name.lower()]), o2, o1))
     
    C=Category()
    for m in mors:
        C.add_morphism(m)
    #print(C.os,"\n",C.ms)
    return C



    