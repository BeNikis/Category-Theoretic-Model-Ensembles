# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:54:58 2020

@author: Simas
"""

from random import *

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
    
        
        
    
    