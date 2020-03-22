# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:08:51 2020

@author: Simas
"""
from random import choice

            
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
    C.add_morphism(Morphism("0->2",rand_fin_f(5,5),obs[0],obs[2]))
    C.add_morphism(DataMorphism("test",[1,2,3],obs[0]))
    
    t=C.diagram_commute_scores()
    print("SCORES:\n")
    
    for k,v in t.items():
        print(str(k),":",v)
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




    