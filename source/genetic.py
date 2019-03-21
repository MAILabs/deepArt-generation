# -*- coding: utf-8 -*-
"""
@title: genetic.py
@author: Dmitri Soshnikov
@email: dmitri@soshnikov.com
"""

import core, config
import numpy as np
import matplotlib.pyplot as plt
import random

def show_images(x):
    fig,ax = plt.subplots(1,len(x))
    for i,im in enumerate(x):
        ax[i].imshow(im)
        ax[i].axis('off')
        ax[i].set_title(str(i))
    plt.show()

def xover(v1,v2):
    v = v1[:]
    n = random.randint(1,len(v2)-2)
    v[n:]=v2[n:]
    return v

def mutate(v):
    n = random.randint(0, len(v)-1)
    v[n] = random.normalvariate(0,1)
    return v

m = core.create_model("DCGAN_1")
vec = np.random.normal(0,1,(13,m.latent_dim))

while True:
    g = m.generate_images(vec)
    show_images(g)
    n = [int(x) for x in input().split()]
    if len(n)!=3: break
    # n = [x for x in range(10) if x not in n1]
    v = [vec[i,:] for i in n]
    for i in range(3): vec[i] = v[i]
    j = 3
    for i,k in [(0,1),(1,2),(0,2)]:
        vec[j] = xover(v[i],v[k])
        vec[j+1] = xover(v[i], v[k])
        vec[j+2] = mutate(xover(v[i], v[k]))
        j+=3
    vec[j] = np.random.normal(0,1,m.latent_dim)


