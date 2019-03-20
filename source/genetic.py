# -*- coding: utf-8 -*-
"""
@title: genetic.py
@author: Dmitri Soshnikov
@email: dmitri@soshnikov.com
"""

import core, config
import numpy as np

m = core.create_model()
vec = np.random.normal(0,1,(10,m.latent_dim))
