# -*- coding: utf-8 -*-
"""
@title core.py
@author: Tuan Le, Dmitri Soshnikov
@email: tuanle@hotmail.de, dmitri@soshnikov.com
"""

import config
import os
from data_preprocess import preprocess
import numpy as np
from vae import VAE
from dcgan import DCGAN

def create_model(model=config.model):
    if model == 'DCGAN_1':
        my_model = DCGAN(name='DCGAN_1')
    elif model == 'DCGAN_2':
        my_model = DCGAN(name='DCGAN_2')
    elif model == 'DCGAN_3':
        my_model = DCGAN(name='DCGAN_3')
    elif model == 'VAE_1':
        my_model = VAE(name='VAE_1')
    elif model.upper() == 'VAE_2':
        my_model = VAE(name='VAE_2')
    elif model == 'VAE_3':
        my_model = VAE(name='VAE_3')
    elif model == 'VAE_4':
        my_model = VAE(name='VAE_4')
    else:
        my_model = VAE(name='VAE_2')
        print('The selected model {} is not in the list [DCGAN_1, DCGAN_2, DCGAN_3, VAE_1, VAE_2, VAE_3, VAE_4]'.format(
            model))
    return my_model


def load_data():
    """
    Load the dataset, either from source files, or from pre-prepared compressed numpy array
    If the pre-prepared file does not exist - create it
    """
    img_filez = config.datafile("train_data_{}_{}.npz".format(config.category,config.num_images))
    if not os.path.isfile(img_filez):
        print("Creating {} file for faster processing...".format(img_filez))
        print("Genre: {}, # of images = {}".format(config.category,config.num_images))
        final_images_stacked = preprocess(genre_or_style=config.category, min_vals=[128,128],n=config.num_images)
        np.savez_compressed(file=img_filez, a=final_images_stacked)
    else:
        print("Load preprocessed image data from {}".format(img_filez))
        final_images_stacked = np.load(file=img_filez)["a"]
    return final_images_stacked