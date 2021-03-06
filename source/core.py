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

possible_models = ['DCGAN_1', 'DCGAN_1X', 'DCGAN_1XX', 'DCGAN_2', 'DCGAN_3', 'VAE_1', 'VAE_2', 'VAE_3','VAE_4']

def create_model(model=config.model,category=config.category):
    if model == 'DCGAN_1':
        my_model = DCGAN(name='DCGAN_1',category=category)
    elif model == 'DCGAN_1X':
        my_model = DCGAN(name='DCGAN_1X',category=category)
    elif model == 'DCGAN_1XX':
        my_model = DCGAN(name='DCGAN_1XX',category=category)
    elif model == 'DCGAN_2':
        my_model = DCGAN(name='DCGAN_2',category=category)
    elif model == 'DCGAN_3':
        my_model = DCGAN(name='DCGAN_3',category=category)
    elif model == 'VAE_1':
        my_model = VAE(name='VAE_1',category=category)
    elif model.upper() == 'VAE_2':
        my_model = VAE(name='VAE_2',category=category)
    elif model == 'VAE_3':
        my_model = VAE(name='VAE_3',category=category)
    elif model == 'VAE_4':
        my_model = VAE(name='VAE_4',category=category)
    else:
        my_model = VAE(name='VAE_2',category=category)
        print('The selected model {} is not in the list [DCGAN_1, DCGAN_1X, DCGAN_1XX, DCGAN_2, DCGAN_3, VAE_1, VAE_2, VAE_3, VAE_4]'.format(
            model))
    return my_model


def load_data(sz):
    """
    Load the dataset, either from source files, or from pre-prepared compressed numpy array
    If the pre-prepared file does not exist - create it
    """
    img_filez = config.datafile("train_data_{}_{}_{}.npz".format(config.category,sz,config.num_images))
    if not os.path.isfile(img_filez):
        print("Creating {} file for faster processing...".format(img_filez))
        print("Genre: {}, # of images = {}".format(config.category,config.num_images))
        final_images_stacked = preprocess(genre_or_style=config.category, min_vals=[sz,sz],n=config.num_images)
        np.savez_compressed(file=img_filez, a=final_images_stacked)
    else:
        print("Load preprocessed image data from {}".format(img_filez))
        final_images_stacked = np.load(file=img_filez)["a"]
    return final_images_stacked