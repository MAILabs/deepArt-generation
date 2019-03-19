# -*- coding: utf-8 -*-
"""
@title train_model.py
@author: Tuan Le
@email: tuanle@hotmail.de
"""

from dcgan import DCGAN
from vae import VAE

import config

from data_preprocess import preprocess
import numpy as np
import matplotlib.pyplot as plt
import sys
import gc
import os


def main(model, epochs, batch_size, save_intervals):

    model = model.upper()

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
        print('The selected model {} is not in the list [DCGAN_1, DCGAN_2, DCGAN_3, VAE_1, VAE_2, VAE_3, VAE_4]'.format(model))

    print("Python main program for generating images using {}".format(model))

    ## preprocess data images if init_train and save the images as pickle file
    img_filez = config.datafile("train_data_{}_{}.npz".format(config.category,config.num_images))
    if not os.path.isfile(img_filez):
        print("Creating {} file for faster processing...".format(img_filez))
        print("Genre: {}, # of images = {}".format(config.category,config.num_images))
        final_images_stacked = preprocess(genre_or_style=config.category, min_vals=[128,128],n=config.num_images)
        np.savez_compressed(file=img_filez, a=final_images_stacked)
    else:
        print("Load preprocessed image data from {}".format(img_filez))
        final_images_stacked = np.load(file=img_filez)["a"]


    my_model.train(data = final_images_stacked, epochs = epochs, batch_size = batch_size, save_intervals = save_intervals, sample_intervals=save_intervals, hi_sample_intervals=save_intervals)

if __name__ == "__main__":
    """
    This script runs the main training programm for the model. Note Model either has to be
    'DCGAN_1', 'DCGAN_2', 'DCGAN_3', 'VAE_1', 'VAE_2', 'VAE_3' or 'VAE_4'
    """
    ### If user inserts via shell console
    if len(sys.argv) == 2:
            model = sys.argv[1]
    else:
            model = "VAE_2"

    epochs = 1000
    batch_size= 16
    save_intervals = 100

    ### If no arguments were inserted when calling this script

    main(model=model, epochs=epochs, batch_size=batch_size, save_intervals=save_intervals)
