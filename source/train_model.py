# -*- coding: utf-8 -*-
"""
@title train_model.py
@author: Tuan Le, Dmitri Soshnikov
@email: tuanle@hotmail.de, dmitri@soshnikov.com
"""

import config

import numpy as np
import matplotlib.pyplot as plt
import sys
import gc
import core
import os


def main(epochs, batch_size, save_intervals):

    model = config.model
    my_model = core.create_model(model)

    print("Python main program for generating images using {}".format(model))

    ## preprocess data images if init_train and save the images as pickle file
    final_images_stacked = core.load_data()

    my_model.train(data = final_images_stacked, epochs = epochs, batch_size = batch_size, save_intervals = save_intervals, sample_intervals=save_intervals, hi_sample_intervals=save_intervals)

if __name__ == "__main__":
    """
    This script runs the main training programm for the model. Note Model either has to be
    """
    ### If user inserts via shell console
    if len(sys.argv) == 2:
            model = sys.argv[1]
    else:
            model = "VAE_2"

    epochs = 100000
    batch_size= 16
    save_intervals = 100

    ### If no arguments were inserted when calling this script

    main(epochs=epochs, batch_size=batch_size, save_intervals=save_intervals)
