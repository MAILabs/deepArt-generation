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
import argparse

def main(model, epochs, batch_size, save_intervals,category):

    my_model = core.create_model(model,category)

    print("Python main program for generating images using {} with category {}".format(model,category))

    ## preprocess data images if init_train and save the images as pickle file
    final_images_stacked = core.load_data(my_model.rows)

    my_model.train(data = final_images_stacked, epochs = epochs, batch_size = batch_size, save_intervals = save_intervals, sample_intervals=save_intervals, hi_sample_intervals=save_intervals)

if __name__ == "__main__":
    """
    This script runs the main training programm for the model. Note Model either has to be
    """

    parser = argparse.ArgumentParser(description='Art Network Trainer')
    parser.add_argument('--model', dest='model', default=config.model, help='model to use',
                        choices=core.possible_models)
    parser.add_argument('--category', dest='category', default=config.category, help='category of painting to use (from the data dir subdirs)')
    parser.add_argument('--epochs', dest='epochs', default=10000, help='number of epochs', type=int)
    parser.add_argument('--batch-size', dest='batch_size', default=16, help='minibatch size', type=int)
    parser.add_argument('--save-intervals', dest='save_intervals', default=500, help='save intervals', type=int)

    args = parser.parse_args()

    main(model=args.model, epochs=args.epochs, batch_size=args.batch_size, save_intervals=args.save_intervals, category=args.category)
