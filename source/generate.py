# -*- coding: utf-8 -*-
"""
@title: generate.py
@author: Dmitri Soshnikov
@email: dmitri@soshnikov.com
"""

import argparse
import os
import sys
import config,core,utils
import numpy as np

parser = argparse.ArgumentParser(description="Art Image Generator")
parser.add_argument('--model',dest='model',default=config.model,help='model to use')
parser.add_argument('--filename',dest='filename',default='img_{N}.jpg',help='image filename to use, use {N} for number, {EP} for epoch, {D} for descriptor')
parser.add_argument('--dest',dest='path',default='.',help='destination directory, defaults to .')
parser.add_argument('--num',dest='n',default=1,type=int,help='number of images to generate')
parser.add_argument('--use-img',dest='use_img',default=False,const=True,action='store_const',help='use data as strating point (valid for VAE only)')
parser.add_argument('--add-noise',dest='noise',default=False,const=True,action='store_const')

args = parser.parse_args()

m = core.create_model(args.model)

if "VAE" in args.model:
    if args.use_img:
        data = core.load_data()[0:args.n]
        vec = m.get_vector_representation(data)
        vec = vec[2]
    else:
        vec = np.random.normal(0,1,(args.n,m.latent_dim))
    if args.noise:
        vec = vec + np.random.normal(0,1,(args.n,m.latent_dim))
    imgs = m.generate_images(vec)
    args.filename = args.filename.replace('{EP}',str(m.epoch))
    args.filename = args.filename.replace('{D}', "{}{}".format(int(args.use_img),int(args.noise)))
    mi, ma = np.min(imgs),np.max(imgs)
    imgs = ((imgs-mi)/(ma-mi)*255).astype(np.uint8)
    for i,im in enumerate(imgs):
        fn = args.filename.replace('{N}',str(i))
        utils.save_image(im,os.path.join(args.path,fn))

else:
    print("Model type {} is not supported yet".format(args.model))