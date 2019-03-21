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
import random

parser = argparse.ArgumentParser(description="Art Image Generator")
parser.add_argument('--model',dest='model',default=config.model,help='model to use',choices=core.possible_models)
parser.add_argument('--filename',dest='filename',default='img_{N}.jpg',help='image filename to use, use {N} for number, {EP} for epoch, {D} for descriptor')
parser.add_argument('--dest',dest='path',default='.',help='destination directory, defaults to .')
parser.add_argument('--num',dest='n',default=1,type=int,help='number of images to generate')
parser.add_argument('--use-img',dest='use_img',default=False,const=True,action='store_const',help='use data as strating point (valid for VAE only)')
parser.add_argument('--shuffle',dest='shuffle',default=False,const=True,action='store_const',help='shuffle input images')
parser.add_argument('--add-noise',dest='noise',default=False,const=True,action='store_const',help='add noise to input vector')
parser.add_argument('--epocycle',dest='epocycle',default=False,const=True,action='store_const',help='do epoch cycle generation')
parser.add_argument('--epoch',dest='epoch',default=None,type=int,help='specify exact epoch to use')
parser.add_argument('--treshold',dest='treshold',default=0.95,type=float,help='treshold to select best images in DCGAN')


args = parser.parse_args()

m = core.create_model(args.model)

def main(m,vec):
    imgs = m.generate_images(vec)
    filename = args.filename.replace('{EP}', str(m.epoch))
    filename = filename.replace('{D}', "{}{}".format(int(args.use_img), int(args.noise)))
    mi, ma = np.min(imgs), np.max(imgs)
    imgs = ((imgs - mi) / (ma - mi) * 255).astype(np.uint8)
    for i, im in enumerate(imgs):
        fn = filename.replace('{N}', str(i))
        utils.save_image(im, os.path.join(args.path, fn))

if "VAE" in args.model:
    if args.use_img:
        data = core.load_data()
        if args.shuffle: random.shuffle(data)
        data = data[0:args.n]
        vec = m.get_vector_representation(data)
        vec = vec[2]
    else:
        vec = np.random.normal(0,1,(args.n,m.latent_dim))
    if args.noise:
        vec = vec + np.random.normal(0,1,(args.n,m.latent_dim))

    if args.epocycle:
        eps = list(m.available_epochs())
        for ep in eps:
            m.load_epoch(ep)
            main(m,vec)
    else:
        if args.epoch is not None:
            m.load_epoch(args.epoch)
        main(m,vec)
elif "DCGAN" in args.model:
        batch_size=16
        vec = np.zeros((args.n,m.latent_dim))
        j=0
        while j<args.n:
            v = np.random.normal(0,1,(batch_size,m.latent_dim))
            ims = m.generator.predict(v)
            res = m.discriminator.predict(ims)
            print("Computing batch, max prob={}".format(np.max(res)))
            for i,(im,z) in enumerate(zip(ims,res)):
                if z>=args.treshold:
                    vec[j]=v[i]
                    j+=1
                    if j>=args.n:
                        break
        if args.epocycle:
            eps = list(m.available_epochs())
            for ep in eps:
                m.load_epoch(ep)
                main(m, vec)
        else:
            if args.epoch is not None:
                m.load_epoch(args.epoch)
            main(m, vec)
else:
    print("Model type {} is not supported yet".format(args.model))