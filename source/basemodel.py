"""
@title basemodel.py
@author: Tuan Le, Dmitri Soshnikov
@email: tuanle@hotmail.de, dmitri@soshnikov.com

Generic script for neural models. Contains some common functions
"""

import numpy as np

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import config
import utils

## Build class for DCGAN

class BaseModel():
    def __init__(self, name):
        self.name = name.upper()
        self.model_path = os.path.join(config.models_dir, name, 'etc')
        self.images_path = os.path.join(config.models_dir, name, 'images')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)
        # Input image shape
        self.epoch = 0
        self.rows = 128
        self.cols = 128
        self.channels = 3
        self.img_shape = (self.rows, self.cols, self.channels)
        self.domain = (0,1)

    def save_samples(self,n):
        final_gen_images = self.generate_random_images(n)
        final_gen_images_int = (final_gen_images * 256.0).astype(np.uint8)
        for i in range(n):
            if config.save_plt:
                plt.imshow(final_gen_images[i, :, :, :], interpolation="nearest")
                plt.savefig(os.path.join(self.images_path, "hi_plt_images_ep%d_%d.jpg" % (self.epoch, i)))
            if config.save_img:
                utils.save_image(final_gen_images_int[i],
                                 os.path.join(self.images_path, "hi_raw_images_ep%d_%d.jpg" % (self.epoch, i)))

    def save_imgs(self,grid=config.save_grid):
        if grid is None:
            return
        r, c = grid
        gen_imgs = self.generate_random_images(r * c)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis("off")
                cnt += 1
                fig.savefig(os.path.join(self.images_path, "image_%d.jpg" % self.epoch))
        plt.close(fig)


    ## Helper for scaling and unscaling:
    def scale(self, x, out_range = (-1, 1)):
        domain = np.min(x), np.max(x)
        # a)scale data such that its symmetric around 0
        y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        # b)rescale data such that it falls into desired output range
        y = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

        return y

    def unscale(self, y, out_range = (-1, 1),domain=None):
        if domain is None: domain = self.domain
        # undo b)
        z = (y - (out_range[1] + out_range[0]) / 2) / (out_range[1] - out_range[0])
        # undo a)
        z = z * (domain[1] - domain[0]) + (domain[1] + domain[0]) / 2

        return z
