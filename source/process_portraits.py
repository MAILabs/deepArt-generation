
import sys
sys.path.append('e:\\art\\code\\deepArt-generation\\source')


import config, utils
from scipy.misc import imsave,imresize
import numpy as np
from skimage import io
from skimage.transform import resize
import os

import matplotlib.pyplot as plt

out_path = 'e:\\art\\wikiportraits'
path = config.datafile('portrait-sel')
# using directory with hand-selected images

print("Reading images...")

all_images_fn = [x for x in os.listdir(path) if x.endswith(".jpg") | x.endswith(".png") | x.endswith(".jpeg")]

all_images = []
ratios = []

for fn in all_images_fn:
    n = int(fn[:fn.find('-')])
    if n>8000: continue
    try:
        im = io.imread(os.path.join(path, fn))
        if im.shape[0] >= 256 and im.shape[1] >=256:
            all_images.append(im)
            ratios.append(im.shape[0]/im.shape[1])
    except:
        print("Error reading {}".format(fn))

plt.hist(ratios,bins=30)

sel_images = [x for x in all_images if 1.23 <= x.shape[0]/x.shape[1] <= 1.35]
print(len(sel_images))

for i,im in enumerate(sel_images):
    # im = resize(image=im, output_shape=(128,128), mode="reflect")
    im = imresize(im, (128,128))
    imsave(os.path.join(out_path, str(i) + '.png'), im)

sel_images = list(map(lambda x: imresize(x,(32,32)),sel_images))
sel_images = np.array(sel_images)

np.savez_compressed(os.path.join(out_path,'wikiportraits.npz'),imgs=sel_images)
