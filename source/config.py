data_dir = 'e:\\Art\\data'
models_dir = 'e:\\Art\\models'
output_dir = 'e:\\Art\\output'
category = 'flower-painting-sel'
num_images = 10000

model = 'DCGAN_1XX' # one of 'DCGAN_1', 'DCGAN_1X', 'DCGAN_1XX', 'DCGAN_2', 'DCGAN_3', 'VAE_1', 'VAE_2', 'VAE_3' or 'VAE_4'

save_plt = False
save_img = True
save_grid = (2,2) # or None

# Ration of width/height. Setting those limits results in images with similar aspect ratio
min_img_ratio = 0.7
max_img_ratio = 1.4

# Prefix numbers are initial numbers in the filenames, eg. 001-flower.jpg
# Setting those limits helps separate certain period of time, or limit # of images
#min_prefix_no = 0
#max_prefix_no = 8300
min_prefix_no = None
max_prefix_no = None

import os

def datafile(fn):
   return os.path.join(data_dir,fn)
