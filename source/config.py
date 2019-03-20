data_dir = 'e:\\Art\\data'
models_dir = 'e:\\Art\\models'
output_dir = 'e:\\Art\\output'
category = 'still-life'
num_images = 1000

model = 'VAE_2' # one of 'DCGAN_1', 'DCGAN_2', 'DCGAN_3', 'VAE_1', 'VAE_2', 'VAE_3' or 'VAE_4'

save_plt = False
save_img = True
save_grid = (2,2) # or None

import os

def datafile(fn):
   return os.path.join(data_dir,fn)
