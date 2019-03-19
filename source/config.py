data_dir = 'e:\\Art\\data'
models_dir = 'e:\\Art\\models'
output_dir = 'e:\\Art\\output'
category = 'still-life'
num_images = 500

import os

def datafile(fn):
   return os.path.join(data_dir,fn)
