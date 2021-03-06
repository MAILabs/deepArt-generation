# -*- coding: utf-8 -*-
"""
@title utils.py
@author: Dmitry Soshnikov
@email: dmitri@soshnikov.com
"""

import glob
from PIL import Image

def find_max_file(pattern,ext):
    max=None
    for x in glob.glob(pattern+"*"+ext):
        n = x[len(pattern):len(x)-len(ext)]
        if max is None or max<int(n):
            max = int(n)
    if max is None: return None,None
    else: return max,(pattern+str(max)+ext)

def pattern_files(pattern,ext):
    for x in glob.glob(pattern+"*"+ext):
        n = x[len(pattern):len(x)-len(ext)]
        yield n

def save_image(im,fn):
    img = Image.fromarray(im)
    img.save(fn)