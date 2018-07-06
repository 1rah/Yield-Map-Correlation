# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:48:00 2018

@author: FluroSat
"""

import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.tools.mask import mask

import numpy as np
from scipy.interpolate import griddata
import affine
import argparse
import math
import re
import shapely.geometry as sg
from shapely.geometry import mapping
import json



if __name__ == '__main__':
    
    tif_a = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T000000_HIRAMS_PLN_ndre_gray_not_corrected.float.tif'
    tif_b = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T001059_T55JGH_S2B_ndre_gray.float.tif'
    show_plots=True
    
    img_b = rio.open(tif_b)
    img_a = rio.open(tif_a)
    
    box_a = sg.box(*img_a.bounds)
    box_b = sg.box(*img_b.bounds)
    
    if re.match('(326|327)\d\d',img_a.crs.to_dict()['init'].split(':')[-1]) is None: #check input is in UTM
        raise TypeError('input images must be in UTM (epsg 326xx, 327xx) input CRS: {}'.format(img_a.crs))
    
    if img_a.crs != img_b.crs:
        raise ValueError('input images have non-matching CRS: {} and {}'.format(img_a.crs, img_b.crs))
    
    #check for overlaps
    ol = box_a.intersection(box_b)
    
    tif = tif_a
    with rio.open(tif) as img:
        masked, new_affine = mask(img, [mapping(ol)], crop=True)
    
    #get smallest gsd image
    
    #create array 1 (base), and array 2(interped)
    
    #top top left (getreference for each image), go down the row till one of the images is out of range
    #repeat across columns until one of the images is out of range
    
    if show_plots:
        plt.figure()
        plt.subplot(121)
        plt.imshow(img_a.read(1))
        plt.subplot(122)
        plt.imshow(img_b.read(1))