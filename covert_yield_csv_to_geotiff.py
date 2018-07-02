# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:47:47 2018

@author: Irah Wajchman


@description:
    Convert a yield csv with headers "easting northing yield" (and UTM coordinates)
    to a single channel Geotiff
    
"""

import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
#import utm
import re
from scipy.interpolate import griddata
import affine
from skimage.transform import resize
#from sklearn.metrics import r2_score
import argparse
import math 

def interpolate_df_to_img_grid(df, new_img_array):
    rs,cs = new_img_array.shape
    ci = np.arange(cs)
    ri = np.arange(rs)
    c,r,z = df.col, df.row, df['yield']
    zi = griddata((c, r), z, (ci[None,:], ri[:,None]), method='linear')
    return zi


if __name__ == '__main__':
    csvfile = r'D:\Steve_SA_winter_2017 Yield Data\sample.csv'
    gsd = 20
    
    #open csv
    df = pd.read_csv(csvfile)
    
    #normalise to 255 get UTM (return mapping in metadata)
    max_yield = df['yield'].max()
    min_yield = df['yield'].min()
    df['px'] = ((df['yield']-min_yield) / max_yield)*255

    left, right = df['easting'].min(), df['easting'].max()
    top, bottom = df['northing'].min(), df['northing'].max()
    
    #generate grid
    width = right-left
    height = bottom-top
    
    n_cols = math.ceil(width/20)
    n_rows = math.ceil(height/20)
    
    #left bounds= left + n_cols*20
    
    
    
    #get min, max boundaries
    #calc affine from gsd and top left corner
    
    
    tif = r'D:\Yield-Map-Correlation\Terraidi_NDRE_rasters\20180212T001111_T55JGH_S2A_ndre_gray.float.tif'
    img = rio.open(tif)
