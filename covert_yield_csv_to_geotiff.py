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




if __name__ == '__main__':
    csvfile = r'D:\Steve_SA_winter_2017 Yield Data\sample2.csv'
    out_tiff = r'D:\Steve_SA_winter_2017 Yield Data\test.tiff'
    gsd = 10
    
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
    
    n_cols = math.ceil(width/20)+1
    n_rows = math.ceil(height/20)+1
       
    #get min, max boundaries
    #calc affine from gsd and top left corner
    a = affine.Affine(gsd,0,left,0,-gsd,bottom)
    df['row'], df['col'] = ~a * (df['easting'], df['northing'])
    ci = np.arange(math.ceil(df['row'].max()))
    ri = np.arange(math.ceil(df['col'].max()))
    c,r,z = df['row'], df['col'], df['yield']
    zi = griddata((c, r), z, (ci[None,:], ri[:,None]), method='linear')


    plt.figure()    
    plt.scatter(df['easting'],df['northing'])

    plt.figure()
    plt.scatter(df['row'],df['col'])
    plt.imshow(zi)

    epsg_code = df['epsg_code'][0]
    zi_height, zi_width = zi.shape
    def norm(a):
        a = np.where(np.isnan(a),0,a)
        return np.array(((a-a.min())/a.max())*255, dtype=np.uint8)
    zi = norm(zi)    
    
    profile = dict()
    
    profile.update({
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': 0,
            'width': zi_width,
            'height': zi_height,
            'count': 1,
            'crs':rio.crs.CRS.from_epsg(epsg_code),
            'affine':a,
            'transform':a.to_gdal(),
            #add min and max for reverse transform
            })
    
    with rio.open(out_tiff, "w", **profile) as dest:
        dest.write(zi, 1)

    tif_path = r'D:\Steve_SA_winter_2017 Yield Data'
    tif_path += r'\20171121T004659_T54HTJ_S2B_ndre_gray.tif'
    img = rio.open(tif_path)

"""
{'driver': 'GTiff',
 'dtype': 'uint8',
 'nodata': None,
 'width': 800,
 'height': 514,
 'count': 1,
 'crs': CRS({'init': 'epsg:32754'}),
 'transform': (277258.338817,
  1.5750000000000868,
  0.0,
  6344313.490408,
  0.0,
  -1.5758754863812432),
 'affine': Affine(1.5750000000000868, 0.0, 277258.338817,
        0.0, -1.5758754863812432, 6344313.490408)}
 """