# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:24:56 2018

@author: FluroSat
"""

import rasterio
import numpy as np
#import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import random

from osgeo import gdal
import affine
import pandas as pd
from math import sqrt
from scipy.interpolate import griddata
from skimage.transform import resize

from scipy import ndimage

gdal.UseExceptions()




tif_plane_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T000000_HIRAMS_PLN_ndre_gray_not_corrected.float.tif'
tif_satt_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T001059_T55JGH_S2B_ndre_gray.float.tif'
csv_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\Terraidi_field3_Exported CSV Points Cotton Yield.CSV'

with rasterio.open(tif_plane_path) as tif:
#    a = affine.Affine.from_gdal(*tif.transform)
    a = tif.affine
    pln_array = tif.read(1)
    
    
with rasterio.open(tif_satt_path) as tif:
#    a = affine.Affine.from_gdal(*tif.transform)
#    a = tif.affine
    sat_array = tif.read(1)
    
    
df = pd.read_csv(csv_path)
df['col'], df['row'] = ~a * (df['UTM Easting'], df['UTM Northing'])


rs,cs = pln_array.shape

#interpolate to the grid
ci = np.arange(cs)
ri = np.arange(rs)
c,r,z = df.col, df.row, df['Cotton Yield']
zi = griddata((c, r), z, (ci[None,:], ri[:,None]), method='linear')
#plt.contourf(ci,ri,zi,15,cmap=plt.cm.jet)


def compare(sat_array, pln_array, zi, n):
    
    
    div = [1, 7,14,41,78,390][n]
    new_shape = (rs/div, cs/div)
    sat_array = resize(sat_array, new_shape)
    pln_array = resize(pln_array, new_shape)
    zi = resize(zi, new_shape)
       
    plt.subplot(nrows, ncols, (n*3)+1)
    yield_array_2 = np.where(pln_array>0,zi,np.nan)
    plt.imshow(yield_array_2)
    
    plt.subplot(nrows, ncols, (n*3)+2)
    sat_array_2 = np.where(pln_array>0,sat_array,np.nan)
    plt.imshow(sat_array_2)
    
    plt.subplot(nrows, ncols, (n*3)+3)
    pln_array_2 = np.where(pln_array>0,pln_array,np.nan)
    plt.imshow(pln_array_2)
    
    
    r,c = np.where(pln_array>0)
    
    pln_flat = pln_array.flatten()
    pln_flat_ndre = pln_flat[pln_flat>0]
    
    sat_flat = sat_array.flatten()
    sat_flat = sat_flat[pln_flat>0]
    
    yield_flat = zi.flatten()
    yield_flat = yield_flat[pln_flat>0]
    
    
    corr_df = pd.DataFrame({
            'ndre_plane':pln_flat_ndre,
            'ndre_sat':sat_flat,
            'yield':yield_flat
                    })
    
    c = corr_df.corr()
    
    c_list.append(dict(c['yield']))


c_list =[]
ncols = 3
n = int(min(rs**(1/3.0), cs**(1/3.0)))
nrows = n
plt.figure()
for nx in [0,1,2,3,4,5]:#range(n):
    compare(sat_array, pln_array, zi, nx)
    
    
plt.show()
c_list= pd.DataFrame(c_list)
##plt.scatter(r,c,c=flt,cmap=plt.cm.jet)

#
#b = ndimage.interpolation.zoom(pln_array,1/100)  # (20, 40)
#plt.imshow(b)
