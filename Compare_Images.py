# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:24:56 2018

@author: FluroSat
"""

import rasterio
import numpy as np
#import pandas as pd
import PIL
import matplotlib.pyplot as plt

import random

from osgeo import gdal
import affine
import pandas as pd
from math import sqrt
from scipy.interpolate import griddata

gdal.UseExceptions()




tif_plane_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T000000_HIRAMS_PLN_ndre_gray_scaled.tif'
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


ri,ci = pln_array.shape
ci = np.arange(ci)
ri = np.arange(ri)
c,r,z = df.col, df.row, df['Cotton Yield']
zi = griddata((c, r), z, (ci[None,:], ri[:,None]), method='linear')
#plt.contourf(ci,ri,zi,15,cmap=plt.cm.jet)

plt.figure()

yield_array_2 = 2 * np.where(pln_array>0,zi,np.nan)
plt.imshow(yield_array_2)

plt.figure()
pln_array_2 = 2 * np.where(pln_array>0,pln_array,np.nan)
plt.imshow(pln_array_2)
