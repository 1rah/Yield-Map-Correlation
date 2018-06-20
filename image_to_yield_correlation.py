# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:59:29 2018

@author: FluroSat
"""


import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio

import numpy as np
import utm
import re
from scipy.interpolate import griddata
import affine

from skimage.transform import resize


def get_closest_dist(df):
    
    def get_closest_dist_col(df,i):
        c = df.iloc[:,i]
        c = abs((c[0] - c).unique())
        c = c[c>0]
        return c.min()
    
    d_min = list()
    for i in [0,1]:
        d_min.append(get_closest_dist_col(df,i))
    
    return tuple(d_min)




#define inputs
tif_plane_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T000000_HIRAMS_PLN_ndre_gray_not_corrected.float.tif'
tif_satt_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T001059_T55JGH_S2B_ndre_gray.float.tif'
csv_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\Terraidi_field3_Exported CSV Points Cotton Yield.CSV'


tif_path = tif_plane_path

gsd_usr = 20

#open image file
#! NOTE: assume in UTM format, single channel (index)
with rio.open(tif_path) as img:
    epsg_code = img.meta['crs']['init'].split(":")[-1]
    assert re.match(u'(327\d\d|326\d\d)', epsg_code)
    img_affine = img.affine
    img_array = img.read(1)


img_gsd = abs(np.array((img_affine.a, img_affine.e)))

new_shape = (
        img_array.shape / 
        (gsd_usr / img_gsd)
        ).astype(np.int)


new_img_array = resize(img_array, new_shape)


x_rs, y_rs = np.array(img_array.shape) / new_img_array.shape

new_affine = list(img_affine[0:6])
new_affine[0] = new_affine[0] * x_rs
new_affine[4] = new_affine[4] * y_rs
new_affine = affine.Affine(*new_affine)

#open csv
#! NOTE: Assume yield csv has headings: Easting, Northing, Yield and is same CRS as the image (UTM)
names="easting northing yield".split()
df = pd.read_csv(csv_path, skiprows=1, names=names)

df['col'], df['row'] = ~new_affine * (df['easting'], df['northing'])
df['rsv'] = ((df['yield'] - df['yield'].min())/df['yield'].max())*255


csv_sample_dist = get_closest_dist(df)







#plt.subplot(211)
plt.imshow(new_img_array)
plt.scatter(df.col, df.row)


if True:
    
    ##interpolate to the grid
    rs,cs = new_img_array.shape
    ci = np.arange(cs)
    ri = np.arange(rs)
    c,r,z = df.col, df.row, df['rsv']
    zi = griddata((c, r), z, (ci[None,:], ri[:,None]), method='linear')
    
    plt.figure()
    plt.contourf(ci,ri,zi,15,cmap=plt.cm.jet)
    
    
    f = ~np.isnan(zi)
    
    plt.figure()
    plt.scatter(new_img_array[f].flatten(), zi[f].flatten())      
