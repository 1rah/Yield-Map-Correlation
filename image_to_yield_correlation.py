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


def filter_low(a):
    pval = 3
    a = np.where(a==0, np.nan, a)
    a_nonan = a[~np.isnan(a)]
    
    lwr = np.percentile(a_nonan, pval)
#    lwr_mean = np.mean(a_nonan[a_nonan < lwr])
    a = np.where(a<lwr, np.nan, a)
    
    upr = np.percentile(a_nonan, 100-pval)
#    upr_mean = np.mean(a_nonan[a_nonan > upr])
    a = np.where(a>upr, np.nan, a)
    
    
    
#    a = a - a.min()
#    a = a / a.max()
    
    return a



def rebin(a, bins):
    f = ~np.isnan(a)
    _cnt, b_vals = np.histogram(a[f].flatten(), bins=bins)
    
    dig = np.digitize(a, b_vals)
    for i, b in enumerate(b_vals):
        dig = np.where( dig==b, b, dig)
    
    dig = np.where( np.isnan(a), np.nan, dig)
    return dig
  
        
        
    


#define inputs
tif_plane_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T000000_HIRAMS_PLN_ndre_gray_not_corrected.float.tif'
tif_satt_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T001059_T55JGH_S2B_ndre_gray.float.tif'
csv_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\Terraidi_field3_Exported CSV Points Cotton Yield.CSV'

#tif_path =  tif_plane_path # tif_satt_path
tif_path = r'D:\Yield-Map-Correlation\Terraidi_NDRE_rasters\20180212T001111_T55JGH_S2A_ndre_gray.float.tif'
csv_path = r'D:\Yield-Map-Correlation\Terraidi_field_CSV Points Cotton Yield\Terraidi_field1_Exported CSV Points Cotton Yield.CSV'

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
        )
new_shape = new_shape.astype(np.int)


new_img_array = resize(
        filter_low(img_array),
#        img_array,
        new_shape,
        )


x_rs, y_rs = np.array(img_array.shape) / new_img_array.shape

new_affine = list(img_affine[0:6])
new_affine[0] = new_affine[0] * x_rs
new_affine[4] = new_affine[4] * y_rs
new_affine = affine.Affine(*new_affine)

#open csv
#! NOTE: Assume yield csv has headings: Easting, Northing, Yield and is same CRS as the image (UTM)
names="easting northing yield".split()
df = pd.read_csv(csv_path)
df.columns = names

df['col'], df['row'] = ~new_affine * (df['easting'], df['northing'])
df['yield_norm'] = ((df['yield'] - df['yield'].min())/df['yield'].max())


csv_sample_dist = get_closest_dist(df)










if True:
    plt.figure()
    #plt.subplot(211)
    plt.imshow(new_img_array)
    plt.scatter(df.col, df.row)
    
    ##interpolate to the grid
    rs,cs = new_img_array.shape
    ci = np.arange(cs)
    ri = np.arange(rs)
    c,r,z = df.col, df.row, df['yield_norm']
    zi = griddata((c, r), z, (ci[None,:], ri[:,None]), method='linear')
    
    zi = filter_low(zi)
    
    bins=25
    zi = rebin(zi,bins)
    new_img_array = rebin(new_img_array,bins)
    
    plt.figure()
#    plt.contourf(ci,ri,zi,15,cmap=plt.cm.jet)
    plt.imshow(zi)
    
    
    f = (~np.isnan(zi)) & (~np.isnan(new_img_array))
    
    plt.figure()
    plt.scatter( new_img_array[f].flatten(), zi[f].flatten() )
    
#    pfit = np.polyfit(new_img_array[f].flatten(), zi[f].flatten(), 5)
#    pfit = np.poly1d(pfit)
    
    x = new_img_array[f].flatten()[:,np.newaxis]
    y2, _, _, _ = np.linalg.lstsq(x, np.square(zi[f].flatten()) )
    
    x = np.linspace(0, new_img_array[f].max(), 20)
    plt.plot(x, np.sqrt(y2*x))
    
    
    plt.figure()
    plt.hist(zi[f].flatten(), bins=10)
    plt.hist(np.log(new_img_array[f].flatten()), bins=10)

    print(np.corrcoef(zi[f].flatten(), new_img_array[f].flatten()) )
    print(np.corrcoef(zi[f].flatten(), np.square(new_img_array[f].flatten())) )
    
    
    plt.figure()
    interp_yield = np.where(np.isnan(img_array), np.nan, np.sqrt(y2*img_array))
    plt.imshow(interp_yield)
