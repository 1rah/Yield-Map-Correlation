# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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


def raster_to_df(array, a, bkgrnd_val=0):
    out=list()
    rows, cols = array.shape
    for r in range(rows):
        for c in range(cols):
            x, y = a * (c, r)
            val = array[r,c]
            if val != bkgrnd_val:
                out.append({'row':r,
                            'col':c,
                            'x':x,
                            'y':y,
                            'value':val,
                            'ndre':(float(val)/255*2)-1})
    return out
    


tif_path = r'D:\Yield-Map-Correlation\Terraidi_NDRE_raster_gray\20180329T001109_T55JGH_S2B_ndre_gray.tif'
csv_path = r'D:\Yield-Map-Correlation\Terraidi_field_CSV Points Cotton Yield\Terraidi_field1_Exported CSV Points Cotton Yield.CSV'

with rasterio.open(tif_path) as tif:
#    a = affine.Affine.from_gdal(*tif.transform)
    a = tif.affine
    ndre_array = tif.read(1)
    

#ndre_array = np.where(ndre_array==0, np.nan, ndre_array)
#ndre_array = (ndre_array.astype(np.float) / 255 * 2)-1
#np.min(ndre_array)
plt.imshow(ndre_array)

## Finds transforms for coords
#GTiff = gdal.Open(tif_path)
#trans = GTiff.GetGeoTransform()
#a = affine.Affine.from_gdal(*trans)

## col, row to x, y
#x, y = a * (col, row)
#
## x, y to col, row
#col, row = ~a * (x, y)

df_ndre = raster_to_df(ndre_array, a)
df_ndre = pd.DataFrame(df_ndre)

df_yield = pd.read_csv(csv_path)

sample_range = len(df_yield)
no_samples = 5
id_list = np.random.choice(sample_range, no_samples, replace=False)

out=[]
for i in id_list:
    pt = df_yield.iloc[i]
    p1x, p1y = pt['UTM_Easting'], pt['UTM_Northing']
    
    def dist(p2x, p2y, p1x=p1x, p1y=p1y):
        dx = (p1x - p2x)**2
        dy = (p1y - p2y)**2
        return (dx+dy)**(0.5)
        
    
    df_ndre['dist'] = dist(df_ndre.x, df_ndre.y)
    
    
    idx_min = df_ndre.dist.idxmin()
    min_pt = df_ndre.iloc[idx_min]
    
    
    line = zip([(u'point_'+k).replace(' ','_') for k in pt.keys()], pt.values)
    line += zip([u'pixel_'+k for k in min_pt.keys()], min_pt.values)
    line = dict(line)
    out.append(line)

out = pd.DataFrame(out)
corr = out[[u'pixel_ndre', u'point_Cotton_Yield']].corr()
corr_coeff = corr.pixel_ndre.point_Cotton_Yield

#df_yield[['x','y']] = ~a*df_yield[['UTM_Northing','UTM_Easting']]


ci = np.arange(np.min(df_ndre.x), np.max(df_ndre.x),10)
ri = np.arange(np.min(df_ndre.y), np.max(df_ndre.y),10)

c,r,z = df_ndre.x, df_ndre.y, df_ndre.ndre

zi = griddata((r, c), z, (ri[None,:], ci[:,None]), method='linear')
plt.contourf(ri,ci,zi,15,cmap=plt.cm.jet)



c_data = df_yield.UTM_Easting
r_data = df_yield.UTM_Northing
plt.scatter(c_data, r_data)

ci = np.linspace(np.min(df_ndre.x), np.max(df_ndre.x),406)
ri = np.linspace(np.min(df_ndre.y), np.max(df_ndre.y),800)



#out = list()
#for i, r in df_yield.iterrows():
#    col, row = ~a * (r['UTM_Easting'], r['UTM_Northing'])
#    line = dict(r)
#    line.update({'row':row, 'col':col})
#    out.append(line)
#out = pd.DataFrame(out)

df = df_yield
~a * (df['x'], df['y'])


plt.figure()
plt.gca().invert_yaxis()

c_data = df_yield.col
r_data = df_yield.row
plt.scatter(c_data, r_data)
plt.show()
