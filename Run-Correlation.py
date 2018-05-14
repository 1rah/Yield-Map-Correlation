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

from osgeo import gdal
import affine
import pandas as pd
from math import sqrt


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
    a = affine.Affine.from_gdal(*tif.transform)
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

pt = df_yield.iloc[0]
p1x, p1y = pt['UTM_Easting'], pt['UTM_Northing']

def dist(p1x, p1y, p2x, p2y):
    dx = (p1x - p2x)**2
    dy = (p1y - p2y)**2
    return sqrt(dx+dy)
    


df_ndre['dist'] = df_ndre.apply( lambda row: dist(p1x, p1y, row.x, row.y), axis=1)
