# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:47:47 2018

@author: Irah Wajchman


@description:
    Convert a yield csv with headers "easting northing yield epsg_code" (and UTM coordinates)
    to a single channel Geotiff
    
"""

import pandas as pd
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
from scipy.interpolate import griddata
import affine
import argparse
import math
import re
import geopandas as gpd
import fiona
import utm
fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by default

def bound_to_utm(bnds):
    lng1, lat1, lng2, lat2 = bnds
    
    u1 = utm.from_latlon(lat1, lng1)
    u2 = utm.from_latlon(lat2, lng2)
    if u1[-2:] != u2[-2:]:
        raise Exception('geometry split across multiple UTM zones')
    
    letter = u1[-1]
    zone = u1[-2]
    # get reference zone
    if letter in 'CDEFGHJKLM':
        new_crs = 32700 + zone
    else:
        assert letter in 'NPQRSTUVWXX'
        new_crs = 32600 + zone
    
    return new_crs


def main_func(
        csv_file = r'D:\Yield-Map-Correlation\inputs\Terraidi_field1_Exported CSV Points Cotton Yield.csv',
        out_tiff = r'D:\Yield-Map-Correlation\inputs\Terraidi_field1_Exported CSV Points Cotton Yield.tif',
        gsd = 10,
        show_plot = False,
        normalise = False,
        print_minmax = True,
        kml_file =  r'D:\Yield-Map-Correlation\inputs\G & C Houston - Terriadi - 1.kml'
        ):
    
    print(locals())
    print(csv_file)
    #open csv
    df = pd.read_csv(csv_file)
    k = list(df.keys())
    inorth =[i for i,kn in enumerate(k) if re.search('north', kn, re.IGNORECASE)]
    ieast = [i for i,kn in enumerate(k) if re.search('east', kn, re.IGNORECASE)]
    iyield = [i for i,kn in enumerate(k) if re.search('yield', kn, re.IGNORECASE)]
    if not 1 == len(inorth) == len(ieast) == len(iyield) and not len(set(inorth+ieast+iyield)) == 3:
        raise TypeError('CSV file must contain a single column for each of yield, northing, easting')
    df['yield'] = df.iloc[:,iyield[0]]
    df['easting'] = df.iloc[:,ieast[0]]
    df['northing'] = df.iloc[:,inorth[0]]
    
    max_yield = df['yield'].max()
    min_yield = df['yield'].min()
    left, right = df['easting'].min(), df['easting'].max()
    top, bottom = df['northing'].min(), df['northing'].max()
    
    kf = gpd.read_file(kml_file)
    epsg_n = bound_to_utm(kf.bounds.values[0])
    
    #normalise to 255 get UTM (return mapping in metadata)
    if normalise == True:
        df['px'] = ((df['yield']-min_yield) / (max_yield-min_yield))*255    
        out_dtype = [np.uint8, 'uint8']
    else:
        df['px'] = df['yield']
        out_dtype = [np.float32, 'float32']
    
    #get min, max boundaries
    width = right-left
    height = bottom-top
    
    #generate grid
    n_cols = math.ceil(width/20)+1
    n_rows = math.ceil(height/20)+1
    
    #generate raster data
    a = affine.Affine(gsd,0,left,0,-gsd,bottom)
    df['row'], df['col'] = ~a * (df['easting'], df['northing'])
    print(df)
    ci = np.arange(math.ceil(df['row'].max()))
    ri = np.arange(math.ceil(df['col'].max()))
    c,r,z = df['row'], df['col'], df['px']
    zi = griddata((c, r), z, (ci[None,:], ri[:,None]), method='linear')
    zi = np.where(np.isnan(zi),0,zi).astype(out_dtype[0])

    #define tiff profile
    epsg_code = df['epsg_code'][0]
    zi_height, zi_width = zi.shape
    profile = dict()
    profile.update({
            'driver': 'GTiff',
            'dtype': out_dtype[1],
            'nodata': 0,
            'width': zi_width,
            'height': zi_height,
            'count': 1,
            'crs':rio.crs.CRS.from_epsg(epsg_code),
            'affine':a,
            })
    
    #write output file
    with rio.open(out_tiff, "w", **profile) as dest:
        dest.write(zi, 1)

    #plot outputs
    if show_plot == True:
        plt.figure()
        plt.subplot(121)
        plt.scatter(df['easting'],df['northing'])
        plt.subplot(122)
        plt.scatter(df['row'],df['col'], alpha=0.2, marker='+', cmap=df['px'])
        plt.imshow(zi)
        plt.show()
        
    if print_minmax == True:
        print('min_yield={}, max_yield={}'.format(min_yield, max_yield))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='write yield csv to image')
    parser.add_argument('csv_file', type=str, help='Path to csv file with headers "easting northing yield epsg_code" (and UTM coordinates)')
    parser.add_argument('out_tiff', type=str, help='Path and filename to write output tiff')
    parser.add_argument('-gsd', default=10, type=int, help="gsd of output image (default = 10m)")
    parser.add_argument('-normalise', action='store_true', help='normalise raster to 8 bit image (values 0 to 255)')
    parser.add_argument('-print_minmax', action='store_true', help='print the minimum and maximum yield values from input csv')
    parser.add_argument('-show_plot', action='store_true', help='show ouput plots of raster vs input csv pts')
    args = parser.parse_args()    
    kwargs = vars(args)
    main_func(**kwargs)
    

