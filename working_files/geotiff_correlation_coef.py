# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:48:00 2018

@author: FluroSat

@ desctiption: 
    Compare two Geotiffs - extract overlapping region, match GSD, determine correlation coefficient
"""

import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.tools.mask import mask

import numpy as np
import argparse
import re
import shapely.geometry as sg
from skimage.transform import resize



def run_main(
        tif_a = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T000000_HIRAMS_PLN_ndre_gray_not_corrected.float.tif',
        tif_b = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T001059_T55JGH_S2B_ndre_gray.float.tif',
        gsd = 0,
        show_plots=True,
        ):

    
    img_b = rio.open(tif_b)
    img_a = rio.open(tif_a)
    
    box_a = sg.box(*img_a.bounds)
    box_b = sg.box(*img_b.bounds)
    
    # Check input geotiffs are in valid and matching projections
    if re.match('(326|327)\d\d',img_a.crs.to_dict()['init'].split(':')[-1]) is None: #check input is in UTM
        raise TypeError('input images must be in UTM (epsg 326xx, 327xx) input CRS: {}'.format(img_a.crs))
    if img_a.crs != img_b.crs:
        raise ValueError('input images have non-matching CRS: {} and {}'.format(img_a.crs, img_b.crs))
    
    #check for overlaps
    ol = box_a.intersection(box_b)
    
    ol_width = abs(ol.bounds[0] - ol.bounds[2])
    ol_height = abs(ol.bounds[1] - ol.bounds[3])
    
    # check input GSD is valid
    if gsd < 0 or gsd > ol_width/2 or gsd > ol_height/2:
        raise ValueError("gsd must be positive value and smaller than a 1/2 region of the overlapping region")
  
    def crop_to_overlap(tif):
        # crop image to overlap region
        with rio.open(tif) as img:
            masked, new_affine = mask(img, [ol.__geo_interface__], crop=True)
            cropped = np.where(masked.mask,np.nan,masked.data)[0] 
        return cropped, new_affine
    
    cropped_a, affine_a = crop_to_overlap(tif_a)
    cropped_b, affine_b = crop_to_overlap(tif_b)
    
    resize_kw = {'anti_aliasing':False, 'preserve_range':True, 'mode':'constant'}
    #set to defined gsd
    if gsd > 0:
        new_shape = np.array((ol_height/gsd, ol_width/gsd), dtype=np.int)
        cropped_a = resize(cropped_a, new_shape, **resize_kw)
        cropped_b = resize(cropped_b, new_shape, **resize_kw)
    
    #set both images to same size (smallest of the two)
    elif cropped_b.shape != cropped_a.shape:
        if cropped_a.size < cropped_b.size:
            cropped_b = resize(cropped_b, cropped_a.shape, **resize_kw)
        else:
            cropped_a = resize(cropped_a, cropped_b.shape, **resize_kw)
    else:
        assert cropped_b.shape == cropped_a.shape
        
    #cal correlation coeffictient
    corr_coef = np.corrcoef(cropped_a.flatten(), cropped_b.flatten())[0][1]
    print(corr_coef)
    
    if show_plots:
        plt.figure()
        plt.subplot(221)
        plt.imshow(img_a.read(1))
        plt.subplot(222)
        plt.imshow(img_b.read(1))
        plt.subplot(223)
        plt.imshow(cropped_a)
        plt.subplot(224)
        plt.imshow(cropped_b)
        plt.show()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='write yield csv to image')
    p.add_argument('tif_a', type=str, help='Path to first geotiff img')
    p.add_argument('tif_b', type=str, help='Path to first geotiff img')
    p.add_argument('-gsd', type=int, default=0, help='Specify image GSD to perfom analysis on')
    p.add_argument('-show_plots', action='store_true', help='Show plots of rasters')
    args = p.parse_args()
    kwargs = vars(args)
    run_main(**kwargs)

