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
from sklearn.metrics import r2_score

## HELPER FUNCTIONS

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


def filter_low(a, pval = 3):
    a = np.where(a==0, np.nan, a)
    f = ~np.isnan(a)
    #filter out small vals
    lwr = np.percentile(a[f], pval)
    a = np.where(a<lwr, np.nan, a)
    #filter out high vals
    upr = np.percentile(a[f], 100-pval)
    a = np.where(a>upr, np.nan, a)
    return a


def rebin(a, bins):
    f = ~np.isnan(a)
    _cnt, b_vals = np.histogram(a[f].flatten(), bins=bins)
    dig = np.digitize(a, b_vals)
    for i, b in enumerate(b_vals):
        dig = np.where( dig==b, b, dig)
    dig = np.where( np.isnan(a), np.nan, dig)
    return dig


def rescale_to_gsd(img_array, img_affine, new_gsd):
    img_gsd = abs(np.array((img_affine.a, img_affine.e)))
    new_shape = (
            img_array.shape / 
            (new_gsd / img_gsd)
            )
    new_shape = new_shape.astype(np.int)
    new_img_array = resize(
            img_array,
            new_shape,
            )
    x_rs, y_rs = np.array(img_array.shape) / new_img_array.shape
    new_affine = list(img_affine[0:6])
    new_affine[0] = new_affine[0] * x_rs
    new_affine[4] = new_affine[4] * y_rs
    new_affine = affine.Affine(*new_affine)
    return new_img_array, new_affine


def interpolate_df_to_img_grid(df, new_img_array):
    rs,cs = new_img_array.shape
    ci = np.arange(cs)
    ri = np.arange(rs)
    c,r,z = df.col, df.row, df['yield']
    zi = griddata((c, r), z, (ci[None,:], ri[:,None]), method='linear')
    return zi


def main_func(**args):
    
    ## MAIN FUNCTION

    #! NOTE: assume in UTM format, single channel (index)
    with rio.open(tif_path) as img:
        epsg_code = img.meta['crs']['init'].split(":")[-1]
        assert re.match(u'(327\d\d|326\d\d)', epsg_code)
        img_affine = img.affine
        img_array = img.read(1)
        
    
    new_img_array, new_affine = rescale_to_gsd(img_array, img_affine, usr_gsd)
    
    
    
    #open csv
    #! NOTE: Assume yield csv has headings: Easting, Northing, Yield and is same CRS as the image (UTM)
    names="easting northing yield".split()
    df = pd.read_csv(csv_path)
    df.columns = names
    
    df['col'], df['row'] = ~new_affine * (df['easting'], df['northing'])
#    df['yield_norm'] = ((df['yield'] - df['yield'].min())/df['yield'].max())
    
    csv_sample_dist = get_closest_dist(df)
    # TODO: check if gsd < min image and csv sample distance
    
    # interpolate csv data to the img grid
    yield_grid = interpolate_df_to_img_grid(df, new_img_array)
    
    if filter_data:
        yield_grid_filt = filter_low(yield_grid)
        new_img_array_filt = filter_low(new_img_array)
    else:
        def zero_filter(a): return np.where(a==0, np.nan, a)
        yield_grid_filt = zero_filter(yield_grid)
        new_img_array_filt = zero_filter(new_img_array)
    
    if rebin_data:
        yield_grid_filt = rebin(yield_grid_filt, bins)
        new_img_array_filt = rebin(new_img_array_filt, bins)
    
    # filter out nan values - based on filter_low()
    f = (~np.isnan(yield_grid_filt)) & (~np.isnan(new_img_array_filt))

    def norm_array(a,f=f): return a #return (a - a[f].min()) / a[f].max()
    yield_grid_norm = norm_array(yield_grid)
    new_img_array_norm = norm_array(new_img_array)

    x = new_img_array_norm[f].flatten()[:,np.newaxis]
    yield_fit, _, _, _ = np.linalg.lstsq(x, (yield_grid_norm[f].flatten()) )
    
    img_corr_coef = np.corrcoef(
            yield_grid_norm[f].flatten(),
            new_img_array_norm[f].flatten(),
            )
    
    #use fit / model to predict yield
    pred = yield_fit * new_img_array[f]
    
    fit_r_squared = r2_score(
            yield_grid_norm[f],
            pred,
            )
    fit_corr_coef = np.corrcoef(
            yield_grid_norm[f].flatten(),
            pred,
            )
    
    out = dict()
    out.update({
            'new_img_array':new_img_array,
            'df':df,
            'yield_grid':yield_grid,
            'val_filter':f,
            'yield_fit':yield_fit,
            'img_correlation_coeff':img_corr_coef,
            'fit_correlation_coeff':fit_corr_coef,
            'fit_r_squared_score':fit_r_squared,
            })
    
    return out


if __name__ == '__main__':
    ## COMMAND LINE INPUTS
    
    #define inputs
    tif_plane_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T000000_HIRAMS_PLN_ndre_gray_not_corrected.float.tif'
    tif_satt_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T001059_T55JGH_S2B_ndre_gray.float.tif'
    csv_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\Terraidi_field3_Exported CSV Points Cotton Yield.CSV'
    
    #tif_path =  tif_plane_path # tif_satt_path
    tif_path = r'D:\Yield-Map-Correlation\Terraidi_NDRE_rasters\20180212T001111_T55JGH_S2A_ndre_gray.float.tif'
    csv_path = r'D:\Yield-Map-Correlation\Terraidi_field_CSV Points Cotton Yield\Terraidi_field1_Exported CSV Points Cotton Yield.CSV'
    
    usr_gsd = 20
    bins = 50
    rebin_data = True
    filter_data = True
    show_plots = True
    
    o = main_func(
            tif_path = tif_path,
            csv_path = csv_path,
            usr_gsd = 20,
            bins = 50,
            rebin_data = True,
            filter_data = True,
            )
    
    new_img_array = o['new_img_array']
    df = o['df']
    zi = o['yield_grid']
    y2 = o['yield_fit']
    f = o['val_filter']
    icc = o['img_correlation_coeff'] 
    fcc = o['fit_correlation_coeff'] 
    fr2 = o['fit_r_squared_score']
    
    
    if show_plots:
        plt.figure()
        plt.tight_layout()
        
        class next_subplot:
            def __init__(self):
                self.i=1
            def step(self):
                plt.subplot(3,2,self.i)
                self.i+=1
        
        ns = next_subplot()
        ns.step()
        plt.imshow(new_img_array)
        
        ns.step()
        plt.imshow(zi)
        
        ns.step()
        plt.imshow(zi)
        plt.scatter(df.col, df.row)

        ns.step()
        plt.scatter( new_img_array[f].flatten(), zi[f].flatten() )
        x = np.linspace(0, new_img_array[f].max(), 20)
        plt.plot(x, (y2*x))
        
#        plt.figure()
#        plt.hist(zi[f].flatten(), bins=bins)
#        plt.hist((new_img_array[f].flatten()), bins=bins)
        
        print(icc,'\n\n',fcc, fr2)
        
        
        ##print(np.corrcoef(zi[f].flatten(), new_img_array[f].flatten()) )
        #print(np.corrcoef(zi[f].flatten(), (new_img_array[f].flatten())) )
        #print(r2_score(zi[f], (y2*new_img_array[f])))
        #
        #
        #plt.figure()
        #interp_yield = np.where(np.isnan(img_array), np.nan, np.sqrt(y2*img_array))
        #plt.imshow(interp_yield)
