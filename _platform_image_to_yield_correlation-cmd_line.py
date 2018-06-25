# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:16:00 2018

@author: Irah Wajchman
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
        yield_grid_filt = filter_low(yield_grid, pval)
        new_img_array_filt = filter_low(new_img_array, pval)
    else:
        def zero_filter(a): return np.where(a==0, np.nan, a)
        yield_grid_filt = zero_filter(yield_grid)
        new_img_array_filt = zero_filter(new_img_array)
    
    if rebin_data:
        yield_grid_filt = rebin(yield_grid_filt, bins)
        new_img_array_filt = rebin(new_img_array_filt, bins)
    
    # filter out nan values - based on filter_low()
    f = (~np.isnan(yield_grid_filt)) & (~np.isnan(new_img_array_filt))
    
    
    # calc the correlation coeff
    img_corr_coef = np.corrcoef(
            yield_grid[f].flatten(),
            new_img_array[f].flatten(),
            )
    
    # set filter values to 0
#    yield_grid = np.where(new_img_array == 0, 0, yield_grid)
#    yield_grid = np.where(np.isnan(yield_grid), 0, yield_grid)
    yield_grid = np.where(f, yield_grid, 0)
    
    out = dict()
    out.update({
            'new_img_array':new_img_array,
            'df':df,
            'yield_grid':yield_grid,
            'val_filter':f,
            'img_corr_coef':img_corr_coef,
            'new_affine':new_affine,
            })
    
    return out


if __name__ == '__main__':
    ## COMMAND LINE INPUTS
    
    parser = argparse.ArgumentParser(description='Yield Correlation Map')
    parser.add_argument('tif_path', type=str, help='input image: a single channel index in UTM coordinates (epsg:327xx or 326xx)')
    parser.add_argument('csv_path', type=str, help='yield information: csv with 3 columns "easting northing yield" in same coordinates as input image.')
    parser.add_argument('-out_path', default = None, type=str, help='write the yield map as an image in the same format as the input image.')
    parser.add_argument('-gsd', nargs = 1, default = [20,], type=int, help='the ground sample distance to use for computing the yield grid')
    parser.add_argument('-show_plots', action='store_true', help='use swith to show plots')
    
    
    args = parser.parse_args()
    print(args)
    
    #define inputs
#    tif_plane_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T000000_HIRAMS_PLN_ndre_gray_not_corrected.float.tif'
#    tif_satt_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T001059_T55JGH_S2B_ndre_gray.float.tif'
#    csv_path = r'D:\Yield-Map-Correlation\Terraidi_same_date_comparison\Terraidi_field3_Exported CSV Points Cotton Yield.CSV'
#    
#    #tif_path =  tif_plane_path # tif_satt_path
#    tif_path = r'D:\Yield-Map-Correlation\Terraidi_NDRE_rasters\20180212T001111_T55JGH_S2A_ndre_gray.float.tif'
#    csv_path = r'D:\Yield-Map-Correlation\Terraidi_field_CSV Points Cotton Yield\Terraidi_field1_Exported CSV Points Cotton Yield.CSV'
#    
#    out_path = r'D:\Yield-Map-Correlation\test.tif'
#    
#    usr_gsd = 20
#    bins = 50
#    rebin_data = True
#    show_plots = True
#    filter_data = True
#    pval = 3
#    write_out = True
    
    
    tif_path = args.tif_path
    csv_path = args.csv_path
    usr_gsd = args.gsd[0]
    show_plots = args.show_plots
    
    if args.out_path is None:
        write_out = False
        out_path = None
    else:
        write_out = True
        out_path = args.out_path
    
    rebin_data = True
    bins = 50
    filter_data = False
    pval = 3
    
    o = main_func(
            tif_path = tif_path,
            csv_path = csv_path,
            usr_gsd = usr_gsd,
            bins = bins,
            rebin_data = rebin_data,
            filter_data = filter_data,
            pval = pval,
            )
    
    new_img_array = o['new_img_array']
    df = o['df']
    yield_grid = o['yield_grid']
    f = o['val_filter']
    img_corr_coeff = o['img_corr_coef']
    new_affine = o['new_affine']

    print({'correlation coefficient':img_corr_coeff[0,1]})

    
    if write_out:
        with rio.open(tif_path) as img:
            profile = img.profile.copy()
            t = img.affine.to_gdal()
        
        profile.update({'transform':t})
        profile.update({'affine':new_affine})
        h, w = yield_grid.shape
        profile.update({'height':h})
        profile.update({'width':w})
        
        with rio.open(out_path, 'w', **profile) as dst:
            def norm(a): return (a - a.min()) / a.max()
            yield_grid = norm(yield_grid)
            dst.write( np.array(yield_grid, dtype = profile['dtype']), 1)

    if show_plots:
        plt.figure()
#        plt.tight_layout()
        
        class next_subplot:
            def __init__(self):
                self.i=1
            def step(self):
                plt.subplot(1,4,self.i)
                self.i+=1
        
        ns = next_subplot()
        ns.step()
        plt.imshow(new_img_array)
        
        ns.step()
        plt.imshow(yield_grid)
        
        ns.step()
        plt.imshow(yield_grid)
        plt.scatter(df.col, df.row)
        
        ns.step()
        plt.scatter( new_img_array[f].flatten(), yield_grid[f].flatten() )
        plt.xlabel('image px values')
        plt.ylabel('avg yield per px')
        
        plt.show()


## TODO: FOR FUTURE REF - used for calculating index - yield transformation
    
#    new_img_array = o['new_img_array']
#    df = o['df']
#    yield_grid = o['yield_grid']
#    f = o['val_filter']
##    y2 = o['yield_fit']
##    icc = o['img_correlation_coeff'] 
##    fcc = o['fit_correlation_coeff'] 
##    fr2 = o['fit_r_squared_score']
#    
#    
#    def norm_array(a,f=f): return a #return (a - a[f].min()) / a[f].max()
#    yield_grid = norm_array(yield_grid)
#    new_img_array = norm_array(new_img_array)
#
#    x = new_img_array[f].flatten()[:,np.newaxis]
#    y = (yield_grid[f].flatten()) #fit
#    yield_fit, _, _, _ = np.linalg.lstsq(x, y)
#    
#    img_corr_coef = np.corrcoef(
#            yield_grid[f].flatten(),
#            new_img_array[f].flatten(),
#            )
#    
#    #use fit / model to predict yield
#    pred = (yield_fit * new_img_array[f]) #fit
#    
#    fit_r_squared = r2_score(
#            yield_grid[f],
#            pred,
#            )
#    fit_corr_coef = np.corrcoef(
#            yield_grid[f].flatten(),
#            pred,
#            )
#    
#    
#    if show_plots:
#        plt.figure()
#        plt.tight_layout()
#        
#        class next_subplot:
#            def __init__(self):
#                self.i=1
#            def step(self):
#                plt.subplot(3,2,self.i)
#                self.i+=1
#        
#        ns = next_subplot()
#        ns.step()
#        plt.imshow(new_img_array)
#        
#        ns.step()
#        plt.imshow(yield_grid)
#        
#        ns.step()
#        plt.imshow(yield_grid)
#        plt.scatter(df.col, df.row)
#
#        ns.step()
#        plt.scatter( new_img_array[f].flatten(), yield_grid[f].flatten() )
#        x = np.linspace(0, new_img_array[f].max(), 20)
#        plt.plot(x, (yield_fit*x)) #fit
#        
##        plt.figure()
##        plt.hist(yield_grid[f].flatten(), bins=bins)
##        plt.hist((new_img_array[f].flatten()), bins=bins)
#        
#        print(img_corr_coef,'\n\n',fit_corr_coef, fit_r_squared)
#        
#        
#        ##print(np.corrcoef(zi[f].flatten(), new_img_array[f].flatten()) )
#        #print(np.corrcoef(zi[f].flatten(), (new_img_array[f].flatten())) )
#        #print(r2_score(zi[f], (y2*new_img_array[f])))
#        #
#        #
#        #plt.figure()
#        #interp_yield = np.where(np.isnan(img_array), np.nan, np.sqrt(y2*img_array))
#        #plt.imshow(interp_yield)
