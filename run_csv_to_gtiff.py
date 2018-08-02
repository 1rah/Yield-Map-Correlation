# -*- coding: utf-8 -*-
"""
Created on Thu Aug 02 12:06:59 2018

@author: FluroSat
"""

from glob import glob
import convert_yield_csv_to_geotiff as cyg


csv_file_list = glob(r'D:\Yield-Map-Correlation\inputs\*.csv')
for csv_file in csv_file_list:
    cyg.main_func(
            csv_file=csv_file,
            out_tiff = csv_file.replace(r'.tif', r'.csv'),
            gsd = 10,
            show_plot = True,
            normalise = True,
            print_minmax = False,
            )