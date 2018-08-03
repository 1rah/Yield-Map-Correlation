# -*- coding: utf-8 -*-
"""
Created on Thu Aug 02 12:06:59 2018

@author: FluroSat
"""

from glob import glob
import convert_yield_csv_to_geotiff as cyg
import geopandas as gpd
import fiona
fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by default


csv_file = r'D:\Yield-Map-Correlation\inputs\Terraidi_fiel1_Exported CSV Points Cotton Yield.csv'
kml_file = r'D:\Yield-Map-Correlation\inputs\G & C Houston - Terriadi - 1.kml'

cyg.main_func(
        csv_file=csv_file,
        out_tiff = csv_file.replace(r'.tif', r'.csv'),
        gsd = 10,
        show_plot = True,
        normalise = True,
        print_minmax = False,
        kml_file = kml_file
        )