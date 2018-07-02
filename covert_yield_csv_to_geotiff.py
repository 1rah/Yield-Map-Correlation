# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:47:47 2018

@author: Irah Wajchman


@description:
    Convert a yield csv with headers "easting northing yield" (and UTM coordinates)
    to a single channel Geotiff
    
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




if __name__ == '__main__':
    csvfile = r'D:\Steve_SA_winter_2017 Yield Data\sample.csv'
    
    #open csv
    df = pd.read_csv(csvfile)
    
    #normalise get UTM
    
    
    #generate grid
    
    #get min, max boundaries
    #calc affine from gsd and top left corner