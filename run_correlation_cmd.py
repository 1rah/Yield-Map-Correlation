# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:56:05 2018

@author: FluroSat
"""

import subprocess

subprocess.call('python geotiff_correlation_coef.py "D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T000000_HIRAMS_PLN_ndre_gray_not_corrected.float.tif" "D:\Yield-Map-Correlation\Terraidi_same_date_comparison\20171219T001059_T55JGH_S2B_ndre_gray.float.tif" -show_plots -gsd 50')
