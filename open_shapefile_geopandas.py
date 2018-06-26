# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:05:21 2018

@author: FluroSat

@ref:
    https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.htmlhttps://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html
    https://automating-gis-processes.github.io/2017/lessons/L3/point-in-polygon.html
"""

from osgeo import ogr
import json
import pandas as pd
import matplotlib.pyplot as plt
#from shapely.geometry import Polygon
#import geopandas as gpd
#import numpy as np

shapefile = r'D:\Steve_SA_winter_2017 Yield Data\2017 Yield Data\Clark Bros #30.shp'
kmlfile = r'D:\Steve_SA_winter_2017 Yield Data\sample_kml_fld30_export.kml'

kml_datasource = ogr.Open(kmlfile)
kml_layer = kml_datasource.GetLayer()
for ftr in kml_layer:
    g = ftr.geometry()
    wkt = g.ExportToWkt()


dataSource = ogr.Open(shapefile, 0)
layer = dataSource.GetLayer()

layer.SetSpatialFilter(ogr.CreateGeometryFromWkt(wkt))

pt_list=[]
for ftr in layer:
    ftr = json.loads(ftr.ExportToJson())
    pts = ftr['geometry']['coordinates']
    yld = ftr['properties']['Dry_Yield']
    if yld > 0:
        pt_list.append({
                'longitude':pts[0],
                'latitude':pts[1],
                'dry_yield':yld,
                })

df = pd.DataFrame(pt_list)
plt.scatter(df.latitude, df.longitude)




#gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
#polys = gpd.read_file(kmlfile, driver='KML')
#points = gpd.read_file(shapefile)
#
#m = points.within(polys.loc[0,'geometry'])
#
#points.plot()
#points[m].plot()