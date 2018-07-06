# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:05:21 2018

@author: Irah Wajchman

@description:
    convert a shapefile in WGS84 to csv with headers 'easting northing yield epsg_code'
    easting and northing are in UTM
    uses 'Dry Yield' for yield column
    optional point filtering by kml 

@reference: https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.htmlhttps://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html
"""

from osgeo import ogr
import json
import pandas as pd
import matplotlib.pyplot as plt
import utm
import argparse

def get_utm(lat, lng):
    east, north, zn_num, zn_ltr = utm.from_latlon(lat,lng)
    if zn_ltr in 'CDEFGHJKLM':
        epsg_code = 32700 + zn_num
    else:
        assert zn_ltr in 'NPQRSTUVWXX'
        epsg_code = 32600 + zn_num
    return {'easting':east,
            'northing':north,
            'epsg_code':epsg_code}

def get_features(layer):
    pt_list=[]
    for ftr in layer:
        ftr = json.loads(ftr.ExportToJson())
        lng, lat = ftr['geometry']['coordinates']
        yld = ftr['properties']['Dry_Yield']
        if yld and yld > 0:
            p = get_utm(lat, lng)
            p.update({
                    'longitude':lng,
                    'latitude':lat,
                    'yield':yld,
                    })
            pt_list.append(p)
    df = pd.DataFrame(pt_list)
    return df

def main(
        shapefile = r'D:\Steve_SA_winter_2017 Yield Data\2017 Yield Data\Clark Bros #30.shp',
        csvfile = r'D:\Steve_SA_winter_2017 Yield Data\sample.csv',
        kmlfile = r'D:\Steve_SA_winter_2017 Yield Data\sample_kml_fld30_export.kml',
        print_epsg = True,
        show_plot = True,
        ):
    
    # open shapefile
    dataSource = ogr.Open(shapefile, 0)
    layer = dataSource.GetLayer()
    
    # set filter if kml specified
    if kmlfile is not None:
        print('filtering to '+kmlfile)
        kml_datasource = ogr.Open(kmlfile)
        kml_layer = kml_datasource.GetLayer()
        for ftr in kml_layer:
            g = ftr.geometry()
            wkt = g.ExportToWkt()
            break # assume 1 feature
        layer.SetSpatialFilter(ogr.CreateGeometryFromWkt(wkt))
    
    df = get_features(layer)
    assert len(df.epsg_code.unique()) #all points should be in same UTM zone
    
    if print_epsg:
        print('epsg:{}'.format(df.epsg_code.iloc[0]))
    
    if show_plot:
        plt.figure()
        plt.scatter(df.easting, df.northing)
        plt.show()
    
    df['easting northing yield epsg_code'.split()].to_csv(csvfile)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='input args')
    parser.add_argument('shapefile', type=str, help='shapefile (.shp) to convert (WGS84 latitude, longitude)')
    parser.add_argument('csvfile', type=str, help='output csv with 3 columns "easting northing yield" (UTM coordinates)')
    parser.add_argument('-kmlfile', type=str, help='filter points inside kml boundary (kml must contain a single polygon in WGS84 coords)')
    parser.add_argument('-print_epsg', action='store_true', help='use swith to show plots')
    parser.add_argument('-show_plot', action='store_true', help='use swith to show plots')
    args = parser.parse_args()
    main(**vars(args))


