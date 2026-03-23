import os, pathlib, shutil
import numpy as np
import scipy

import pandas as pd
import xml.etree.ElementTree as ET

from osgeo import gdal
#GDAL ref : https://gdal.org/en/stable/doxygen/classGDALDataset.html

import ASF

def download(buoy_data, data_dir='data', delta=86400 * 10, offset = 86400*0.1, cred_user=None, cred_pass=None):
    '''
    Search and download SLC TOPS burst files overlapping with buoy time-space data from ASF Vertex.
    
    buoy_data : full or partial pandas DataFrame from IABP .csv file
    data_dir  : downloads directory
    delta  : minimum time sampling density (seconds)
    offset : maximum time offset of image from a buoy measurement (seconds)
    cred_user : EarthData or ASF credentials
    cred_pass : EarthData or ASF credentials
    '''
    buoy_ids = set(buoy_data.BuoyID)
    for buoy_id in buoy_ids:
        buoy_path = data_dir+'/'+str(buoy_id)
        if not os.path.exists(buoy_path): os.mkdir(buoy_path)
        scroller = ASF.BuoyScroller(buoy_id, buoy_data)
        scroller.search(delta=delta, offset=offset)
        scroller.download_results(cache_dir=buoy_path,cred_user=cred_user,cred_pass=cred_pass)

def process_downloads(buoy_data, data_dir='data', idate=3):
    '''
    sort sentinel-1 file format {data_dir}/{BUOY_ID}/S1_capID_swath_datetime_... into
    {data_dir}/{BUOY_ID}/{DATETIME}/... format
    '''
    buoy_ids = set(buoy_data.BuoyID)
    for buoy_id in buoy_ids:
        buoy_path = data_dir+'/'+str(buoy_id)
        fnl = [el for el in os.listdir(buoy_path) if len(el.split('.'))==2]
        dtl = list(set([el.split('_')[idate] for el in fnl]))
        for dtn in dtl:
            dtn_path = buoy_path + '/' + str(dtn)
            if not os.path.exists(dtn_path): os.mkdir(dtn_path)
            [os.rename(buoy_path+'/'+fn, dtn_path+'/'+"_".join(fn.split('_')[4:])) for fn in fnl if dtn in fn]
            merge_gdal(dtn_path)
            parse_date(buoy_data
            
            
def merge_gdal(merge_dir, fn_o='src.tiff', master_tags=['HH','HV','VH','VV']):
    '''merge all .tiff in {merge_dir} into {merge_dir}/src.tiff
    note : master_tags entries are length-2 strings at start of target filename.
    note2: master files must have an associated .xml file to be recognized as such.'''
    fn_i = [el for el in os.listdir(merge_dir) if el.split('.')[-1]=='tiff']
    imaster = [i for i in range(len(fn_i)) if fn_i[i][:2] in master_tags and os.path.exists(os.path.join(merge_dir,os.path.splitext(fn_i[i])[0]+'.xml'))][0]
    # - copy master metadata to src.tiff
    ds_i = gdal.Open(os.path.join(merge_dir,fn_i[imaster]))
    iw, ih, irdt, isrs, igtf = ds_i.RasterXSize, ds_i.RasterYSize, ds_i.GetRasterBand(1).DataType, ds_i.GetProjection(), ds_i.GetGeoTransform()
    ds_o = gdal.GetDriverByName("GTiff").Create(os.path.join(merge_dir,fn_o), iw, ih, 1, irdt)
    ds_o.SetProjection(isrs)
    ds_o.SetGeoTransform(igtf)
    
    band_idx = 1
    for i in range(len(fn_i)):
        ds_i = gdal.Open(os.path.join(merge_dir, fn_i[i]))
        if (i != imaster): ds_i = gdal.Warp('',ds_i,format="MEM",width=iw, height=ih, dstSRS=isrs, dstGeoTransform=igtf, resampleAlg='nearest')
        for j in range(1, ds_i.RasterCount + 1):
            if (band_idx > 1) : ds_o.AddBand(irdt)
            ds_o.GetRasterBand(band_idx).WriteArray(ds_i.GetRasterBand(j).ReadAsArray())
            band_idx += 1
    
    shutil.copy( # -- copy .xml file from master .tiff
        os.path.join(merge_dir, os.path.splitext(fn_i[imaster])[0]+'xml'),
        os.path.join(merge_dir, os.path.splitext(fn_o)[0]+'.xml'))
    )
    ds_i = None
    ds_o = None

def parse_date(src_fn, buoy_data, datestring, window=(1024,128)):
    '''identifies position of buoys with least time delta and creates slice 'tgt_{window}.tiff''''
    gdal_data = gdal.Open(src_fn)
    
    burst_idx = int([el for el in path.split('/')[-1].split('_') if len(el)==3 and el[:2]=='IW'][0][-1]) - 1
    grid = ET.parse(path + '.xml').getroot().find('metadata').findall('product')[burst_idx]
    grid = grid.find('content').find('geolocationGrid').find('geolocationGridPointList')
    meta_gcps = pd.DataFrame([{e.tag : e.text for e in el} for el in grid])
    slc_dt = datetime.datetime.strptime(meta_gcps['azimuthTime'].iloc[0], '%Y-%m-%dT%H:%M:%S.%f')
    
    buoy_data['dt'] = buoy_data.apply(lambda x: datetime.datetime(int(x['Year']), int(x['Month']), int(x['Day']),
                                                            int(x['Hour']), int(x['Minute']), int(x['Second'])), axis=1)
    buoy_data['ddt'] = abs(buoy_data['dt'] - slc_dt)
    
    closest = buoy_data.sort_values(by='ddt').iloc[0]
    if (closest.Lon > 180): closest.Lon -= 360

    ll2px_worker = ll2px_function(gdal_data)
    paint_target(gdal_data, lat=closest.Lat, lon=closest.Lon, window=window, ll2px_worker=ll2px_worker)

def ll2px_function(dataset):
    
    points = np.array([[gcp.GCPLine, gcp.GCPPixel, gcp.GCPY, gcp.GCPX] for gcp in dataset.GetGCPs()])
    return scipy.interpolate.RBFInterpolator(points[:,2:],points[:,:2])

def paint_target(dataset, lat, lon, ll2px_worker=None, window=(1024, 128), add_mask=False, overwrite=True):
    
    if (dataset.RasterCount == 1):
        print('dataset only has one band; data processing may be incomplete')
        #raise ValueError('This dataset contains no bonus data! Choose a processed data .tiff.')
    if(ll2px_worker is None):
        ll2px_worker = ll2px_function(dataset)
        
    tg_i, tg_j = np.round(ll2px_worker([[lat, lon]])[0]).astype(int)

    # action 1 : create ice target mask in SLC .tiff

    if (add_mask):
        dataBand = dataset.GetRasterBand(1) #SLC default band
        dataBand.CreateMaskBand(gdal.GMF_PER_DATASET) #mask shared by all bands in dataset
        maskBand = dataBand.GetMaskBand()
    
        np_mask = np.zeros((dataBand.YSize, dataBand.XSize), dtype=bool)
        np_mask[tg_i, tg_j] = True
    
        maskBand.WriteArray(np_mask)

    # action 2 : write target subset .tiff
    # src : https://gdal.org/en/stable/api/python/utilities.html#osgeo.gdal.TranslateOptions
    upper= max(0,tg_i - window[0]//2)
    left = max(0,tg_j - window[1]//2)
    width  = min(window[0], dataset.RasterXSize - left)
    height = min(window[1], dataset.RasterYSize - upper)
    print (f'creating window of (left,width,upper,height) ({left} {width}, {upper} {height}) for raster of shape ({dataset.RasterXSize},{dataset.RasterYSize})')
    path = pathlib.Path(dataset.GetDescription()).parent
    # should create new dataset at target location
    if (not overwrite and os.path.exists(f"{path}/target.tiff")):
        raise ValueError('target file already exists! Suspending operation.')
    #targ_subset = gdal.Translate(destName=f"{path}/target.tiff", srcDS=dataset, srcWin=[left,upper,window[1],window[0]])
    targ_subset = gdal.Translate(destName=f"{path}/target.tiff", srcDS=dataset, srcWin=[left,upper,width,height])
    # targ_subset = None