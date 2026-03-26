import os, pathlib, shutil, warnings
import numpy as np
import scipy
import cv2

import datetime
import pandas as pd
import xml.etree.ElementTree as ET

import concurrent.futures

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

def process_downloads(buoy_data, data_dir='data', idate=3, window=(1024, 128)):
    '''
    sort sentinel-1 file format {data_dir}/{BUOY_ID}/S1_capID_swath_datetime_... into
    {data_dir}/{BUOY_ID}/{DATETIME}/... format
    '''
    buoy_ids = set(buoy_data.BuoyID)
    for buoy_id in buoy_ids:
        buoy_path = data_dir+'/'+str(buoy_id)
        fnl = [el for el in os.listdir(buoy_path) if len(el.split('.'))==2] # exclude directories
        dtl = list(set([el.split('_')[idate] for el in fnl])) + [el for el in os.listdir(buoy_path) if len(el.split('.'))==1]
        for dtn in dtl:
            print(f'processing: date {dtn} for buoy {buoy_id}')
            dtn_path = buoy_path + '/' + str(dtn)
            if not os.path.exists(dtn_path): os.mkdir(dtn_path)
            #[os.rename(buoy_path+'/'+fn, dtn_path+'/'+"_".join(fn.split('_')[4:])) for fn in fnl if dtn in fn]
            [os.rename(os.path.join(buoy_path,fn), os.path.join(dtn_path,fn)) for fn in fnl if dtn in fn]
            merge_gdal(dtn_path, fn_o='src.tiff', master_tags=['HH','HV','VH','VV'])
            crop_target(os.path.join(dtn_path,'src.tiff'), buoy_data[buoy_data.BuoyID==buoy_id], window)            

# ---- raster registration & stacking

def merge_gdal(merge_dir, fn_o='src.tiff', master_tags=['HH','HV','VH','VV'], no_warp_precision=1.0, stride=(24,8), batch_factor=16, max_workers=None, override_gdal_cache=True):
    '''merge all .tiff in {merge_dir} into {merge_dir}/src.tiff
    note : master_tags is a list of strings to be located between underscores.
    note2: master files must have an associated .xml file to be recognized as such.'''
    fn_i = [el for el in os.listdir(merge_dir) if os.path.splitext(el)[1]=='.tiff' and el != fn_o]
    imaster = [
        i for i in range(len(fn_i))
        if bool(set(fn_i[i].split('_')) & set(master_tags)) # any2any match
        and os.path.exists(os.path.join(merge_dir,os.path.splitext(fn_i[i])[0]+'.xml'))
    ][0]
     # clear existing file
    if (os.path.exists(os.path.join(merge_dir, fn_o))):
        os.remove(os.path.join(merge_dir, fn_o))
    
    # - copy master metadata to src.tiff
    gdal.UseExceptions()
    if (override_gdal_cache): gdal.SetCacheMax(gdal.GetUsablePhysicalRAM() // 4)
    ds_i = gdal.Open(os.path.join(merge_dir,fn_i[imaster]))
    iw,ih,irdt,icmp = ds_i.RasterXSize,ds_i.RasterYSize,ds_i.GetRasterBand(1).DataType,ds_i.GetMetadata('IMAGE_STRUCTURE')
    ds_o = gdal.GetDriverByName("MEM").Create('', iw, ih, 1, irdt)
    ds_o.SetGCPs(ds_i.GetGCPs(), ds_i.GetGCPProjection())
    ds_i = None
    
    # - transfer all bands from source .tiffs to src.tiff
    # primary GCP
    f1 = np.array([(e.GCPLine, e.GCPPixel, e.GCPX, e.GCPY, e.GCPZ/10000) for e in ds_o.GetGCPs()]) # scale Z by 1/10000 for more lat/lon sensitivity
    f1 = scipy.interpolate.RBFInterpolator(f1[:,:2], f1[:,2:]) # line 2 geo
    f1_coords = np.stack(np.meshgrid(np.arange(0,ih+stride[0],stride[0]), np.arange(0,iw+stride[1],stride[1]), indexing='ij'),axis=-1)
    f1_geords = f1(f1_coords.reshape((np.prod(f1_coords.shape[:-1]), f1_coords.shape[-1]))) #N,ngeoparams
    # secondary rasters
    band_idx = 1
    for i in range(len(fn_i)):
        ds_i = gdal.Open(os.path.join(merge_dir, fn_i[i]))
        f2 = np.array([(e.GCPLine, e.GCPPixel, e.GCPX, e.GCPY, e.GCPZ/10000) for e in ds_i.GetGCPs()])
        f2 = scipy.interpolate.RBFInterpolator(f2[:,2:], f2[:,:2]) # geo 2 line
        f2_coords = f2(f1_geords).astype(np.float32).reshape(f1_coords.shape)
        displacement = ((f2_coords - f1_coords)**2).sum(axis=-1)
        no_warp_flag = (iw == ds_i.RasterXSize and ih == ds_i.RasterYSize) and ((i == imaster) or
            (np.mean(displacement) < no_warp_precision**2 and np.max(displacement**0.5) < no_warp_precision*4))
        f2_coords = np.moveaxis(f2_coords, -1, 0) #map_coordinates assumes axis=0 stacking of coordinate arrays
        if (not no_warp_flag): print (f'{fn_i[i]} lacks matching pixel map (MSE:{displacement.mean()}) : resampling raster...')
        for j in range(1, ds_i.RasterCount + 1):
            if (band_idx > 1) : ds_o.AddBand(irdt)
            if (no_warp_flag):
                o_arr = ds_i.GetRasterBand(j).ReadAsArray()
            else:
                rst_arr = ds_i.GetRasterBand(j).ReadAsArray()
                o_arr = np.zeros((ih, iw), dtype=rst_arr.dtype)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_batched_coord_warp, src=rst_arr, dst=o_arr, pi=pi, pj=pj, crds=f2_coords, strd=stride, bf=batch_factor)
                        for pj in range(0,f2_coords.shape[2]+batch_factor,batch_factor)
                        for pi in range(0,f2_coords.shape[1]+batch_factor,batch_factor)
                    ]
                    completed, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
                    #print(len(completed))
            b = ds_o.GetRasterBand(band_idx)
            b.WriteArray(o_arr)
            b.SetDescription(ds_i.GetRasterBand(j).GetDescription())
            band_idx += 1
        ds_i = None
    gdal.Translate(os.path.join(merge_dir, fn_o), ds_o, format='GTiff', creationOptions=[f"{k}={v}" for k,v in icmp.items() if v is not None])
    ds_o = None

    # - copy master metadata to src.xml, removing irrelevant swaths
    tree = ET.parse(os.path.join(merge_dir, os.path.splitext(fn_i[imaster])[0]+'.xml'))
    meta = tree.getroot().find('metadata')
    token_l = fn_i[imaster].split('_')
    i = 0
    while i < len(meta.findall('product')):
        tmp = meta.findall('product')[i]
        swath, polar = tmp.find('swath').text, tmp.find('polarisation').text
        if not (swath in token_l and polar in token_l):
            meta.remove(meta.findall('product')[i])
            meta.remove(meta.findall('noise')[i])
            meta.remove(meta.findall('calibration')[i])
        else : i += 1
    tree.write(os.path.join(merge_dir, os.path.splitext(fn_o)[0]+'.xml'))
    tree = None

def _batched_coord_warp(src, dst, pi, pj, crds, strd, bf):
    y0, x0 = pi * strd[0], pj * strd[1]
    if (y0 >= dst.shape[0] or x0 >= dst.shape[1]): return
    pc1, pc2 = (cv2.resize(crds[0, pi:pi+bf, pj:pj+bf], (strd[1]*bf,strd[0]*bf), cv2.INTER_LINEAR),
                cv2.resize(crds[1, pi:pi+bf, pj:pj+bf], (strd[1]*bf,strd[0]*bf), cv2.INTER_LINEAR))
    y1 = min(y0 + pc1.shape[0], dst.shape[0])
    x1 = min(x0 + pc1.shape[1], dst.shape[1])
    pc1 = np.stack((pc1,pc2), axis=0)[:, :y1-y0, :x1-x0]
    dst[y0:y1,x0:x1] = scipy.ndimage.map_coordinates(src, pc1, order=0)

# ----- buoy location operation

def crop_target(src_fn, buoy_data, window=(1024,128)):
    '''identifies & interpolates position of buoys with least time delta and creates slice 'tgt_{window}.tiff' '''
    gdal.UseExceptions()
    gdal_data = gdal.Open(src_fn)

    # assumes only 1 swath in metadata
    xml_fn = os.path.splitext(src_fn)[0] + '.xml'
    grid = ET.parse(xml_fn).getroot().find('metadata').find('product').find('content').find('geolocationGrid').find('geolocationGridPointList')
    slc_dt = datetime.datetime.strptime(pd.DataFrame([{e.tag : e.text for e in el} for el in grid])['azimuthTime'].iloc[0], '%Y-%m-%dT%H:%M:%S.%f')
    
    buoy_data['dt'] = buoy_data.apply(
        lambda x: datetime.datetime(int(x['Year']), int(x['Month']), int(x['Day']),
                                    int(x['Hour']), int(x['Minute']), int(x['Second'])),
        axis=1
    )
    buoy_data['ddt'] = (buoy_data['dt'] - slc_dt).apply(lambda x: x.total_seconds())

     # get up to 12 closest measurements, then order in time axis ; filter for interpolable numeric values
    closest = buoy_data.sort_values(by='ddt', key=(lambda x: abs(x))).iloc[:12].sort_values(by='ddt').select_dtypes(include='number')
    
    # convert lon-lat to euclidian coordinates to prevent polar instabilities
    closest['llx'] = np.cos(np.deg2rad(closest.Lat)) * np.cos(np.deg2rad(closest.Lon))
    closest['lly'] = np.cos(np.deg2rad(closest.Lat)) * np.sin(np.deg2rad(closest.Lon))
    closest['llz'] = np.sin(np.deg2rad(closest.Lat))

    # interpolation
    ccols = [col for col in closest.columns if col != 'ddt']
    x, y = closest.ddt.values, np.stack([closest[col].values for col in ccols], axis=-1)
    if (x.min() * x.max() > 0): warnings.warn(f"{src_fn} image time does not intersect buoy times: fit will be ill-conditioned.")
    y = scipy.interpolate.PchipInterpolator(x,y)(0)
    closest = pd.Series(y, index=ccols)

    # restore lon-lat from normalized euclidian coordinates
    llnm = (closest['llx']**2 + closest.lly**2 + closest['llz']**2)**0.5
    closest['Lat'] = np.rad2deg(np.asin(closest['llz']/llnm))
    closest['Lon'] = np.rad2deg(np.atan2(closest['lly'],closest['llx']))

    ll2px_worker = ll2px_function(gdal_data)
    _paint_target(gdal_data, lat=closest['Lat'], lon=closest['Lon'], window=window, ll2px_worker=ll2px_worker)

def ll2px_function(dataset):
    
    points = np.array([[gcp.GCPLine, gcp.GCPPixel, gcp.GCPY, gcp.GCPX] for gcp in dataset.GetGCPs()])
    return scipy.interpolate.RBFInterpolator(points[:,2:],points[:,:2])

def _paint_target(dataset, lat, lon, ll2px_worker=None, window=(1024, 128), add_mask=False, overwrite=True):
    
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