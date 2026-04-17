import os, pathlib, shutil, warnings, getpass, warnings
import numpy as np
import scipy
import cv2

import datetime
import pandas as pd
import xml.etree.ElementTree as ET

import concurrent.futures

from osgeo import gdal, osr
#GDAL ref : https://gdal.org/en/stable/doxygen/classGDALDataset.html

import ASF

def download(buoy_data, data_dir='data', delta=86400 * 10, offset = 86400*0.1, cred_user=None, cred_pass=None, stage=0):
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
    if (cred_user is None or cred_pass is None):
        cred_user = input('Username:')
        cred_pass = getpass.getpass('Password:')
    for buoy_id in buoy_ids:
        buoy_path = os.path.join(data_dir,str(buoy_id))
        if not os.path.exists(buoy_path): os.mkdir(buoy_path)
        scroller_fn = os.path.join(buoy_path, f'{buoy_id}_BuoyScroller.pickle')
        if (os.path.exists(scroller_fn)):
            scroller = ASF.BuoyScroller.from_cache(scroller_fn)
        else:
            scroller = ASF.BuoyScroller(buoy_id, buoy_data)
            scroller.search(delta=delta, offset=offset)
            scroller.save_cache(scroller_fn)
        scroller.download_results(cache_dir=buoy_path,cred_user=cred_user,cred_pass=cred_pass)

def process_downloads(buoy_data, data_dir='data', stages=(0,1,2), yx_res=(3.0, 3.0), window=(128, 128), min_entries=1, overwrite=False, delete_raw=True, idate=3, override_gdal_cache=True):
    '''
    sort sentinel-1 file format {data_dir}/{BUOY_ID}/S1_capID_swath_datetime_... into
    {data_dir}/{BUOY_ID}/{DATETIME}/... format
    '''
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    if (override_gdal_cache): gdal.SetCacheMax(gdal.GetUsablePhysicalRAM() // 4)
    buoy_ids = set(buoy_data.BuoyID)
    for buoy_id in buoy_ids:
        buoy_path = data_dir+'/'+str(buoy_id)
        if (not os.path.exists(buoy_path)): continue
        all_fn_look = os.listdir(buoy_path)
        if (len([el for el in all_fn_look if os.path.splitext(el)[1] != '.pickle']) < min_entries):
            shutil.rmtree(buoy_path); continue
        fnl = [el for el in all_fn_look if (len(el.split('.'))==2 and os.path.splitext(el)[1] != '.pickle')] # exclude directories & .pickle files
        print(fnl)
        dtl = list(set([el.split('_')[idate] for el in fnl])) + [el for el in all_fn_look if len(el.split('.'))==1]
        for dtn in dtl:
            print(f'processing: date {dtn} for buoy {buoy_id}')
            dtn_path = buoy_path + '/' + str(dtn)
            if not os.path.exists(dtn_path): os.mkdir(dtn_path)
            [os.rename(os.path.join(buoy_path,fn), os.path.join(dtn_path,fn)) for fn in fnl if dtn in fn]
            if ((0 in stages) and (overwrite or (not os.path.exists(os.path.join(dtn_path,'src.tiff'))))):
                merge_gdal(dtn_path, fn_o='src.tiff', master_tags=['HH','HV','VH','VV'])
            src_fn = os.path.join(dtn_path, 'src.tiff')
            if (os.path.exists(src_fn)):
                if ((1 in stages) and (overwrite or (not os.path.exists(os.path.join(dtn_path, 'src_pln.tiff'))))):
                    planarproj_op(src_fn, 'src_pln.tiff', yx_res=yx_res, gcp_count=2000)
                if ((2 in stages) and (overwrite or (not os.path.exists(os.path.join(dtn_path, 'tgt_pln.tiff'))))):
                    crop_target_pln(src_fn, 'tgt_pln.tiff', buoy_data[buoy_data.BuoyID==buoy_id], window)
            else: print(f'skipping secondary operation : no source file located at {src_fn}.')
            if (os.path.exists(os.path.join(dtn_path, 'tgt.tiff'))):
                clean_dtn(dtn_path, blacklist=('tgt','tgt_grd','tgt_pln','src','src_grd','src_pln'))

def clean_dtn(clean_dir, blacklist):
    fn_i = [el for el in os.listdir(clean_dir) if os.path.splitext(el)[0] not in blacklist]
    for fn in fn_i: os.remove(os.path.join(clean_dir, fn))


# ---- raster registration & stacking

def merge_gdal(merge_dir, fn_o='src.tiff', master_tags=['HH','HV','VH','VV'], no_warp_precision=1.0, stride=(24,8), batch_factor=16, max_workers=None):
    '''merge all .tiff in {merge_dir} into {merge_dir}/src.tiff
    note : master_tags is a list of strings to be located between underscores.
    note2: master files must have an associated .xml file to be recognized as such.'''
    blacklist = [fn_o, 'target.tiff', 'tgt.tiff', 'src_grd.tiff']
    fn_i = [el for el in os.listdir(merge_dir) if os.path.splitext(el)[1]=='.tiff' and el not in blacklist]
    imaster = [
        i for i in range(len(fn_i))
        if bool(set(os.path.splitext(fn_i[i])[0].split('_')) & set(master_tags)) # any2any match
        and os.path.exists(os.path.join(merge_dir,os.path.splitext(fn_i[i])[0]+'.xml'))
    ]
    if len(imaster) > 0:
        imaster = imaster[0]
    else:
        print(f'could not merge directory {merge_dir} : no file met master_tag condition. Available files:')
        [print(fn) for fn in fn_i]
        return
     # clear existing file
    if (os.path.exists(os.path.join(merge_dir, fn_o))):
        os.remove(os.path.join(merge_dir, fn_o))
    
    # - copy master metadata to src.tiff
    ds_i = gdal.Open(os.path.join(merge_dir,fn_i[imaster]))
    ds_o_srs = osr.SpatialReference(); ds_o_srs.ImportFromWkt(ds_i.GetGCPProjection())
    if ds_o_srs.GetAuthorityCode(None) != '4326': raise ValueError('current program assumes EPSG:4326; input CRS differs.')
    iw,ih,irdt,icmp = ds_i.RasterXSize,ds_i.RasterYSize,ds_i.GetRasterBand(1).DataType,ds_i.GetMetadata('IMAGE_STRUCTURE')
    ds_o = gdal.GetDriverByName("MEM").Create('', iw, ih, 1, irdt)
    ds_o.SetGCPs(ds_i.GetGCPs(), ds_i.GetGCPProjection())
    ds_i = None
    
    # - transfer all bands from source .tiffs to src.tiff
    # primary GCP
    #f1 = np.array([(e.GCPLine, e.GCPPixel, e.GCPX, e.GCPY) for e in ds_o.GetGCPs()]) ISSUE : polar singularity problem
    f1 = np.array([[e.GCPLine, e.GCPPixel] + list(srs_llh2xyz(ds_o_srs, e.GCPY, e.GCPX, e.GCPZ)) for e in ds_o.GetGCPs()])
    f1 = scipy.interpolate.RBFInterpolator(f1[:,:2], f1[:,2:]) # line 2 geo
    f1_coords = np.stack(np.meshgrid(np.arange(0,ih+stride[0],stride[0]), np.arange(0,iw+stride[1],stride[1]), indexing='ij'),axis=-1)
    f1_geords = f1(f1_coords.reshape((np.prod(f1_coords.shape[:-1]), f1_coords.shape[-1]))) #N,ngeoparams
    # secondary rasters
    band_idx = 1
    for i in range(len(fn_i)):
        ds_i = gdal.Open(os.path.join(merge_dir, fn_i[i]))
        ds_i_srs = osr.SpatialReference(); srs.ImportFromWkt(ds_i.GetGCPProjection())
        f2 = np.array([[e.GCPLine, e.GCPPixel] + list(srs_llh2xyz(ds_i_srs, e.GCPY, e.GCPX, e.GCPZ)) for e in ds_i.GetGCPs()])
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

def srs_llh2xyz(srs, lat, lon, h=0): # h : radial height
    a, f = srs.GetSemiMajor(), 1/srs.GetInvFlattening()  # ellipsoid parameters
    e2 = f*(2 - f) # eccentricity squared
    N = (a / np.sqrt(1 - e2*np.sin(np.deg2rad(lat))**2))
    x = (N + h)*np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(lon))
    y = (N + h)*np.cos(np.deg2rad(lat))*np.sin(np.deg2rad(lon))
    z = (N*(1 - e2) + h)*np.sin(np.deg2rad(lat))
    return x, y, z

def srs_xyz2llh(srs, x, y, z):
    a, f = srs.GetSemiMajor(), 1/srs.GetInvFlattening()
    b, p, e2 = a * (1 - f),  np.sqrt(x**2 + y**2), f*(2 - f)
    theta = np.arctan2(a*z, b*p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ((a**2 - b**2)/b)*np.sin(theta)**3, p - a*e2*np.cos(theta)**3)
    h = p / np.cos(lat) - (a / np.sqrt(1 - e2*np.sin(lat)**2))
    return np.rad2deg(lat), np.rad2deg(lon), h
    
def _batched_coord_warp(src, dst, pi, pj, crds, strd, bf):
    y0, x0 = pi * strd[0], pj * strd[1]
    if (y0 >= dst.shape[0] or x0 >= dst.shape[1]): return
    pc1, pc2 = (cv2.resize(crds[0, pi:pi+bf, pj:pj+bf], (strd[1]*bf,strd[0]*bf), cv2.INTER_LINEAR),
                cv2.resize(crds[1, pi:pi+bf, pj:pj+bf], (strd[1]*bf,strd[0]*bf), cv2.INTER_LINEAR))
    y1 = min(y0 + pc1.shape[0], dst.shape[0])
    x1 = min(x0 + pc1.shape[1], dst.shape[1])
    pc1 = np.stack((pc1,pc2), axis=0)[:, :y1-y0, :x1-x0]
    dst[y0:y1,x0:x1] = scipy.ndimage.map_coordinates(src, pc1, order=1)

# ---- georeferencing stage

def planarproj_op(src_fn, fn_o='src_pln.tiff', yx_res=(3.0,3.0), gcp_count=2000, crd_coarse=64):
    ds_i = gdal.Open(src_fn)
    dirpath = pathlib.Path(ds_i.GetDescription()).parent

    transformer = gdal.Transformer(ds_i, None, ["METHOD=GCP_TPS"])
    success, (cen_lon, cen_lat, _) = transformer.TransformPoint(False, ds_i.RasterXSize/2,  ds_i.RasterYSize / 2)
    if (not success): raise ValueError('GCP_TPS method failed to retrieve values.')
    if osr.SpatialReference(ds_i.GetGCPProjection()).GetAuthorityCode(None) != '4326':
        raise ValueError('current program assumes EPSG:4326; input CRS differs.')
    f1 = np.array([[e.GCPLine,e.GCPPixel]+list(srs_llh2xyz(osr.SpatialReference(ds_i.GetGCPProjection()), e.GCPY, e.GCPX, e.GCPZ)) for e in ds_i.GetGCPs()])
    f2 = scipy.interpolate.RBFInterpolator(f1[:,2:], f1[:,:2]) # x/y/z euclidian meters to line/pixel
    f1 = scipy.interpolate.RBFInterpolator(f1[:,:2], f1[:,2:]) # line/pixel to x/y/z in meters (euclidian wgs84)

    bord = np.stack(np.meshgrid(np.arange(ds_i.RasterYSize), np.arange(ds_i.RasterXSize), indexing='ij'), axis=-1)
    bord[1:-1,1:-1] = -1; bord = bord[(bord > -1).any(axis=-1)]; bord = f1(bord) # mask flattens dims 1,2 to boundary coordinates
    c_pos = f1(np.array([ds_i.RasterYSize/2, ds_i.RasterXSize/2])[None,:])[0] # center pixel to center [1,3]
    
    # tangent-plane north & east vectors
    n_e = max((np.cross([0,0,1], c_pos), np.cross([1,0,0], c_pos)), key=lambda v: np.linalg.norm(v));
    n_n = np.cross(c_pos, n_e);
    n_n, n_e = n_n/np.linalg.norm(n_n), n_e/np.linalg.norm(n_e)
    
    # find maximal vertical and horizontal distances
    n_n_d = np.sum((bord - c_pos[None,:]) * n_n[None,:], axis=-1) # dot product
    n_e_d = np.sum((bord - c_pos[None,:]) * n_e[None,:], axis=-1)
    n_n_min, n_n_max = n_n_d.min(), n_n_d.max()
    n_e_min, n_e_max = n_e_d.min(), n_e_d.max()
    ds_y_n, ds_x_n = int(np.ceil((n_n_max - n_n_min)/yx_res[0])), int(np.ceil((n_e_max - n_e_min)/yx_res[1]))
    
    srs4326 = osr.SpatialReference(epsg=4326)
    a = srs4326.GetSemiMajor(); b = a*(1 - 1/srs4326.GetInvFlattening())
    ds_o = gdal.GetDriverByName("MEM").Create('', ysize=ds_y_n, xsize=ds_x_n, bands=ds_i.RasterCount, eType=ds_i.GetRasterBand(1).DataType)

    # generate coordinate grid
    crd_o = np.stack(np.meshgrid( # linspace spanning pixel center locations on coarse grid
        np.linspace(n_n_min + 0.5*(n_n_max-n_n_min)/ds_y_n, n_n_max - 0.5*(n_n_max-n_n_min)/ds_y_n, ds_y_n//crd_coarse),
        np.linspace(n_e_min + 0.5*(n_e_max-n_e_min)/ds_x_n, n_e_max - 0.5*(n_e_max-n_e_min)/ds_x_n, ds_x_n//crd_coarse), indexing='ij'
    ),axis=-1)
    crd_o = crd_o[...,0,None]*n_n[None,None,:] + crd_o[...,1,None]*n_e[None,None,:] + c_pos[None,None,:]  # n,e ->  xyz-coordinates in tangent plane
    crd_o /= ((crd_o**2 / np.array([a, a, b])[None,None,:]**2).sum(axis=-1)**0.5)[...,None] # deform tangent plane to WGS84 ellipsoid surface
    
    # save function map from output dataset pixel coordinates to xyz euclidian
    f3 = scipy.interpolate.RegularGridInterpolator((np.linspace(0,ds_y_n-1,crd_o.shape[0]),np.linspace(0,ds_x_n-1,crd_o.shape[1])), crd_o, bounds_error=False)

    # map xyz coordinates to source image pixel space
    crd_o = f2(crd_o.reshape((-1,3))).reshape((crd_o.shape[0],crd_o.shape[1],2)) # source image pixel coordinates
    crd_o = np.stack([cv2.resize(crd_o[...,i], (ds_x_n, ds_y_n)) for i in range(crd_o.shape[-1])], axis=0)
    occ = np.ones((ds_o.RasterYSize, ds_o.RasterXSize), dtype=np.uint8)
    for i in range(1, ds_i.RasterCount + 1):
        band = scipy.ndimage.map_coordinates(ds_i.GetRasterBand(i).ReadAsArray(), crd_o)
        occ[band == 0] = 0  # map_coordinates sets cval or 0 out of bounds
        ds_o.GetRasterBand(i).WriteArray(band)

    # generate new GCP grid for lat-lon georeference capability
    spc = ((ds_o.RasterYSize*yx_res[0]*ds_o.RasterXSize*yx_res[1])/gcp_count)**0.5
    gcpi = np.stack(np.meshgrid(
        np.arange(0,ds_o.RasterYSize,int(spc/yx_res[0])),
        np.arange(0,ds_o.RasterXSize,int(spc/yx_res[1])),indexing='ij'
    ),axis=-1).astype(float)
    occ = cv2.resize(occ,(gcpi.shape[1],gcpi.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
    gcpi = gcpi[occ]
    f3 = f3(gcpi)
    gcps = srs_xyz2llh(srs4326, *[f3[:,i] for i in range(3)])
    gcps = [gdal.GCP(gcps[1][i], gcps[0][i], gcps[2][i], gcpi[i,1], gcpi[i,0]) for i in range(len(gcps))]
    ds_o.SetGCPs(gcps, srs4326.ExportToWkt())

    ds_i_creationOptions = [f"{k}={v}" for k,v in ds_i.GetMetadata('IMAGE_STRUCTURE').items() if v is not None]
    gdal.Translate(os.path.join(dirpath, fn_o), ds_o, format='GTiff', creationOptions=ds_i_creationOptions)

# ----- buoy location slice

def crop_target_pln(src_fn, fn_o, buoy_data, window=(64,64)):
    '''identifies and interpolates position of buoys with least time delta and creates slice 'tgt.tiff'.'''
    gdal.UseExceptions()
    if (not os.path.exists(src_fn)):
        print(f'could not crop source .tiff : path {src_fn} does not exist'); return
    if (isinstance(src_fn, gdal.Dataset)):
        gdal_data = src_fn; src_fn = src_fn.GetDescription()
    else: gdal_data = gdal.Open(src_fn)

    xml_fn  = os.path.splitext(src_fn)[0] + '.xml'
    by_pt, closest = buoy_loc(xml_fn, buoy_data)
    ds_gcp = np.array([[gcp.GCPLine, gcp.GCPPixel, gcp.GCPY, gcp.GCPX] for gcp in gdal_data.GetGCPs()])
    y, x = scipy.interpolate.RBFInterpolator(ds_gcp[:,2:],ds_gcp[:,:2])([[by_pt.Lat, by_pt.Lon]])[0]
    ymin, xmin = int(round(y - window[0]/2)), int(round(x - window[1]/2))
    print (f'attempting crop at location (y, x) ({y} {x}) for raster of shape (y:{gdal_data.RasterYSize},x:{gdal_data.RasterXSize})')
    path = pathlib.Path(gdal_data.GetDescription()).parent
    pd.concat((pd.DataFrame(by_pt).T, closest)).to_json(os.path.join(path, os.path.splitext(fn_o)[0] + '.json')) #save buoy location
    try: # projWin : subwindow in projected coordinates to extract: [ulx, uly, lrx, lry]
        gdal.Translate(os.path.join(path, fn_o), gdal_data, srcWin=[xmin,ymin,window[1],window[0]], resampleAlg='nearest')
    except: print(f'GDAL failed to write file to {fn_o}')

def buoy_loc(slc_xml_fn, buoy_data, xml_srs=None):
    if (xml_srs is None): xml_srs = osr.SpatialReference(epsg=4326)
    grid = ET.parse(slc_xml_fn).getroot().find('metadata').find('product').find('content').find('geolocationGrid').find('geolocationGridPointList')
    slc_dt = datetime.datetime.strptime(pd.DataFrame([{e.tag : e.text for e in el} for el in grid])['azimuthTime'].iloc[0], '%Y-%m-%dT%H:%M:%S.%f')
    by_dt = buoy_data.copy()
    by_dt.loc[:,'dt'] = by_dt.apply(lambda x: datetime.datetime(
        int(x['Year']), int(x['Month']), int(x['Day']),
        int(x['Hour']), int(x['Minute']), int(x['Second'])
    ), axis=1)
    by_dt.loc[:,'ddt'] = (by_dt['dt'] - slc_dt).apply(lambda x: x.total_seconds())
    
    # get up to 12 closest measurements, then order in time axis ; filter for interpolable numeric values
    # +: convert lon-lat to euclidian coordinates to prevent polar instabilities
    closest = by_dt.sort_values(by='ddt', key=(lambda x: abs(x))).iloc[:12].sort_values(by='ddt').select_dtypes(include='number')
    closest.loc[:,'llx'], closest.loc[:,'lly'], closest.loc[:,'llz'] = srs_llh2xyz(xml_srs, closest.loc[:,'Lat'], closest.loc[:,'Lon'])
    
    # interpolation
    ccols = [col for col in closest.columns if col != 'ddt']
    x, y = closest.ddt.values, np.stack([closest[col].values for col in ccols], axis=-1)
    if (x.min() * x.max() > 0): warnings.warn(f"{src_fn_xml} image time does not intersect buoy times: fit will be ill-conditioned.")
    y = scipy.interpolate.PchipInterpolator(x,y)(0)
    interp = pd.Series(y, index=ccols)

    xyz = np.array([interp.loc['llx'], interp.loc['lly'], interp.loc['llz']]); xyz /= (np.linalg.norm(xyz) / xml_srs.GetSemiMajor())
    interp.loc['Lat'], interp.loc['Lon'], _ = srs_xyz2llh(xml_srs, *xyz)
    return interp, closest


    
#
#
#

#
#
#

#
#
#

#
#
#

#
#
#

#
#
# -- UNUSED : GRD-coordinate target slice (reason: polar instabilities)

def geoproj_op(src_fn, fn_o='src_grd.tiff', ll_res=(5e-5, 5e-5)):
    '''
    geoproject source SLC using GCPs.
    '''
    ds_i = gdal.Open(src_fn)
    dirpath = pathlib.Path(ds_i.GetDescription()).parent
    wo_ = gdal.SuggestedWarpOutput(ds_i, ["DST_SRS=EPSG:4326"])
    print(f'GDAL suggested bounds :: iw:{wo_.width} ih:{wo_.height}, xmin:{wo_.xmin}, xmax:{wo_.xmax}, ymin:{wo_.ymin}, ymax:{wo_.ymax}')
    print(f'.. fixing from resolution x:{(wo_.xmax - wo_.xmin) / wo_.width}, y:{(wo_.ymax - wo_.ymin) / wo_.height} to x:{ll_res[0]}, y:{ll_res[1]}')
    gdal.Warp(
        os.path.join(dirpath, fn_o), ds_i, dstSRS='EPSG:4326', resampleAlg='bilinear', transformerOptions=['METHOD=GCP_TPS'],
        creationOptions=[f"{k}={v}" for k,v in ds_i.GetMetadata('IMAGE_STRUCTURE').items() if v is not None],
        xRes=ll_res[1], yRes=ll_res[0], outputBounds=(wo_.xmin, wo_.ymin, wo_.xmax, wo_.ymax),
    ) # note : x:lon(ind=1), y:lat(ind=0)
def crop_target_grd(src_fn, fn_o, buoy_data, window=(256,256), ll_res=(5e-5, 5e-5), lon_geom_correct=True):
    '''identifies and interpolates position of buoys with least time delta and creates slice 'tgt.tiff'.'''
    if (not os.path.exists(src_fn)):
        print(f'could not crop source .tiff : path {src_fn} does not exist')
        return
    if (isinstance(src_fn, gdal.Dataset)):
        gdal_data = src_fn
        src_fn = src_fn.GetDescription()
    else: gdal_data = gdal.Open(src_fn)
    by_dt = buoy_data.copy()

    xml_fn = os.path.splitext(src_fn)[0] + '.xml'
    closest, _ = buoy_loc(xml_fn, buoy_data)

    # should create new dataset at target location
    lon_correction_factor = 1.0
    if (lon_geom_correct):
        lon_correction_factor = 1/max(np.cos(np.deg2rad(closest.loc['Lat'])), 1e-2)

    lat_min = closest.loc['Lat'] - ll_res[0] * window[0]/2
    lat_max = closest.loc['Lat'] + ll_res[0] * window[0]/2
    lon_min = closest.loc['Lon'] - ll_res[1] * window[1]/2 * lon_correction_factor
    lon_max = closest.loc['Lon'] + ll_res[1] * window[1]/2 * lon_correction_factor
    
    path = pathlib.Path(gdal_data.GetDescription()).parent
    closest.to_json(os.path.join(path, os.path.splitext(fn_o)[0] + '.json')) #save buoy location
    try: # projWin : subwindow in projected coordinates to extract: [ulx, uly, lrx, lry] .. width:lon ; hgt:lat
        targ_subset = gdal.Translate(os.path.join(path, fn_o), gdal_data, width=window[1], height=window[0], projWin=[lon_min,lat_max,lon_max,lat_min])
    except: print(f'GDAL failed to write file to {fn_o}')



# -- UNUSED : SLC-coordinate target slice

def crop_target_slc(src_fn, buoy_data, window=(1024,128), overwrite=False):
    '''identifies & interpolates position of buoys with least time delta and creates slice 'tgt_{window}.tiff' '''
    if (not os.path.exists(src_fn)):
        print(f'could not crop source .tiff : path {src_fn} does not exist')
        return
    gdal_data = gdal.Open(src_fn)
    by_dt = buoy_data.copy()
    xml_fn = os.path.splitext(src_fn)[0] + '.xml'
    closest, _ = buoy_loc(xml_fn, buoy_data)
    ll2px_worker = ll2px_function(gdal_data)
    if (not overwrite and os.path.exists(os.path.join(os.path.dirname(src_fn),"target.tiff"))):
        print ('dataset already has target.tiff; skipping operation')
        return
    _paint_target_slc(gdal_data, lat=closest['Lat'], lon=closest['Lon'], window=window, ll2px_worker=ll2px_worker, overwrite=overwrite)
def ll2px_function(dataset):
    points = np.array([[gcp.GCPLine, gcp.GCPPixel, gcp.GCPY, gcp.GCPX] for gcp in dataset.GetGCPs()])
    return scipy.interpolate.RBFInterpolator(points[:,2:],points[:,:2])
def _paint_target_slc(dataset, lat, lon, ll2px_worker=None, window=(1024, 128), add_mask=False, overwrite=False):
    if (dataset.RasterCount == 1):
        print('dataset only has one band; data processing may be incomplete')
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
    width  = max(min(window[0], dataset.RasterXSize - left), 0)
    height = max(min(window[1], dataset.RasterYSize - upper), 0)
    print (f'creating window of (left,width,upper,height) ({left} {width}, {upper} {height}) for raster of shape ({dataset.RasterXSize},{dataset.RasterYSize})')
    path = pathlib.Path(dataset.GetDescription()).parent
    # should create new dataset at target location
    if (not overwrite and os.path.exists(os.path.join(path, "target.tiff"))):
        raise ValueError('target file already exists! Suspending operation.')
    try: gdal.Translate(destName=os.path.join(path, "target.tiff"), srcDS=dataset, srcWin=[left,upper,width,height])
    except: return
    # targ_subset = None