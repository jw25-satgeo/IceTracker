import numpy as np
import datetime
import scipy
import os
import pickle
import netCDF4


class Single_Dataset:
    default_label = "Dataset_latest"

    @staticmethod
    def load_cache(cache_dir=None, cache_fn=None):
        if (cache_dir is None): cache_dir = f'{default_label}_cache'
        if (cache_fn is None): cache_fn = 'pdset.pickle'
        fn = os.path.join(cache_dir, cache_fn)
        with open(fn, 'rb') as f: return pickle.load(f)

    def cache(self, cache_fn=None):
        if (cache_fn is None): cache_fn = f'{self.cache_dir}/pdset.pickle'
        with open(cache_fn, 'wb') as f:
            pickle.dump(self, f)

#SNAP
import unit_lookup
def SNAP_date2ts(datestring):
    try: return datetime.datetime.strptime(datestring, '%d-%b-%Y %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
    except: return datetime.datetime.strptime(datestring, '%d-%b-%Y %H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp()
def SNAP_unitFixer(meta_tuple):
    data, unit, desc = meta_tuple
    if unit in unit_lookup.metric2base.keys(): unit, factor = unit_lookup.metric2base[unit]; data *= factor
    elif unit == 'utc': data, unit = SNAP_date2ts(data), 's'
    return data, unit, desc

# base file import & preprocessing w. SNAPISTA
class Single_Dataset_SNAPbase(Single_Dataset):
    default_label="SNAPbase_latest"
    def __init__(self, src_fn=None, cache_dir=None, nc_fn=None, read_nc=True, alos1_deskew=False):
        self.cache_dir = cache_dir
        if (nc_fn is None): nc_fn = 'BEAM.nc'
        self.nc_fn = nc_fn
        if (cache_dir is None): self.cache_dir = f'{self.default_label}_cache'
        if((cache_dir is None) or (not read_nc)):
            self.SnapistaProc(src_fn, nc_fn, alos1_deskew=alos1_deskew)
        elif (nc_fn in os.listdir(self.cache_dir)):
            print(f'located cache at {self.cache_dir}')
        else: raise ValueError(f'could not find cache at {self.cache_dir}.')
        self.ncGet()
        print('dataset loaded.')
        
    def SnapistaProc(self, src_fn, nc_fn, alos1_deskew=False):
        from esa_snappy import snapista as snp
        print ('beginning ESA SNAPPY preprocessing..')
        graph = snp.Graph()
        bnd_used = ['i_HH','q_HH,i_HV,q_HV,i_VH,q_VH,i_VV,q_VV']
        dem_used = 'Copernicus 30m Global DEM'
        graph.add_node(operator=snp.Operator('Read', file=src_fn), node_id = 'read') # Read
        calib = snp.Operator('Calibration', sourceBandNames=bnd_used, outputImageInComplex='true', outputSigmaBand='false', selectedPolarisations='HH,HV,VH,VV')
        if (alos1_deskew):
            deskew = snp.Operator('ALOS-Deskewing', sourceBandNames=bnd_used, demName=dem_used) # Deskew
            graph.add_node(operator = deskew, node_id = 'deskew', source = 'read')
            graph.add_node(operator = calib, node_id = 'calib', source = 'deskew')
        else: graph.add_node(operator = calib, node_id = 'calib', source = 'read')
        write = snp.Operator('Write', file=f'{self.cache_dir}/{nc_fn}', formatName='NETCDF4-BEAM')
        graph.add_node(operator = write, node_id = 'write', source = 'calib')
        try: os.mkdir(self.cache_dir)
        except FileExistsError: print(f'{self.cache_dir} already exists - data will be overwritten.')
        graph.run()

    def ncGet(self):
        ncd = netCDF4.Dataset(os.path.join(self.cache_dir, self.nc_fn))
        print('extracting parameters...')
        # Metadata retrieval
        def get_attr(filt, fix_unit=False): 
            attrns = [el for el in ncd.variables['metadata'].ncattrs() if filt in el]
            data, desc, unit = None, None, None
            for i in range(len(attrns)):
                if ('_unit' == attrns[i][-5:]): unit = ncd.variables['metadata'].getncattr(attrns[i])
                elif ('_descr' == attrns[i][-6:]): desc = ncd.variables['metadata'].getncattr(attrns[i])
                else: data = ncd.variables['metadata'].getncattr(attrns[i])
            if (fix_unit): data, unit, desc = SNAP_unitFixer((data,unit,desc))
            return data, unit, desc
        self.metadata = {}

        #radar processing properties
        self.metadata['wv'] = get_attr('wavelength', True)
        self.metadata['freq'] = get_attr('radar_frequency', True)
        self.metadata['rg_bw'] = get_attr('range_bandwidth', True)
        self.metadata['rg_sr'] = get_attr('range_sampling_rate', True)
        self.metadata['rg_sp'] = get_attr('range_spacing', True)
        self.metadata['az_bw'] = get_attr('azimuth_bandwidth', True)
        self.metadata['az_sr'] = get_attr('azimuth_sampling_rate', True)
        self.metadata['az_sp'] = get_attr('azimuth_spacing', True)
        self.metadata['prf'] = get_attr('pulse_repetition_frequency', True)
        self.metadata['0dopp_t0'] = get_attr('first_line_time', True)
        self.metadata['0dopp_t1'] =  get_attr('last_line_time', True)
        self.metadata['inc_near'] = get_attr('incidence_near', True)
        self.metadata['inc_far'] = get_attr('incidence_far', True)
        
        #orbit state vectors
        buffer, labels = [], ['Abstracted_Metadata:Orbit_State_Vectors:orbit_vector','time', 'x_pos', 'y_pos', 'z_pos', 'x_vel', 'y_vel','z_vel']
        for i in sorted(set([int(el.split('orbit_vector')[1].split(':')[0]) for el in ncd.variables['metadata'].ncattrs() if labels[0] in el])):
            tmp = [ncd.variables['metadata'].getncattr(labels[0] + f'{i}:{el}') for el in labels[1:]]; tmp[0] = SNAP_date2ts(tmp[0])
            tmp = [float(el) for el in tmp]; buffer.append(tmp)
        self.metadata['orbit_vector'] = np.array(buffer); buffer = None
            
        #latitude & longitude corners
        self.metadata['lat'] = [get_attr('first_near_lat')[0], get_attr('first_far_lat')[0], get_attr('last_near_lat')[0], get_attr('last_far_lat')[0]]
        self.metadata['lon'] = [get_attr('first_near_long')[0], get_attr('first_far_long')[0], get_attr('last_near_long')[0], get_attr('last_far_long')[0]]
        #match label of range and azimuth dimensions
        rg = ncd.variables['slant_range_time'][:]
        self.dim_desc = ('band', 'azi', 'rng') if (np.abs(rg[-1,0]-rg[0,0]) < np.abs(rg[0,-1]-rg[0,0])) else ('band','rng','azi')
        
        print('retrieved metadata.')

        slckey = [el for el in ncd.variables.keys() if 'i_' in el or 'q_' in el]
        slckey = [(slckey[i], slckey[i+1]) for i in np.arange(0,len(slckey),2)]
        self.nc_slckey = slckey
        self.slc = np.array([ncd.variables[el[0]][:] + ncd.variables[el[1]][:] * 1j for el in slckey])
        print('retrieved SLC complex.')

        print('rescaling auxiliary data...')
        ia = ncd.variables['incident_angle'][:]
        refmap = np.meshgrid(np.linspace(0,ia.shape[0]-1,self.slc.shape[1]),
                             np.linspace(0,ia.shape[1]-1,self.slc.shape[2]), indexing='ij')
        def rescale_tp2ref(tp): return scipy.ndimage.map_coordinates(tp, refmap)
        if (ia > np.pi/2).any(): ia = np.radians(ia); print('converted incidence angle from degrees to radians.')
        self.ia = rescale_tp2ref(ia)
        self.rg = rescale_tp2ref(ncd.variables['slant_range_time'][:])
        self.lat = rescale_tp2ref(ncd.variables['latitude'][:])
        self.lon = rescale_tp2ref(ncd.variables['longitude'][:])
        print('all parameters extracted.')