import datetime, time, calendar
import pathlib, itertools, random, threading, getpass
import concurrent.futures

import pickle

import asf_search
from asf_search.download.file_download_type import FileDownloadType

def wrap_360m180(x):
    if (x > 180):
        return x - 360
    return x
    
class BuoyScroller:
    
    def __init__(self, buoy_id, buoy_data, query_retries=5):
        self.query_retries = query_retries

        self.label = f'{buoy_id}_BuoyScroller'

        buoy_datax = buoy_data[buoy_data['BuoyID']==buoy_id]
        
        # extract time & position
        self.dates = list(buoy_datax.apply(lambda x: datetime.datetime(int(x['Year']), int(x['Month']), int(x['Day']),
                                                                      int(x['Hour']), int(x['Minute']), int(x['Second'])), axis=1))
        self.wktgeos = buoy_datax.apply(lambda x: f"POINT({float(wrap_360m180(x['Lon']))} {float(x['Lat'])})", axis=1)
        self.wktgeos.index = self.dates
        self.wktgeos = dict(self.wktgeos)
        
        # initialize result list
        self.results = {}

    def save_cache(self, fn=None):
        if (fn is None):
            fn = f'{self.label}.pickle'
        with open(fn, mode='wb') as f:
            return pickle.dump(self, f)
    
    @classmethod
    def from_cache(self, file_name):
        with open(file_name, mode='rb') as f:
            return pickle.load(f)
    
    def download_results(self, cache_dir=None, cred_user=None, cred_pass=None, max_workers=10, fileType=FileDownloadType.ALL_FILES):
        if (cache_dir is None): cache_dir = f'{self.label}_downloads'
        pathlib.Path(cache_dir).mkdir(exist_ok=True)

        if (cred_user is None or cred_pass is None):
            cred_user = input('Username:')
            cred_pass = getpass.getpass('Password:')
            
        session = asf_search.ASFSession().auth_with_creds(cred_user, cred_pass)
        rate_limiter = RateLimiter(rate_per_sec=4.0)
        
        asf_results = list(itertools.chain.from_iterable(self.results.values()))
        
        downloader = self.single_download
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(downloader,
                product=result,
                path=cache_dir,
                session=session,
                fileType=fileType,
                limiter=rate_limiter
            ) for result in asf_results]
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()
                
    def single_download(self, product, path, session, limiter, fileType):
        for url in product.get_urls(fileType):
            filename = product.properties['fileID'] + '.' + url.split('.')[-1]
            if (pathlib.Path(path+'/'+filename).exists()):
                print(f'{filename} already exists - skipping download.')
                return
            for i in range(self.query_retries):
                limiter.acquire()
                try:
                    asf_search.download.download_url(url, path=path, filename=filename, session=session)
                    break
                except Exception as e:
                    print(f"Error on {filename}:{e} \n ... retrying {self.query_retries - i} more times.")
                    single_results = None
                    time.sleep((2 ** i) + random.random())
    
    def search(self, delta=86400, offset=43200, max_results=10, dataset=asf_search.constants.DATASET.SLC_BURST, polarization=None):
        
        # set temporal density 
        access_dates = []
        for date in self.dates:
            if (len(access_dates) > 0 and access_dates[-1] + datetime.timedelta(seconds=delta) > date):
                continue
            access_dates.append(date)
        
        #BEGIN SEARCH QUERIES
        self.results = {}
        rate_limiter = RateLimiter(rate_per_sec=4.0)

        # day-by-day search
        tasks = [(date, self.wktgeos[date]) for date in access_dates]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(
                self.single_search_geo,
                rate_limiter,
                dataset,
                polarization,
                wkt,
                date,
                date + datetime.timedelta(seconds=offset),
                max_results
            ) for date, wkt in tasks]
            for future in concurrent.futures.as_completed(futures):
                date_int, result = future.result()
                if (len(result) > 0):
                    print(f'{len(result)} results found!')
                    self.results.setdefault(date_int, []).extend(result)

    
    def single_search_geo(self, limiter, dataset, polarization, wkt, start_dt, end_dt, max_results=10):
        print("Querying:", start_dt, wkt)
        single_results = None
        for i in range(self.query_retries):
            limiter.acquire()
            try:
                single_results = asf_search.geo_search(
                    dataset = dataset,
                    polarization=polarization,
                    intersectsWith = wkt,
                    start=start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    end=end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    maxResults=max_results )
                break
            except Exception as e:
                print(f"Error on {start_dt}-{end_dt}:{e} \n ... retrying {self.query_retries - i} more times.")
                single_results = None
                time.sleep((2 ** i) + random.random())
        if (single_results is None):
            single_results = []
        return int(start_dt.strftime('%Y%m%d%H%M%S')), single_results


class RateLimiter:
    def __init__(self, rate_per_sec, capacity=None):
        self.rate = rate_per_sec
        self.capacity = capacity or rate_per_sec
        self._tokens = self.capacity
        self._lock = threading.Lock()
        self._last = time.perf_counter()

    def acquire(self):
        while True:
            with self._lock:
                now = time.perf_counter()
                # refill tokens
                delta = now - self._last
                self._last = now
                self._tokens = min(self.capacity, self._tokens + delta * self.rate)

                if self._tokens >= 1:
                    self._tokens -= 1
                    return  # success — caller may proceed

            # no token available: sleep a bit and retry
            time.sleep(1.0 / self.rate)