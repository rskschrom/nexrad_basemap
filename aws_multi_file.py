import boto3
from numpy import array, argmin, abs
from datetime import datetime, timedelta
import os
#from nws_ppi_times import plot_low_sweeps, plot_single
from nws_qvp import single_qvp

# function to convert string time 'HHMMSS' into number of seconds past midnight
def dtstring2secs(dtstring):
    yyyy = int(dtstring[0:4])
    mm = int(dtstring[4:6])
    dd = int(dtstring[6:8])

    h = int(dtstring[9:11])  
    m = int(dtstring[11:13])
    s = int(dtstring[13:15])
    ts = datetime(yyyy, mm, dd, h, m, s).timestamp()
    return ts

# set up access to aws server
s3 = boto3.resource('s3')
bucket = s3.Bucket('noaa-nexrad-level2')

# set radar site and set time as now
radsite = 'KBGM'
'''
year = 2020
month = 7
day = 25
hour = 19
minute = 0
end = datetime(year, month, day, hour, minute)
'''
code_min = 0
end = datetime.utcnow()+timedelta(minutes=code_min)

# get beginning loop time
loop_min = 12*60
loop_len = timedelta(minutes=loop_min)
start = end-loop_len

# merge all files between start and end days
syyyy = start.year
smm = start.month
sdd = start.day
sday_dt = datetime(syyyy, smm, sdd, 0, 0)

eyyyy = end.year
emm = end.month
edd = end.day
eday_dt = datetime(eyyyy, emm, edd, 0, 0)

# loop over days between start and end times
nday = (eday_dt-sday_dt).days
keys = []
fnames = []
times = []

for i in range(nday+1):
    now = sday_dt+timedelta(days=i)
    yyyy = now.year
    mm = now.month
    dd = now.day
    print(yyyy, mm, dd)

    # read files from server
    prefix = '{:04d}/{:02d}/{:02d}/{}'.format(yyyy, mm, dd, radsite)
    objs = bucket.objects.filter(Prefix=prefix)
    keys_new = [o.key for o in objs]
    fnames_new = [k.split('/')[-1] for k in keys_new]
    times_new = [f.replace(radsite, '').replace('_V06', '') for f in fnames_new]

    # append to arrays for all days
    keys = keys + keys_new
    fnames = fnames + fnames_new
    times = times + times_new

# loop over times and download files
print(fnames)
nexrad_files = []

for i in range(loop_min):
    now = start+timedelta(minutes=i)
    yyyy = now.year
    mm = now.month
    dd = now.day
    hh = now.hour
    mn = now.minute
    ss = now.second

    # get file with closest time to want time
    secs = [dtstring2secs(t) for t in times]
    want_time = f'{yyyy:04d}{mm:02d}{dd:02d}_{hh:02d}{mn:02d}{ss:02d}'
    want_secs = dtstring2secs(want_time)
    secs_arr = array(secs)
    closeind = argmin(abs(secs_arr-want_secs))

    # download file
    dkey = keys[closeind]
    dfile = fnames[closeind]
    if not os.path.isfile(dfile):
        s3_client = boto3.client('s3')
        s3_client.download_file('noaa-nexrad-level2', dkey, dfile)
        os.system('gunzip {}'.format(dfile))

    # add filename to list of ones to plot
    if not dfile in nexrad_files:
        nexrad_files.append(dfile)
        print(nexrad_files)

# create qvp files
for nf in nexrad_files:
     print(nf)
     try:
        single_qvp(radsite, nf, sw_ang=10.)
     except OSError:
        print('bad data...')
