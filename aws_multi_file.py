import boto3
from numpy import array, argmin, abs
from datetime import datetime, timedelta
import os
from nws_qvp import single_qvp

# function to convert string time 'HHMMSS' into number of seconds past midnight
def tstring2secs(tstring):
    h = float(tstring[0:2])
    m = float(tstring[2:4])
    s = float(tstring[4:6])
    return 3600.*h+60.*m+s

# set up access to aws server
s3 = boto3.resource('s3')
bucket = s3.Bucket('noaa-nexrad-level2')

# set radar site and set time as now
radsite = 'KBGM'

year = 2020
month = 1
day = 25
hour = 19
minute = 0
end = datetime(year, month, day, hour, minute)

#end = datetime.utcnow()

# get beginning loop time
loop_min = 360
loop_len = timedelta(minutes=loop_min)
start = end-loop_len

# loop over times and download files
for i in range(loop_min):
    now = start+timedelta(minutes=i)
    yyyy = now.year
    mm = now.month
    dd = now.day
    hh = now.hour
    mn = now.minute
    ss = now.second

    # get keys for restricted files
    prefix = '{:04d}/{:02d}/{:02d}/{}'.format(yyyy, mm, dd, radsite)
    objs = bucket.objects.filter(Prefix=prefix)
    keys = [o.key for o in objs]
    fnames = [k.split('/')[-1] for k in keys]
    times = [f.split('_')[1] for f in fnames]

    # remove erroneous files with 'NEXRAD' in name
    numtimes = 0
    times_valid = []
    for t in times:
        if t=='NEXRAD':
            pass
        else:
            times_valid.append(t)
        numtimes = numtimes+1

    # get file with closest time to want time
    secs = [tstring2secs(t) for t in times_valid]
    want_time = '{:02d}{:02d}{:02d}'.format(hh, mn, ss)
    want_secs = tstring2secs(want_time)
    secs_arr = array(secs)
    closeind = argmin(abs(secs_arr-want_secs))

    # download file
    dkey = keys[closeind]
    dfile = fnames[closeind]
    print(dfile)
    if (not os.path.isfile(dfile)) and (not os.path.isfile(dfile.replace('.gz', ''))):
        s3_client = boto3.client('s3')
        s3_client.download_file('noaa-nexrad-level2', dkey, dfile)
        os.system('gunzip {}'.format(dfile))
        dfile = dfile.replace('.gz', '')

        # create qvp file
        try:
            single_qvp(radsite, dfile, sw_ang=10.)
        except OSError:
            print('bad data...')
