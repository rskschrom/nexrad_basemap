import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import numpy as np
from pyart.io.nexrad_archive import read_nexrad_archive
from numpy import genfromtxt
import os
import glob
import time
import calendar
from datetime import datetime, timedelta
from kdp_estimation import calc_kdp

# function to de-alias phidp
#----------------------------------
def dealiasPhiDP(phiDP):
    deal_phi = np.ma.empty([phiDP.shape[0], phiDP.shape[1]])
    deal_phi[phiDP<0.] = 180.+phiDP[phiDP<0.] 
    deal_phi[phiDP>=0.] = phiDP[phiDP>=0.]
    return deal_phi   

# calculate qvp and write to file
#-----------------------------------
def single_qvp(site, fpath, sw_ang=10.):
    # open radar file
    radar = read_nexrad_archive(fpath)
    radlat = radar.latitude['data'][0]
    radlon = radar.longitude['data'][0]
    nyvel = radar.get_nyquist_vel(1)

    # get sweeps
    fixed_angles = radar.fixed_angle['data']
    nang = len(fixed_angles)
    sw05 = np.arange(nang)[np.abs(fixed_angles-sw_ang)<1.]
    sw_inds = sw05[::2]

    # loop over sweeps
    for sw in sw_inds:
        azi = 90.-radar.get_azimuth(sw)
        elev = radar.get_elevation(sw) 
        ran = radar.range['data']

        sweep = radar.extract_sweeps([sw])
        fixed_angle = fixed_angles[sw]

        # calculate sweep time
        vol_time = sweep.time['units']
        sw_toffs = sweep.time['data'][0]
        sw_time = datetime.fromtimestamp(time.mktime(time.strptime(vol_time, 'seconds since %Y-%m-%dT%H:%M:%SZ')))
        sw_time = sw_time+timedelta(seconds=sw_toffs)
        sw_ts_utc = calendar.timegm(sw_time.utctimetuple())

        # get time strings
        yyyy = '{:04d}'.format(sw_time.year)
        mm = '{:02d}'.format(sw_time.month)
        dd = '{:02d}'.format(sw_time.day)
        hh = '{:02d}'.format(sw_time.hour)
        mn = '{:02d}'.format(sw_time.minute)
        ss = '{:02d}'.format(sw_time.second)

        print(yyyy, mm, dd, hh, mn, ss)

        ref = sweep.fields['reflectivity']['data']
        zdr = sweep.fields['differential_reflectivity']['data']
        rhohv = sweep.fields['cross_correlation_ratio']['data']
        phidp = sweep.fields['differential_phase']['data']

        dims = ref.shape
        numradials = dims[0]+1
        numgates = dims[1]
        angle = np.mean(elev)

        # mask data by rhohv and threshold
        #-----------------------------------------------
        ref = np.ma.masked_where(rhohv<0.4, ref)
        zdr = np.ma.masked_where(rhohv<0.4, zdr)
        phidp = np.ma.masked_where(rhohv<0.4, phidp)
        rhohv = np.ma.masked_where(rhohv<0.4, rhohv)

        zdr = np.ma.masked_where(ref<-15., zdr)
        phidp = np.ma.masked_where(ref<-15., phidp)
        rhohv = np.ma.masked_where(ref<-15., rhohv)
        ref = np.ma.masked_where(ref<-15., ref)

        # calculate kdp
        #-----------------------------------------------
        print('Calculating KDP...')
        phidp = dealiasPhiDP(phidp)
        kdp_alt, delta, phidp_alt = calc_kdp(phidp)
        kdp_alt = np.ma.masked_where(ref<-5., kdp_alt)

        # calculate beam height 
        #-----------------------------------------------------------
        radz = 10.
        erad = np.pi*angle/180.

        ke = 4./3.
        a = 6378137.
        zcor = np.sqrt(ran**2.+(ke*a)**2.+2.*ran*ke*a*np.sin(erad))-ke*a+radz

        # generate qvp
        #---------------------
        ref_qvp = np.ma.mean(ref, axis=0)
        zdr_qvp = np.ma.mean(zdr, axis=0)
        kdp_qvp = np.ma.mean(kdp_alt, axis=0)
        rhv_qvp = np.ma.mean(rhohv, axis=0)
        print(rhv_qvp.shape, zcor.shape)

        # write to text file
        fqvp = open(f'{yyyy}{mm}{dd}_{hh}{mn}_{site}.txt', 'w')
        fqvp.write(f'time:{sw_ts_utc:d}\n')
        fqvp.write(f'z\tzh\tzdr\tkdp\trhv\n')

        # limit to 15 km ARL
        nz = np.argmin(np.abs(zcor-15000.))
        ref_qvp[ref_qvp.mask] = -999.
        zdr_qvp[zdr_qvp.mask] = -999.
        kdp_qvp[kdp_qvp.mask] = -999.
        rhv_qvp[rhv_qvp.mask] = -999.
        for i in range(nz):
            fqvp.write(f'{zcor[i]:.1f}\t{ref_qvp[i]:.1f}\t{zdr_qvp[i]:.2f}\t{kdp_qvp[i]:.2f}\t{rhv_qvp[i]:.3f}\n')
        fqvp.close()
