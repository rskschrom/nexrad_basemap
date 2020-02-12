import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import numpy as np
from pyart.io.nexrad_archive import read_nexrad_archive
from numpy import genfromtxt
from collections import Counter
import os
import glob
import time
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from metpy.plots import USCOUNTIES
from kdp_estimation import calc_kdp

# function to de-alias phidp
#----------------------------------
def dealiasPhiDP(phiDP):
    deal_phi = np.ma.empty([phiDP.shape[0], phiDP.shape[1]])
    deal_phi[phiDP<0.] = 180.+phiDP[phiDP<0.] 
    deal_phi[phiDP>=0.] = phiDP[phiDP>=0.]
    return deal_phi   

# function for creating color map
#----------------------------------
def createCmap(mapname):
    fil = open('/home/robert/research/nws/'+mapname+'.rgb')
    cdata = genfromtxt(fil,skip_header=2)
    cdata = cdata/256
    cmap = cm.ListedColormap(cdata, mapname)
    fil.close()
    return cmap

# function to convert x,y to lon,lat
#-----------------------------------
def xy2latlon(x, y, lat0, lon0):
    km2deg = 110.62
    lat = y/km2deg+lat0
    lon = x/(km2deg*np.cos(np.pi*lat0/180.))+lon0
    return lat, lon


# plot all sweeps near 0.5 degrees
#-----------------------------------
def plot_low_sweeps(site, fpath, ds_set=150., xcen_set=150., ycen_set=70., sw_ang=0.5):
    # open radar file
    radar = read_nexrad_archive(fpath)
    radlat = radar.latitude['data'][0]
    radlon = radar.longitude['data'][0]
    nyvel = radar.get_nyquist_vel(1)

    # plot stuff
    mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    mpl.rc('text', usetex=True)
    figcnt = 0

    # get lat lon corners of data domain
    ds = ds_set
    xcen = xcen_set
    ycen = ycen_set

    minlat, minlon = xy2latlon(xcen-ds, ycen-ds, radlat, radlon)
    maxlat, maxlon = xy2latlon(xcen+ds, ycen+ds, radlat, radlon)

    xlabel = 'X-distance (km)'
    ylabel = 'Y-distance (km)'

    # color map stuff
    zh_map = createCmap('zh2_map')
    zdr_map = createCmap('zdr_map')
    vel_map = createCmap('vel2_map')
    phv_map = createCmap('phv_map')
    kdp_map = createCmap('kdp_map')

    numcolors = 33
    numlevs = numcolors-1
    cinds = np.linspace(0., 254., numcolors).astype(int)

    zh_cols = zh_map.colors[cinds]
    zdr_cols = zdr_map.colors[cinds]
    vel_cols = vel_map.colors[cinds]
    phv_cols = phv_map.colors[cinds]
    kdp_cols = kdp_map.colors[cinds]

    zh_levs = np.linspace(-10., 70., numlevs)
    sw_levs = np.linspace(0., 8., numlevs)
    zdr_levs = np.linspace(-2.4, 6.9, numlevs)
    phidp_levs = np.linspace(50., 100., numlevs)
    phv_levs = np.linspace(0.71, 1.06, numlevs)
    kdp_levs = np.linspace(-0.8, 2.3, numlevs)
    vel_levs = np.linspace(-35., 35., numlevs)

    zh_mapn, zh_norm = cm.from_levels_and_colors(zh_levs, zh_cols, extend='both')
    zdr_mapn, zdr_norm = cm.from_levels_and_colors(zdr_levs, zdr_cols, extend='both')
    phidp_mapn, phidp_norm = cm.from_levels_and_colors(phidp_levs, zh_cols, extend='both')
    phv_mapn, phv_norm = cm.from_levels_and_colors(phv_levs, phv_cols, extend='both')
    kdp_mapn, kdp_norm = cm.from_levels_and_colors(kdp_levs, kdp_cols, extend='both')
    vel_mapn, vel_norm = cm.from_levels_and_colors(vel_levs, vel_cols, extend='both')
    sw_mapn, sw_norm = cm.from_levels_and_colors(sw_levs, zh_cols, extend='both')

    # set common font sizes
    cblb_fsize = 18
    cbti_fsize = 16
    axtl_fsize = 20
    axlb_fsize = 20
    axti_fsize = 18

    # cartopy features
    coast = cfeature.GSHHSFeature(scale='high', levels=[1], edgecolor='gray')
    states = cfeature.STATES.with_scale('50m')
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='50m',
                                         category='cultural', name=shapename)

    # create basemap
    #m = Basemap(llcrnrlon=minlon, llcrnrlat=minlat, urcrnrlon=maxlon, urcrnrlat=maxlat,
    #            projection='merc', lat_1=15., lat_2=35., lon_0=-80.,
    #            resolution='h', area_thresh=1000.)

    # get sweeps
    fixed_angles = radar.fixed_angle['data']
    nang = len(fixed_angles)
    sw05 = np.arange(nang)[np.abs(fixed_angles-sw_ang)<0.2]
    print(fixed_angles)

    # seperate out z and mdv sweeps
    sw_inds = sw05[::2]
    swv_inds = sw05[1::2]

    # loop over sweeps
    for sw in sw_inds:
        azi_p = 90.-radar.get_azimuth(sw)
        elev_p = radar.get_elevation(sw) 

        ran = radar.range['data']

        sweep = radar.extract_sweeps([sw])
        fixed_angle = fixed_angles[sw]

        if fixed_angle<2.:
            sweep_v = radar.extract_sweeps([sw+1])
        else:
            sweep_v = sweep

        # calculate sweep time
        vol_time = sweep.time['units']
        sw_toffs = sweep.time['data'][0]
        sw_time = datetime.fromtimestamp(time.mktime(time.strptime(vol_time, 'seconds since %Y-%m-%dT%H:%M:%SZ')))
        sw_time = sw_time+timedelta(seconds=sw_toffs)

        # get time strings
        yyyy = '{:04d}'.format(sw_time.year)
        mm = '{:02d}'.format(sw_time.month)
        dd = '{:02d}'.format(sw_time.day)
        hh = '{:02d}'.format(sw_time.hour)
        mn = '{:02d}'.format(sw_time.minute)
        ss = '{:02d}'.format(sw_time.second)

        print(yyyy, mm, dd, hh, mn, ss)

        ref_p = sweep.fields['reflectivity']['data']
        zdr_p = sweep.fields['differential_reflectivity']['data']
        rhohv_p = sweep.fields['cross_correlation_ratio']['data']
        vel_p = sweep_v.fields['velocity']['data']
        phidp_p = sweep.fields['differential_phase']['data']
        sw_p = sweep_v.fields['spectrum_width']['data']
        azi_v_p = 90.-radar.get_azimuth(sw+1)
        elev_v_p = radar.get_elevation(sw+1)

        dims = ref_p.shape
        numradials = dims[0]+1
        numgates = dims[1]

        # expand radially to remove no data spike
        elev = np.empty([numradials])
        elev_v = np.empty([numradials])
        azi = np.empty([numradials])
        azi_v = np.empty([numradials])
        ref = np.empty([numradials, numgates])
        zdr = np.empty([numradials, numgates])
        phidp = np.empty([numradials, numgates])
        rhohv = np.empty([numradials, numgates])
        vel = np.empty([numradials, numgates])
        sw = np.empty([numradials, numgates])

        elev[0:numradials-1] = elev_p
        elev[numradials-1] = elev_p[0]
        elev_v[0:numradials-1] = elev_v_p
        elev_v[numradials-1] = elev_v_p[0]
        azi[0:numradials-1] = azi_p
        azi[numradials-1] = azi_p[0]
        azi_v[0:numradials-1] = azi_v_p
        azi_v[numradials-1] = azi_v_p[0]
        ref[0:numradials-1,:] = ref_p
        ref[numradials-1,:] = ref_p[0]
        zdr[0:numradials-1,:] = zdr_p
        zdr[numradials-1,:] = zdr_p[0]
        phidp[0:numradials-1,:] = phidp_p
        phidp[numradials-1,:] = phidp_p[0]
        rhohv[0:numradials-1,:] = rhohv_p
        rhohv[numradials-1,:] = rhohv_p[0]
        vel[0:numradials-1,:] = vel_p
        vel[numradials-1,:] = vel_p[0]
        sw[0:numradials-1,:] = sw_p
        sw[numradials-1,:] = sw_p[0]

        # get stats on velocity
        vcount = Counter(vel.flatten())

        angle = np.mean(elev)
        angle_v = np.mean(elev_v)

        # mask data by rhohv and threshold
        #-----------------------------------------------
        ref = np.ma.masked_where(rhohv<0.4, ref)
        zdr = np.ma.masked_where(rhohv<0.4, zdr)
        phidp = np.ma.masked_where(rhohv<0.4, phidp)
        vel = np.ma.masked_where(rhohv<0.4, vel)
        sw = np.ma.masked_where(rhohv<0.4, sw)
        rhohv = np.ma.masked_where(rhohv<0.4, rhohv)

        zdr = np.ma.masked_where(ref<-15., zdr)
        phidp = np.ma.masked_where(ref<-15., phidp)
        rhohv = np.ma.masked_where(ref<-15., rhohv)
        vel = np.ma.masked_where(ref<-15., vel)
        sw = np.ma.masked_where(ref<-15., sw)
        ref = np.ma.masked_where(ref<-15., ref)

        # calculate kdp
        #-----------------------------------------------
        print('Calculating KDP...')
        phidp = dealiasPhiDP(phidp)
        kdp_alt, delta, phidp_alt = calc_kdp(phidp)
        kdp_alt = np.ma.masked_where(ref<-5., kdp_alt)

        # calculate x and y coordinates (wrt beampath) for plotting
        #-----------------------------------------------------------
        ran_2d = np.tile(ran,(numradials,1))
        azi.shape = (azi.shape[0], 1)
        azi_v.shape = (azi_v.shape[0], 1)
        azi_2d = np.tile(azi,(1,numgates))
        azi_v_2d = np.tile(azi_v,(1,numgates))

        radz = 10.
        erad = np.pi*angle/180.
        erad_v = np.pi*angle_v/180.

        ke = 4./3.
        a = 6378137.

        # beam height and beam distance
        zcor = np.sqrt(ran_2d**2.+(ke*a)**2.+2.*ran_2d*ke*a*np.sin(erad))-ke*a+radz
        scor = ke*a*np.arcsin(ran_2d*np.cos(erad)/(ke*a+zcor))/1000.

        xcor = scor*np.cos(np.pi*azi_2d/180.)
        ycor = scor*np.sin(np.pi*azi_2d/180.)

        # for velocity
        zcor_v = np.sqrt(ran_2d**2.+(ke*a)**2.+2.*ran_2d*ke*a*np.sin(erad_v))-ke*a+radz
        scor_v = ke*a*np.arcsin(ran_2d*np.cos(erad_v)/(ke*a+zcor))/1000.

        xcor_v = scor_v*np.cos(np.pi*azi_v_2d/180.)
        ycor_v = scor_v*np.sin(np.pi*azi_v_2d/180.)

        # convert to lon,lat for basemap plotting
        lat, lon = xy2latlon(xcor, ycor, radlat, radlon)
        lat_v, lon_v = xy2latlon(xcor_v, ycor_v, radlat, radlon)

        # plot
        #---------------------
        print('Plotting...')
        plt.figure(figcnt)
        figcnt = figcnt+1

        #x, y = m(lon, lat)
        #x_v, y_v = m(lon_v, lat_v)
        #rx, ry = m(radlon, radlat)

        # ZH plot
        #------------------------------
        ax1 = plt.subplot(2,2,1, projection=ccrs.PlateCarree())

        plt.pcolormesh(lon, lat, ref, cmap=zh_map, vmin=-10., vmax=80., transform=ccrs.PlateCarree())
        cb1 = plt.colorbar(fraction=0.04)
        cb1.set_label('(dBZ)', fontsize=cblb_fsize)
        cb1_la = [f'{ti:.0f}' for ti in cb1.get_ticks()]
        cb1.ax.set_yticklabels(cb1_la, fontsize=cbti_fsize)
        ax1.set_title('Z$\sf{_H}$', x=0.0, y=1.02, horizontalalignment='left',
                      fontsize=axtl_fsize)
        #ax1.add_feature(coast)
        #ax1.add_feature(states, edgecolor='k')
        ax1.add_geometries(shpreader.Reader(states_shp).geometries(),
                           ccrs.PlateCarree(), edgecolor='k',
                           facecolor='none', linewidth=0.5)
        ax1.add_feature(USCOUNTIES.with_scale('20m'), edgecolor='k', linewidth=0.2)
        ax1.set_extent([minlon,maxlon,minlat,maxlat])

        # ZDR plot
        #------------------------------
        ax2 = plt.subplot(2,2,2, projection=ccrs.PlateCarree())

        plt.pcolormesh(lon, lat, zdr, cmap=zdr_map, vmin=-2.4, vmax=6.9, transform=ccrs.PlateCarree())
        cb2 = plt.colorbar(fraction=0.04)
        cb2.set_label('(dB)', fontsize=cblb_fsize)
        cb2_la = [f'{ti:.1f}' for ti in cb2.get_ticks()]
        cb2.ax.set_yticklabels(cb2_la, fontsize=cbti_fsize)
        ax2.set_title('Z$_{\sf{DR}}$', x=0.0, y=1.02, horizontalalignment='left',
                      fontsize=axtl_fsize)

        ax2.add_geometries(shpreader.Reader(states_shp).geometries(),
                           ccrs.PlateCarree(), edgecolor='k',
                           facecolor='none', linewidth=0.5)
        ax2.add_feature(USCOUNTIES.with_scale('20m'), edgecolor='k', linewidth=0.2)
        ax2.set_extent([minlon,maxlon,minlat,maxlat])

        # KDP plot
        #------------------------------
        ax3 = plt.subplot(2,2,3, projection=ccrs.PlateCarree())
        plt.pcolormesh(lon, lat, kdp_alt, cmap=kdp_map, vmin=-1.6, vmax=4.3, transform=ccrs.PlateCarree())

        cb3 = plt.colorbar(fraction=0.04)
        cb3.set_label('(deg./km)', fontsize=cblb_fsize)
        cb3_la = [f'{ti:.1f}' for ti in cb3.get_ticks()]
        cb3.ax.set_yticklabels(cb3_la, fontsize=cbti_fsize)
        ax3.set_title('$\sf{K_{DP}}$', x=0.0, y=1.02, horizontalalignment='left',
                      fontsize=axtl_fsize)

        ax3.add_geometries(shpreader.Reader(states_shp).geometries(),
                           ccrs.PlateCarree(), edgecolor='k',
                           facecolor='none', linewidth=0.5)
        ax3.add_feature(USCOUNTIES.with_scale('20m'), edgecolor='k', linewidth=0.2)
        ax3.set_extent([minlon,maxlon,minlat,maxlat])

        # RhoHV plot
        #------------------------------
        ax4 = plt.subplot(2,2,4, projection=ccrs.PlateCarree())
        plt.pcolormesh(lon, lat, rhohv, cmap=phv_map, vmin=0.695, vmax=1.045, transform=ccrs.PlateCarree())
        cb4 = plt.colorbar(fraction=0.04)
        cb4.set_label('', fontsize=20)
        cb4_la = [f'{ti:.2f}' for ti in cb4.get_ticks()]
        cb4.ax.set_yticklabels(cb4_la, fontsize=cbti_fsize)
        ax4.set_title('$\\rho\sf{_{HV}}$', x=0.0, y=1.02, horizontalalignment='left',
                      fontsize=axtl_fsize)

        ax4.add_geometries(shpreader.Reader(states_shp).geometries(),
                           ccrs.PlateCarree(), edgecolor='k',
                           facecolor='none', linewidth=0.5)
        ax4.add_feature(USCOUNTIES.with_scale('20m'), edgecolor='k', linewidth=0.2)
        ax4.set_extent([minlon,maxlon,minlat,maxlat])

        # save image as .png
        #-------------------------------
        title = '{} - {}/{}/{} - {}:{} UTC - {:.1f} deg. PPI'.format(site, yyyy, mm, dd,
                                                                     hh, mn, float(angle))
        plt.suptitle(title, fontsize=24)
        plt.subplots_adjust(top=0.95, hspace=-0.1, wspace=0.2)
        imgname = yyyy+mm+dd+'_'+hh+mn+'_'+site.lower()+'.png'
        plt.savefig(imgname, format='png', dpi=120)
        plt.close()

        # crop out white space from figure
        os.system('convert -trim '+imgname+' '+imgname)
