import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from mpl_toolkits.basemap import Basemap
import numpy as np
from pyart.io.nexrad_archive import read_nexrad_archive
from numpy import genfromtxt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import convolve1d
import os

# function to do convolution (along range dimension)
#----------------------------------
def conv(x, w):
    numk = len(w)
    numx = x.shape[1]
    y = np.ma.masked_all([x.shape[0], numx])

    for i in range(numx-numk):
        numvalid = x[:,i:i+numk].count(axis=1)
        y[numvalid==numk,i+(numk-1)/2] = np.ma.dot(x[numvalid==numk,i:i+numk], w)

    return y

# function to de-alias phidp
#----------------------------------
def dealiasPhiDP(phiDP):
    deal_phi = np.ma.empty([phiDP.shape[0], phiDP.shape[1]])
    deal_phi[phiDP<0.] = 180.+phiDP[phiDP<0.] 
    deal_phi[phiDP>=0.] = phiDP[phiDP>=0.]
    return deal_phi   

# function for smoothing phidp
#----------------------------------
def smPhiDP(phiDP, ran):
    # smooth phiDP field and take derivative
    # calculate lanczos filter weights
    numRan = ran.shape[0]
    numK = 31
    fc = 0.015
    kt = np.linspace(-(numK-1)/2, (numK-1)/2, numK)
    w = np.sinc(2.*kt*fc)*(2.*fc)*np.sinc(kt/(numK/2))

    #smoothPhiDP = convolve1d(phiDP, w, axis=1, mode='constant', cval=-999.)
    smoothPhiDP = conv(phiDP, w)
    #smoothPhiDP = np.ma.masked_where(smoothPhiDP==-999., smoothPhiDP)

    return smoothPhiDP

# function for estimating kdp
#----------------------------------
def calculateKDP(phiDP, ran):
    # smooth phiDP field and take derivative
    numRan = ran.shape[0]
    kdp = np.ma.masked_all(phiDP.shape)
    smoothPhiDP = smPhiDP(phiDP, ran)

    # take derivative of kdp field
    winLen = 11
    rprof = ran[0:winLen*2-1]/1000.

    for i in range(numRan-winLen*3):
        numvalid = smoothPhiDP[:,i:i+winLen*2-1].count(axis=1)
        max_numv = np.max(numvalid)
        if max_numv==(winLen*2-1):
            kdp[numvalid==(winLen*2-1),i+winLen] = 0.5*np.polyfit(rprof,
                smoothPhiDP[numvalid==(winLen*2-1),i:i+winLen*2-1].transpose(), 1)[0]
    return kdp

# function for creating color map
#----------------------------------
def createCmap(mapname):
    fil = open(mapname+'.rgb')
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

# open file
#-----------------------------------
print 'Opening file...'

yyyy = '2016'
mm = '10'
dd = '07'

hh = '12'
mn = '29'
ss = '01'
site = 'KMLB'

filename = '{}{}{}{}_{}{}{}_V06'.format(site, yyyy, mm, dd,
                                        hh, mn, ss)
rad = read_nexrad_archive(filename)
rad_sw = rad.extract_sweeps([0])

# get variables
#-----------------------------------
elev_p = rad_sw.elevation['data']
azi_p = 90.-rad_sw.azimuth['data']
ran = rad_sw.range['data']
ref_p = rad_sw.fields['reflectivity']['data']
zdr_p = rad_sw.fields['differential_reflectivity']['data']
phidp_p = rad_sw.fields['differential_phase']['data']
rhohv_p = rad_sw.fields['cross_correlation_ratio']['data']
radlat = rad_sw.latitude['data'][0]
radlon = rad_sw.longitude['data'][0]

dims = ref_p.shape
numradials = dims[0]+1
numgates = dims[1]

# expand radially to remove no data spike
elev = np.ma.empty([numradials])
azi = np.ma.empty([numradials])
ref = np.ma.empty([numradials, numgates])
zdr = np.ma.empty([numradials, numgates])
phidp = np.ma.empty([numradials, numgates])
rhohv = np.ma.empty([numradials, numgates])

elev[0:numradials-1] = elev_p
elev[numradials-1] = elev_p[0]
azi[0:numradials-1] = azi_p
azi[numradials-1] = azi_p[0]
ref[0:numradials-1,:] = ref_p
ref[numradials-1,:] = ref_p[0]
zdr[0:numradials-1,:] = zdr_p
zdr[numradials-1,:] = zdr_p[0]
phidp[0:numradials-1,:] = phidp_p
phidp[numradials-1,:] = phidp_p[0]
rhohv[0:numradials-1,:] = rhohv_p
rhohv[numradials-1,:] = rhohv_p[0]

angle = np.mean(elev)

# mask data by rhohv and threshold
#-----------------------------------------------
ref = np.ma.masked_where(rhohv<0.4, ref)
zdr = np.ma.masked_where(rhohv<0.4, zdr)
phidp = np.ma.masked_where(rhohv<0.4, phidp)
rhohv = np.ma.masked_where(rhohv<0.4, rhohv)

zdr = np.ma.masked_where(ref<-5., zdr)
phidp = np.ma.masked_where(ref<-5., phidp)
rhohv = np.ma.masked_where(ref<-5., rhohv)
ref = np.ma.masked_where(ref<-5., ref)

# calculate kdp
#-----------------------------------------------
print 'Calculating KDP...'
phidp = dealiasPhiDP(phidp)
kdp = calculateKDP(phidp, ran)
kdp = np.ma.masked_where(ref<-5., kdp)

# calculate x and y coordinates (wrt beampath) for plotting
#-----------------------------------------------------------
ran_2d = np.tile(ran,(numradials,1))
azi.shape = (azi.shape[0], 1)
azi_2d = np.tile(azi,(1,numgates))

radz = 10.
erad = np.pi*angle/180.

ke = 4./3.
a = 6378137.

# beam height and beam distance
zcor = np.sqrt(ran_2d**2.+(ke*a)**2.+2.*ran_2d*ke*a*np.sin(erad))-ke*a+radz
scor = ke*a*np.arcsin(ran_2d*np.cos(erad)/(ke*a+zcor))/1000.

xcor = scor*np.cos(np.pi*azi_2d/180.)
ycor = scor*np.sin(np.pi*azi_2d/180.)

# plot with actual range as coordinate
#xcor = ran_2d*np.cos(np.pi*azi_2d/180.)/1000.
#ycor = ran_2d*np.sin(np.pi*azi_2d/180.)/1000.

# convert to lon,lat for basemap plotting
lat, lon = xy2latlon(xcor, ycor, radlat, radlon)

# plot
#---------------------
print 'Plotting...'

# change fonts
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('text', usetex=True)

zh_map = createCmap('zh2_map')
zdr_map = createCmap('zdr_map')
vel_map = createCmap('vel2_map')
phv_map = createCmap('phv_map')
kdp_map = createCmap('kdp_map')

ds = 300.
xcen = 0.
ycen = 0.

# get lat lon corners of data domain
minlat, minlon = xy2latlon(xcen-ds, ycen-ds, radlat, radlon)
maxlat, maxlon = xy2latlon(xcen+ds, ycen+ds, radlat, radlon)

xlabel = 'X-distance (km)'
ylabel = 'Y-distance (km)'

# create custom number of color levels for plots
#-----------------------------------------------
numcolors = 33
numlevs = numcolors-1
cinds = np.linspace(0., 254., numcolors).astype(int)

zh_cols = zh_map.colors[cinds]
zdr_cols = zdr_map.colors[cinds]
vel_cols = vel_map.colors[cinds]
phv_cols = phv_map.colors[cinds]
kdp_cols = kdp_map.colors[cinds]

zh_levs = np.linspace(10., 72., numlevs)
zdr_levs = np.linspace(-2.4, 6.9, numlevs)
phidp_levs = np.linspace(60., 122., numlevs)
phv_levs = np.linspace(0., 1.0, numlevs)**(1./2.)*1.05
kdp_levs = np.linspace(-1.6, 4.6, numlevs)

zh_mapn, zh_norm = cm.from_levels_and_colors(zh_levs, zh_cols, extend='both')
zdr_mapn, zdr_norm = cm.from_levels_and_colors(zdr_levs, zdr_cols, extend='both')
phidp_mapn, phidp_norm = cm.from_levels_and_colors(phidp_levs, zh_cols, extend='both')
phv_mapn, phv_norm = cm.from_levels_and_colors(phv_levs, phv_cols, extend='both')
kdp_mapn, kdp_norm = cm.from_levels_and_colors(kdp_levs, kdp_cols, extend='both')

# smooth zdr for contour plotting
sig = 2.5
sm_zdr = gaussian_filter(zdr, sig)
zdr_lev = 1.2

fig = plt.figure()
# set common font sizes
cblb_fsize = 10
cbti_fsize = 8
axtl_fsize = 12
axlb_fsize = 12
axti_fsize = 10

# create basemap
m = Basemap(llcrnrlon=minlon, llcrnrlat=minlat, urcrnrlon=maxlon, urcrnrlat=maxlat,
            projection='lcc', lat_1=15., lat_2=35., lon_0=-80.,
            resolution='h', area_thresh=1000.)
x, y = m(lon, lat)
rx, ry = m(radlon, radlat)

# ZH plot
#------------------------------
ax1 = fig.add_subplot(2,2,1)
#plt.contour(xcor, ycor, sm_zdr, levels=[zdr_lev], colors=('k',), linewidths=(1.3,))
m.pcolormesh(x, y, ref, cmap=zh_mapn, norm=zh_norm)
cb1 = plt.colorbar()
cb1.set_label('(dBZ)', fontsize=cblb_fsize)
cb1_la = [ti.get_text().replace('$', '') for ti in cb1.ax.get_yticklabels()]
cb1.ax.set_yticklabels(cb1_la, fontsize=cbti_fsize)
ax1.set_title('a) Z$\sf{_H}$', x=0.0, y=1.02, horizontalalignment='left',
              fontsize=axtl_fsize)
#ax1.tick_params(axis='both', which='both', labelbottom='off', bottom='off',
#                labelleft='off', left='off', top='off', right='off')
m.drawcoastlines(0.5)
m.drawstates(linewidth=0.5)
m.plot(rx, ry, 'bo', markersize=8)
#m.readshapefile('c_04jn14', 'counties')

# ZDR plot
#------------------------------
ax2 = fig.add_subplot(2,2,2)
#plt.contour(xcor, ycor, sm_zdr, levels=[zdr_lev], colors=('k',), linewidths=(1.3,))
m.pcolormesh(x, y, zdr, cmap=zdr_mapn, norm=zdr_norm)
cb2 = plt.colorbar()
cb2.set_label('(dB)', fontsize=cblb_fsize)
cb2_la = [ti.get_text().replace('$', '') for ti in cb2.ax.get_yticklabels()]
cb2.ax.set_yticklabels(cb2_la, fontsize=cbti_fsize)
ax2.set_title('b) Z$\sf{_{DR}}$', x=0.0, y=1.02, horizontalalignment='left',
              fontsize=axtl_fsize)
m.drawcoastlines(0.5)
m.drawstates(linewidth=0.5)
m.plot(rx, ry, 'bo', markersize=8)

'''
# PhiDP plot
#------------------------------
ax3 = fig.add_subplot(2,2,3)
#plt.contour(xcor, ycor, sm_zdr, levels=[zdr_lev], colors=('k',), linewidths=(1.3,))
plt.pcolormesh(xcor, ycor, phidp, cmap=phidp_mapn, norm=phidp_norm)
cb3 = plt.colorbar()
cb3.set_label('(deg.)', fontsize=cblb_fsize)
cb3_la = [ti.get_text().replace('$', '') for ti in cb3.ax.get_yticklabels()]
cb3.ax.set_yticklabels(cb3_la, fontsize=cbti_fsize)
ax3.set_xlim([-ds, ds])
ax3.set_ylim([-ds, ds])
ax3.set_title('c) $\sf{\psi_{DP}}$', x=0.0, y=1.02, horizontalalignment='left',
              fontsize=axtl_fsize)
ax3.set_xlabel(xlabel, fontsize=axlb_fsize)
ax3.set_ylabel(ylabel, fontsize=axlb_fsize)
# change tick mark sizes and fonts
#ax3.set_xticklabels(ax3.get_xticks())
#ax3.set_yticklabels(ax3.get_yticks())
#ax3.tick_params(axis='both', which='major', labelsize=axti_fsize, pad=10)
ax3.tick_params(axis='both', which='both', labelbottom='off', bottom='off',
                labelleft='off', left='off', top='off', right='off')
'''
# KDP plot
#------------------------------
ax3 = fig.add_subplot(2,2,3)
#plt.contour(xcor, ycor, sm_zdr, levels=[zdr_lev], colors=('k',), linewidths=(1.3,))
m.pcolormesh(x, y, kdp, cmap=kdp_mapn, norm=kdp_norm)
cb3 = plt.colorbar()
cb3.set_label('(deg./km)', fontsize=cblb_fsize)
cb3_la = [ti.get_text().replace('$', '') for ti in cb3.ax.get_yticklabels()]
cb3.ax.set_yticklabels(cb3_la, fontsize=cbti_fsize)
ax3.set_title('c) $\sf{K_{DP}}$', x=0.0, y=1.02, horizontalalignment='left',
              fontsize=axtl_fsize)
m.drawcoastlines(0.5)
m.drawstates(linewidth=0.5)
m.plot(rx, ry, 'bo', markersize=8)

# RhoHV plot
#------------------------------
ax4 = fig.add_subplot(2,2,4)
#plt.contour(xcor, ycor, sm_zdr, levels=[zdr_lev], colors=('k',), linewidths=(1.3,))
m.pcolormesh(x, y, rhohv, cmap=phv_mapn, norm=phv_norm)
cb4 = plt.colorbar()
cb4.set_label('', fontsize=20)
cb4_la = ['{:.2f}'.format(float(ti.get_text().replace('$', ''))) for ti in cb4.ax.get_yticklabels()]
cb4.ax.set_yticklabels(cb4_la, fontsize=cbti_fsize)
ax4.set_title('d) $\\rho\sf{_{HV}}$', x=0.0, y=1.02, horizontalalignment='left',
              fontsize=axtl_fsize)
m.drawcoastlines(0.5)
m.drawstates(linewidth=0.5)
m.plot(rx, ry, 'bo', markersize=8)

# save image as .png
#-------------------------------
title = '{} - {}/{}/{} - {}:{} UTC - {:.1f} deg. PPI'.format(site, yyyy, mm, dd,
                                                                   hh, mn, float(angle))
plt.suptitle(title, fontsize=16)
plt.subplots_adjust(top=0.89, hspace=0.15, wspace=0.)
imgname = yyyy+mm+dd+'_'+hh+mn+'_'+site.lower()+'.png'
plt.savefig(imgname, format='png', dpi=250)

# crop out white space from figure
os.system('convert -trim '+imgname+' '+imgname)
