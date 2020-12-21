import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import scipy.interpolate as inp
from scipy.stats import binned_statistic_2d
import glob
import datetime

# function for creating color map
def createCmap(mapname):
    fil = open(mapname+'.rgb')
    cdata = np.genfromtxt(fil,skip_header=2)
    cdata = cdata/256
    cmap = cm.ListedColormap(cdata, mapname)
    fil.close()
    return cmap


# format for time axis
def timeticks(x, pos):
    dt = datetime.datetime.utcfromtimestamp(x) 
    hour = dt.hour
    minute = dt.minute
    return f'{hour:02d}:{minute:02d}'

# read in qvp data from files
site = 'KBGM'
date_pre = '202001'
qvp_files = glob.glob(f'{date_pre}*{site}.txt')
print(qvp_files)
nqvp = len(qvp_files)
times = [None]*nqvp
zh = [None]*nqvp
zdr = [None]*nqvp
kdp = [None]*nqvp
rhohv = [None]*nqvp
zcor = [None]*nqvp

for i in range(nqvp):
    with open(qvp_files[i]) as f:
        first_line = f.readline()
    times[i] = int(first_line.split(':')[1])

    data = np.genfromtxt(qvp_files[i], skip_header=2)
    zcor[i] = data[:,0]
    zh[i] = data[:,1]
    zdr[i] = data[:,2]
    kdp[i] = data[:,3]
    rhohv[i] = data[:,4]

# make data 1d numpy arrays
zc1d = np.array([])
zh1d = np.array([])
zdr1d = np.array([])
kdp1d = np.array([])
rhv1d = np.array([])
t1d = np.array([])

for i in range(nqvp):
    zc1d = np.concatenate((zc1d, zcor[i]))
    zh1d = np.concatenate((zh1d, zh[i]))
    zdr1d = np.concatenate((zdr1d, zdr[i]))
    kdp1d = np.concatenate((kdp1d, kdp[i]))
    rhv1d = np.concatenate((rhv1d, rhohv[i]))
    t1d = np.concatenate((t1d, np.full((len(zcor[i])), times[i])))

# mask missing data
zh1d = np.ma.masked_where(zh1d==-999., zh1d)
zdr1d = np.ma.masked_where(zdr1d==-999., zdr1d)
kdp1d = np.ma.masked_where(kdp1d==-999., kdp1d)
rhv1d = np.ma.masked_where(rhv1d==-999., rhv1d)
#sw_time = datetime.fromtimestamp(ts_utc)

kdp1d = np.ma.masked_where(zc1d<1000., kdp1d)

#  create time-height grid
delta_t_sec = 600.
delta_z_m = 30.

ntgrid = int((np.max(t1d)-np.min(t1d))/(delta_t_sec))+1
nzgrid = int(15000./delta_z_m+1.)
tg1d = np.arange(ntgrid)*delta_t_sec+np.min(t1d)
zg1d = np.arange(nzgrid)*delta_z_m
tgrid, zgrid = np.meshgrid(tg1d, zg1d, indexing='ij')

# bin data
zhgrid,_,_,_ = binned_statistic_2d(t1d[~zh1d.mask], zc1d[~zh1d.mask], zh1d[~zh1d.mask],
                                   statistic='median', bins=(tg1d, zg1d))
zdrgrid,_,_,_ = binned_statistic_2d(t1d[~zdr1d.mask], zc1d[~zdr1d.mask], zdr1d[~zdr1d.mask],
                                   statistic='median', bins=(tg1d, zg1d))
kdpgrid,_,_,_ = binned_statistic_2d(t1d[~kdp1d.mask], zc1d[~kdp1d.mask], kdp1d[~kdp1d.mask],
                                   statistic='median', bins=(tg1d, zg1d))
rhvgrid,_,_,_ = binned_statistic_2d(t1d[~rhv1d.mask], zc1d[~rhv1d.mask], rhv1d[~rhv1d.mask],
                                   statistic='median', bins=(tg1d, zg1d))

zdrgrid = np.ma.masked_where(zhgrid<0., zdrgrid)
kdpgrid = np.ma.masked_where(zhgrid<0., kdpgrid)
rhvgrid = np.ma.masked_where(zhgrid<0., rhvgrid)
zhgrid = np.ma.masked_where(zhgrid<0., zhgrid)

# plot
fmtx = mpl.ticker.FuncFormatter(timeticks)
fmty = mpl.ticker.StrMethodFormatter("{x:.0f}")
fig = plt.figure(figsize=(12,16))
zmax = 10.

# color map stuff
zh_map = createCmap('zh2_map')
zdr_map = createCmap('zdr_map')
rhv_map = createCmap('phv_map')
kdp_map = createCmap('kdp_map')

# zh
ax = fig.add_subplot(4,1,1)
plt.pcolormesh(tgrid, zgrid/1.e3, zhgrid, cmap=zh_map, vmin=0., vmax=60.)
cb = plt.colorbar()
cb.set_label('(dBZ)')
ax.set_ylim([0., zmax])
ax.xaxis.set_major_formatter(fmtx)
ax.yaxis.set_major_formatter(fmty)
ax.set_ylabel('Height (km)', fontsize=16)
ax.set_title('$Z_{H}$', fontsize=24, x=0., y=1.02, ha='left')

# zdr
ax = fig.add_subplot(4,1,2)
plt.pcolormesh(tgrid, zgrid/1.e3, zdrgrid, cmap=zdr_map, vmin=-2.4, vmax=6.9)
cb = plt.colorbar()
cb.set_label('(dB)')
ax.set_ylim([0., zmax])
ax.xaxis.set_major_formatter(fmtx)
ax.yaxis.set_major_formatter(fmty)
ax.set_ylabel('Height (km)', fontsize=16)
ax.set_title('$Z_{DR}$', fontsize=24, x=0., y=1.02, ha='left')

# kdp
ax = fig.add_subplot(4,1,3)
plt.pcolormesh(tgrid, zgrid/1.e3, kdpgrid, cmap=kdp_map, vmin=-0.5, vmax=2.)
cb = plt.colorbar()
cb.set_label('(deg/km)')
ax.set_ylim([0., zmax])
ax.xaxis.set_major_formatter(fmtx)
ax.yaxis.set_major_formatter(fmty)
ax.set_ylabel('Height (km)', fontsize=16)
ax.set_title('$K_{DP}$', fontsize=24, x=0., y=1.02, ha='left')

# rhohv
ax = fig.add_subplot(4,1,4)
plt.pcolormesh(tgrid, zgrid/1.e3, rhvgrid, cmap=rhv_map, vmin=0.81, vmax=1.03)
cb = plt.colorbar()
cb.set_label('')
ax.set_ylim([0., zmax])
ax.xaxis.set_major_formatter(fmtx)
ax.yaxis.set_major_formatter(fmty)
ax.set_xlabel('Time (UTC)', fontsize=16)
ax.set_ylabel('Height (km)', fontsize=16)
ax.set_title('$\\rho_{HV}$', fontsize=24, x=0., y=1.02, ha='left')

# make plot + title
plt.subplots_adjust(hspace=0.4)

# get date or date range
times = np.array(times)
dtmin = datetime.datetime.utcfromtimestamp(np.min(times))
dtmax = datetime.datetime.utcfromtimestamp(np.max(times))
year_min = dtmin.year
year_max = dtmax.year
month_min = dtmin.strftime("%B")
month_max = dtmax.strftime("%B")
day_min = dtmin.day
day_max = dtmax.day

date_form = f'{day_min} {month_min} {year_min}'

if day_min!=day_max:
    date_form = f'{day_min}\N{MINUS SIGN}{day_max} {month_min} {year_min}'
    if dtmin.month!=dtmax.month:
        date_form = f'{day_min} {month_min}\N{MINUS SIGN}{day_max} {month_max} {year_min}'
        if dtmin.month!=dtmax.month:
            date_form = f'{day_min} {month_min} {year_min}\N{MINUS SIGN}{day_max} {month_max} {year_max}'

plt.suptitle(f'{date_form} - {site} QVP '+'(10$^{\circ}$)', fontsize=26, y=0.94)
plt.savefig(f'time_height_{site}.png', bbox_inches='tight')
