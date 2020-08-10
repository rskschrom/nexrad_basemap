import numpy as np
from scipy.optimize import fsolve

# beam function to solve
def beam_func(range_m, gdis_m, elev_radians, rad_alt_m):
    ke = 4./3.
    a = 6378137.
    fx = gdis_m-ke*a*np.arcsin(range_m*np.cos(elev_radians)/
         (rad_alt_m+np.sqrt(range_m**2.+(ke*a)**2.+2.*range_m*ke*a*np.sin(elev_radians))))
    return fx

# radar beam height and ground distance (Doviak and Zrnic 1993)
def beam_height_dz(elev_radians, rad_alt_m, range_m):
    # 4/3 earth approximation
    ke = 4./3.
    a = 6378137.

    # beam height and beam distance (meters)
    bhgt = np.sqrt(range_m**2.+(ke*a)**2.+2.*range_m*ke*a*np.sin(elev_radians))-ke*a+rad_alt_m
    gdis = ke*a*np.arcsin(range_m*np.cos(elev_radians)/(ke*a+bhgt))
    return bhgt, gdis

# get altitude of beam given ground distance and beam elevation angle
def beam_alt(elev_radians, rad_alt_m, gdis_m):
    range_sol = fsolve(beam_func, gdis_m, args=(gdis_m, elev_radians, rad_alt_m))
    bhgt,_ = beam_height_dz(elev_radians, rad_alt_m, range_sol)
    return bhgt

