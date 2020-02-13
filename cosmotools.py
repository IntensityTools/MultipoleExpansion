import numpy as np
import scipy
from scipy import integrate
import astropy
from astropy import constants
c = astropy.constants.c.value #299,792,458 m/s
c_km = c/1e3 # km per second
#from nbodykit.lab import cosmology

def SetCosmology(builtincosmo):
    global cosmo
    global NBkcosmo
    global H_0
    global h
    if builtincosmo=='WMAP5':
        from astropy.cosmology import WMAP5 as cosmo
        #NBkcosmo = cosmology.WMAP5
    if builtincosmo=='Planck15':
        from astropy.cosmology import Planck15 as cosmo
        #NBkcosmo = cosmology.Planck15
    H_0 = cosmo.H(0).value
    h = H_0/100
    print('\nSelected Cosmology:',cosmo,'\n')
    return

def getNBKcosmo():
    return NBkcosmo

def f(z):
    gamma = 0.545
    return cosmo.Om(z)**gamma

def E(z):
    return np.sqrt( 1 - Om0 + Om0*(1+z)**3 )

def H(z):
    return cosmo.H(z).value

def Omega_M(z):
    return cosmo.Om(z)

def D_com(z):
    #Comoving distance [Mpc/h]
    h = H_0/100
    return cosmo.comoving_distance(z).value * h
