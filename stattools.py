import numpy as np
from scipy import integrate
import pktools

def ChiSquare(x_obs,x_mod,x_err):
    '''
    Reduced Chi-squared function
    '''
    return np.sum( ((x_obs-x_mod)/x_err)**2 )

def MultipoleError(P_N,pk0,pk2,pk4,nmodes):
    '''
    Theoretical error for Pk multipoles
    '''
    nkbin = len(pk0)
    pk0err,pk2err,pk4err = np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin)
    for ik in range(nkbin):
        if (nmodes[ik] > 0):
            int0 = integrate.quad(pkvarint,0,1,args=(0,pk0[ik],pk2[ik],pk4[ik],P_N))[0]
            int2 = integrate.quad(pkvarint,0,1,args=(2,pk0[ik],pk2[ik],pk4[ik],P_N))[0]
            int4 = integrate.quad(pkvarint,0,1,args=(4,pk0[ik],pk2[ik],pk4[ik],P_N))[0]
            pk0err[ik] = 1*np.sqrt(int0/nmodes[ik])
            pk2err[ik] = 5*np.sqrt(int2/nmodes[ik])
            pk4err[ik] = 9*np.sqrt(int4/nmodes[ik])
    return pk0err,pk2err,pk4err

def WedgesError(nwedge,pk0,pk2,pk4,sig_pl,nmodes):
    '''
    Theoretical error for Pk clustering wedges
    '''
    nkbin = len(pk0)
    dmu = 1/nwedge
    pkmuerr = np.zeros((nwedge,nkbin))
    for imu in range(nwedge):
        mu1 = dmu*imu
        mu2 = dmu*(imu+1)
        var = np.zeros(nkbin)
        for i in range(3):
            coeff = integrate.quad(pktools.pkmuint,mu1,mu2,args=(i*2))[0]
            var += ((coeff/dmu)**2)*(sig_pl[i]**2)
        pkmuerr[imu] = np.sqrt(var)
    return pkmuerr

def pkvarint(mu,l,pk0,pk2,pk4,P_N):
    '''
    Integral to obtain Pk-variance
    '''
    leg2 = (3*(mu**2)-1)/2
    leg4 = (35*(mu**4)-30*(mu**2)+3)/8
    pk = pk0 + pk2*leg2 + pk4*leg4
    pkvar = (pk + P_N)**2
    if (l == 0):
        return pkvar
    elif (l == 2):
        return (leg2**2)*pkvar
    elif (l == 4):
        return (leg4**2)*pkvar
