import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import cosmotools

########################################################################
# Get dimensions of close-fitting cuboid for given range of (R.A.,     #
# Dec., redshift).                                                     #
########################################################################

def boxsize(zmin,zmax,rmin,rmax,dmin,dmax):
    rmin,rmax,dmin,dmax = np.radians([rmin,rmax,dmin,dmax])
    rcen = rmin + (rmax - rmin)/2; dcen = dmin + (dmax - dmin)/2
    zcen = zmin + (zmax - zmin)/2
    lx = (rmax - rmin) * np.cos(dcen) * cosmotools.D_com(zcen)
    ly = (dmax - dmin) * cosmotools.D_com(zcen)
    lz = cosmotools.D_com(zmax) - cosmotools.D_com(zmin)
    return lx,ly,lz

def AngularDistance(r1,r2,d1,d2):
    return np.arccos( np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2) )

########################################################################
# Estimate the 3D power spectrum of a density field.                    #
########################################################################

def getpkspec(datgrid1,datgrid2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin):
    '''
    If auto-correlating then datgrid1 should equal datgrid2
    '''
    w = np.ones((nx,ny,nz)) #Give all a constant weighting
    w = w / np.sum(w) #normalise weighting so they integrate to 1
    datgrid1,datgrid2 = datgrid1*w,datgrid2*w
    fgrid1 = np.fft.rfftn(datgrid1)
    fgrid2 = np.fft.rfftn(datgrid2)
    pkspec = np.real( fgrid1 * np.conj(fgrid2) )
    vol_box = lx*ly*lz
    nc = nx*ny*nz
    vol_cell = vol_box / nc
    pkspec = vol_cell * pkspec / np.sum(w**2) #L.Wolz method: https://arxiv.org/pdf/1510.05453.pdf
    return pkspec

########################################################################
# Compute correction due to alias effect (pixelisation onto grid)      #
########################################################################

def W_alias(nx,ny,nz,lx,ly,lz,p):
    '''
    Following Jing et al compute correction based on mass assignment function
    where p=1 (for NGP), p=2 (for CIC) or p=3 (for TSC). Divide measured power
    through by W^2 to complete correction
    '''
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)[:,np.newaxis,np.newaxis]
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)[np.newaxis,:,np.newaxis]
    kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1][np.newaxis,np.newaxis,:]
    kx[kx==0]=1e-30; ky[ky==0]=1e-30; kz[kz==0]=1e-30 #Amend to avoid divide by zeros
    return (np.sin(kx*lx/(2*nx))/(kx*lx/(2*nx)) * np.sin(ky*ly/(2*ny))/(ky*ly/(2*ny)) * np.sin(kz*lz/(2*nz))/(kz*lz/(2*nz)))**p

########################################################################
# Bin 3D power spectrum in angle-averaged bins.                        #
########################################################################

def binpk(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin):
    kspec,muspec,indep = getkspec(nx,ny,nz,lx,ly,lz)
    pkspec = pkspec[indep == True]
    kspec = kspec[indep == True]
    ikbin = np.digitize(kspec,np.linspace(kmin,kmax,nkbin+1))
    nmodes,pk = np.zeros(nkbin,dtype=int),np.zeros(nkbin)
    for ik in range(nkbin):
        nmodes[ik] = int(np.sum(np.array([ikbin == ik+1])))
        if (nmodes[ik] > 0):
            pk[ik] = np.mean(pkspec[ikbin == ik+1])
    return pk

########################################################################
# Bin 3D power spectrum in angle-averaged bins, weighting by Legendre  #
# polynomials.                                                         #
########################################################################

def binpole(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin):
    kspec,muspec,indep = getkspec(nx,ny,nz,lx,ly,lz)
    pkspec = pkspec[indep == True]
    kspec = kspec[indep == True]
    muspec = muspec[indep == True]
    leg2spec = ((3*(muspec**2))-1)/2
    leg4spec = ((35*(muspec**4))-(30*(muspec**2))+3)/8
    ikbin = np.digitize(kspec,np.linspace(kmin,kmax,nkbin+1))
    nmodes,pk0,pk2,pk4 = np.zeros(nkbin,dtype=int),np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin)
    for ik in range(nkbin):
        nmodes[ik] = int(np.sum(np.array([ikbin == ik+1])))
        if (nmodes[ik] > 0):
            pk0[ik] = np.mean(pkspec[ikbin == ik+1])
            pk2[ik] = 5*np.mean((pkspec*leg2spec)[ikbin == ik+1])
            pk4[ik] = 9*np.mean((pkspec*leg4spec)[ikbin == ik+1])
    return [pk0,pk2,pk4],nmodes

########################################################################
# Obtain 3D grid of k-modes.                                           #
########################################################################

def getkspec(nx,ny,nz,lx,ly,lz):
    kx = 2*np.pi*np.fft.fftfreq(nx,d=lx/nx)
    ky = 2*np.pi*np.fft.fftfreq(ny,d=ly/ny)
    kz = 2*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:int(nz/2)+1]
    indep = getindep(nx,ny,nz)
    indep[0,0,0] = False
    kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
    kspec[0,0,0] = 1.
    muspec = np.absolute(kz[np.newaxis,np.newaxis,:])/kspec
    kspec[0,0,0] = 0
    return kspec,muspec,indep

########################################################################
# Obtain array of independent 3D modes.                                #
########################################################################

def getindep(nx,ny,nz):
    indep = np.full((nx,ny,int(nz/2)+1),False,dtype=bool)
    indep[:,:,1:int(nz/2)] = True
    indep[1:int(nx/2),:,0] = True
    indep[1:int(nx/2),:,int(nz/2)] = True
    indep[0,1:int(ny/2),0] = True
    indep[0,1:int(ny/2),int(nz/2)] = True
    indep[int(nx/2),1:int(ny/2),0] = True
    indep[int(nx/2),1:int(ny/2),int(nz/2)] = True
    indep[int(nx/2),0,0] = True
    indep[0,int(ny/2),0] = True
    indep[int(nx/2),int(ny/2),0] = True
    indep[0,0,int(nz/2)] = True
    indep[int(nx/2),0,int(nz/2)] = True
    indep[0,int(ny/2),int(nz/2)] = True
    indep[int(nx/2),int(ny/2),int(nz/2)] = True
    return indep

########################################################################
# Convert power spectrum multipoles P_l(k) to P(k,mu) or P(kpar,kperp).#
########################################################################

def pkpoletopkmu(nmu,pkmults):
    nkbin = len(pkmults[0])
    dmu = 1/nmu
    pkmuobs = np.zeros((nkbin,nmu))
    for imu in range(nmu):
        mu1 = dmu*imu
        mu2 = dmu*(imu+1)
        obs = np.zeros(nkbin)
        for i in range(3):
            coeff = quad(pkmuint,mu1,mu2,args=(i*2))[0]
            obs[:] += (coeff/dmu)*pkmults[i]
        pkmuobs[:,imu] = obs
    return pkmuobs

def pkmuint(mu,l):
    return getleg(l,mu)

def getleg(l,mu):
    if (l == 2):
        leg = (3*(mu**2)-1)/2
    elif (l == 4):
        leg = (35*(mu**4)-30*(mu**2)+3)/8
    else:
        leg = 1
    return leg

def pkmutopk2(kmin2,kmax2,nk2,kmin,kmax,nk,nmu,pkmu):
    # Generate random points across (kperp,kpar) space
    nran = 1000000
    rkperp = kmin2 + (kmax2-kmin2)*np.random.rand(nran)
    rkpar = kmin2 + (kmax2-kmin2)*np.random.rand(nran)
    # Convert these points to (k,mu) values
    rk = np.sqrt(rkperp**2 + rkpar**2)
    rmu = rkpar/rk
    # Bin these points in (k,mu) bins
    klims = np.concatenate((np.linspace(kmin,kmax,nk+1),np.array([1.])))
    ikbin = np.digitize(rk,klims) - 1
    mulims = np.linspace(0.,1.,nmu+1)
    mulims[0],mulims[nmu] = -0.01,1.01
    imubin = np.digitize(rmu,mulims) - 1
    # Power spectrum values of these points
    pkmuobs1 = np.zeros((nk+1,nmu))
    pkmuobs1[:nk,:] = pkmu
    rpkobs = pkmuobs1[ikbin,imubin]
    # Count points in (k,mu) bins
    pkmucount,edges = np.histogramdd(np.vstack([ikbin+0.5,imubin+0.5]).transpose(),bins=(nk,nmu),range=((0,nk),(0,nmu)))
    # Count points in (kperp,kpar) bins
    pk2count,edges = np.histogramdd(np.vstack([rkperp,rkpar]).transpose(),bins=(nk2,nk2),range=((kmin2,kmax2),(kmin2,kmax2)))
    # Bin these points in (kperp,kpar) bins
    pk2obs,edges = np.histogramdd(np.vstack([rkperp,rkpar]).transpose(),bins=(nk2,nk2),range=((kmin2,kmax2),(kmin2,kmax2)),normed=False,weights=rpkobs)
    pk2obs = pk2obs/pk2count
    return pk2obs
