import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
import matplotlib.pyplot as plt
import mpl_style
plt.style.use(mpl_style.style1)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import cosmotools
cosmotools.SetCosmology('Planck15')
import pktools
import teletools

# Dimension of data cube:
lx,ly,lz = 1000,1000,1000 #Mpc/h
nx,ny,nz = 225,225,225
lzbuff = 119 #Used to achieve a lz=762Mpc/h which is ~Deltaz=0.4 at zeff=0.8
xbins = np.linspace(0,lx,nx+1) #cartesian-z pixel bins
ybins = np.linspace(0,ly,ny+1) #cartesian-z pixel bins
lz = lz - 2*lzbuff

# Cosmological & Survey Parameters:
zeff = 0.82 #Redshift for simulation boz
#zeff = 2.03 #Redshift for simulation boz
d_c = cosmotools.D_com(zeff) #comoving distance to effective redshift
theta_FWHM,R_beam = teletools.getbeampars(zeff,d_c)

#####################################################
# Read-in and Process HI and FG maps                #
#####################################################

dT_HI_nobeam = np.load('MultiDarkSims/dT_HI-MDSAGE_z_%s.npy'%zeff)
dT_HI = np.zeros(np.shape(dT_HI_nobeam))
dT_FG = np.load('MultiDarkSims/dT_FG-MDSAGE_z_%s.npy'%zeff)
dT_obs_nobeam = dT_HI_nobeam + dT_FG
dT_obs = np.zeros(np.shape(dT_obs_nobeam))
xbincentres = xbins+(xbins[1]-xbins[0])/2
xbincentres = xbincentres[:len(xbincentres)-1] #remove last value since this is outside of bins
ybincentres = ybins+(ybins[1]-ybins[0])/2
ybincentres = ybincentres[:len(ybincentres)-1] #remove last value since this is outside of bins
#Construct a quasi-redshift range based on lz (z-length) of box for noise map:
d_c = cosmotools.D_com(zeff) #distance to box
ztests = np.linspace(0,10,100)         #Build spline to obtain redshifts
d_c_spline = cosmotools.D_com(ztests)  #   based on lz com-distances
zspline = interp1d(d_c_spline, ztests , kind='cubic')
lzmin = d_c-lz/2
redbins = zspline(lzmin + np.linspace(0,lz,nz+1))
for i in range(nz):
    dT_HI[:,:,i] = teletools.smoothimage(dT_HI_nobeam[:,:,i],0,0,lx,ly,xbincentres,ybincentres,R_beam)
    dT_obs[:,:,i] = teletools.smoothimage(dT_obs_nobeam[:,:,i],0,0,lx,ly,xbincentres,ybincentres,R_beam)
dT_clean = teletools.FASTICAclean(dT_obs, N_IC=4)

#####################################################
# Measure the auto-power spectrum                   #
#####################################################

kmin,kmax = 0.005,0.5
nkbin = 50
dk = (kmax-kmin)/nkbin
k = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
pkspec = pktools.getpkspec(dT_HI,dT_HI,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspec_nobeam = pktools.getpkspec(dT_HI_nobeam,dT_HI_nobeam,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspecFG = pktools.getpkspec(dT_clean,dT_clean,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
W = pktools.W_alias(nx,ny,nz,lx,ly,lz,p=1)
pkspec, pkspec_nobeam, pkspecFG = pkspec/W**2, pkspec_nobeam/W**2, pkspecFG/W**2  # Correct for aliasing

#####################################################
# Perform multipole expansion on power spectrum     #
#####################################################

pkmults,nmodes = pktools.binpole(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmults_nobeam,nmodes = pktools.binpole(pkspec_nobeam,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmultsFG,nmodes = pktools.binpole(pkspecFG,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)

#####################################################
# Convert from multipoles to P(kperp,kpar)          #
#####################################################

nwedge = 50
pkmu = pktools.pkpoletopkmu(nwedge,pkmults)
pkmu_nobeam = pktools.pkpoletopkmu(nwedge,pkmults_nobeam)
pkmuFG = pktools.pkpoletopkmu(nwedge,pkmultsFG)
kmin2,kmax2 = 0.005,0.3
nk2d = 40
pk2d = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nwedge,pkmu)
pk2d_nobeam = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nwedge,pkmu_nobeam)
pk2dFG = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nwedge,pkmuFG)

#####################################################
# Plot P(kperp,kpara)                               #
#####################################################

dk = (kmax2-kmin2)/nk2d
k = np.linspace(kmin2+0.5*dk,kmax2-0.5*dk,nk2d)

FGsatcap = 1.5
fontsize=22
plt.figure(figsize=(12,5))
for opt in range(2):
    plt.subplot(121+opt)
    histmap = np.zeros((nk2d,nk2d))
    for i in range(nk2d):
        for j in range(nk2d):
            if opt==0: histmap[i,j] = pk2d_nobeam[j,i]/pk2d[j,i]
            if opt==1:
                histmap[i,j] = pk2d[j,i]/pk2dFG[j,i]
                histmap[histmap>FGsatcap] = FGsatcap
                histmap[histmap<0.8] = 0.8
    if opt==0: histmap[np.isnan(histmap)] = 1
    if opt==1: histmap[np.isnan(histmap)] = FGsatcap
    plt.imshow(histmap,extent=[kmin2,kmax2,kmin2,kmax2],cmap='bone_r')
    plt.xlabel(r'$k_\perp \, [h \, {\rm Mpc}^{-1}]$',fontsize=fontsize)
    plt.ylabel(r'$k_\parallel \, [h \, {\rm Mpc}^{-1}]$',fontsize=fontsize)
    plt.xticks([0.1,0.2,0.3])
    plt.yticks([0.1,0.2,0.3])
    if opt==0: plt.title(r'$P(k_\parallel,k_\perp)_\text{NoBeam}/P(k_\parallel,k_\perp)_\text{WithBeam}$',fontsize=fontsize-2)
    if opt==1: plt.title(r'$P(k_\parallel,k_\perp)_\text{NoFG}/P(k_\parallel,k_\perp)_\text{SubFG}$',fontsize=fontsize-2)
    clb = plt.colorbar(fraction=0.1)
    clb.ax.tick_params(labelsize=fontsize-2)
    if opt==1:
        z = (plt.contour(histmap,[1.1, 1.4],extent=[kmin2,kmax2,kmin2,kmax2]))
        plt.clabel(z,fmt = '%1.1f')
    plt.tick_params(labelsize=fontsize-2)
plt.subplots_adjust(top=0.92,
bottom=0.16,
left=0.07,
right=0.97,
hspace=0.22,
wspace=0.22)
plt.show()

#####################################################
# Same Test but with different N_IC inputs          #
#####################################################

dT_clean1 = teletools.FASTICAclean(dT_obs, N_IC=3)
dT_clean2 = teletools.FASTICAclean(dT_obs, N_IC=4)
dT_clean3 = teletools.FASTICAclean(dT_obs, N_IC=6)
dT_clean4 = teletools.FASTICAclean(dT_obs, N_IC=8)
dT_clean5 = teletools.FASTICAclean(dT_obs, N_IC=10)
dT_clean6 = teletools.FASTICAclean(dT_obs, N_IC=12)

#####################################################
# Measure the auto-power spectrum                   #
#####################################################

kmin,kmax = 0.02,0.5
nkbin = 50
dk = (kmax-kmin)/nkbin
k = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
pkspec = pktools.getpkspec(dT_HI,dT_HI,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspecFG1 = pktools.getpkspec(dT_clean1,dT_clean1,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspecFG2 = pktools.getpkspec(dT_clean2,dT_clean2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspecFG3 = pktools.getpkspec(dT_clean3,dT_clean3,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspecFG4 = pktools.getpkspec(dT_clean4,dT_clean4,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspecFG5 = pktools.getpkspec(dT_clean5,dT_clean5,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspecFG6 = pktools.getpkspec(dT_clean6,dT_clean6,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
W = pktools.W_alias(nx,ny,nz,lx,ly,lz,p=1)
pkspec, pkspecFG1, pkspecFG2, pkspecFG3, pkspecFG4, pkspecFG5, pkspecFG6 = pkspec/W**2, pkspecFG1/W**2, pkspecFG2/W**2, pkspecFG3/W**2, pkspecFG4/W**2, pkspecFG5/W**2, pkspecFG6/W**2  # Correct for aliasing

#####################################################
# Perform multipole expansion on power spectrum     #
#####################################################

pkmults,nmodes = pktools.binpole(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmultsFG1,nmodes = pktools.binpole(pkspecFG1,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmultsFG2,nmodes = pktools.binpole(pkspecFG2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmultsFG3,nmodes = pktools.binpole(pkspecFG3,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmultsFG4,nmodes = pktools.binpole(pkspecFG4,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmultsFG5,nmodes = pktools.binpole(pkspecFG5,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmultsFG6,nmodes = pktools.binpole(pkspecFG6,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)

#####################################################
# Convert from multipoles to P(kperp,kpar)          #
#####################################################

nwedge = 50
pkmu = pktools.pkpoletopkmu(nwedge,pkmults)
pkmuFG1 = pktools.pkpoletopkmu(nwedge,pkmultsFG1)
pkmuFG2 = pktools.pkpoletopkmu(nwedge,pkmultsFG2)
pkmuFG3 = pktools.pkpoletopkmu(nwedge,pkmultsFG3)
pkmuFG4 = pktools.pkpoletopkmu(nwedge,pkmultsFG4)
pkmuFG5 = pktools.pkpoletopkmu(nwedge,pkmultsFG5)
pkmuFG6 = pktools.pkpoletopkmu(nwedge,pkmultsFG6)
kmin2,kmax2 = 0.005,0.3
nk2d = 40
pk2d = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nwedge,pkmu)
pk2dFG1 = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nwedge,pkmuFG1)
pk2dFG2 = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nwedge,pkmuFG2)
pk2dFG3 = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nwedge,pkmuFG3)
pk2dFG4 = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nwedge,pkmuFG4)
pk2dFG5 = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nwedge,pkmuFG5)
pk2dFG6 = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nwedge,pkmuFG6)

#####################################################
# Plot P(kperp,kpara)                               #
#####################################################

dk = (kmax2-kmin2)/nk2d
k = np.linspace(kmin2+0.5*dk,kmax2-0.5*dk,nk2d)

pk2dFGlist = [pk2dFG6,pk2dFG5,pk2dFG4,pk2dFG3,pk2dFG2,pk2dFG1]

FGsatcap = 2
fontsize=22
plt.figure(figsize=(7,4.5))
histmap = np.zeros((nk2d,nk2d))
colors = ['black','brown','r','tomato','lightsalmon','linen']
for FGi in range(6):
    pk2dFG = pk2dFGlist[FGi]
    for i in range(nk2d):
        for j in range(nk2d):
            histmap[i,j] = pk2d[j,i]/pk2dFG[j,i]
    histmap[histmap>FGsatcap] = FGsatcap
    histmap[histmap<0.8] = 0.8
    histmap[np.isnan(histmap)] = FGsatcap
    z = (plt.contourf(histmap,[1.4,2.9],extent=[kmin2,kmax2,kmin2,kmax2],alpha=0.7,colors=colors[FGi]))
    zc = (plt.contour(histmap,[1.4],extent=[kmin2,kmax2,kmin2,kmax2],alpha=0.7,color='black'))
    plt.clabel(zc,fmt = '%1.1f')

proxy=[]
for i in range(6):
    proxy.append( plt.Rectangle((0,0),1,1, fc=colors[i]) )
plt.legend(proxy, ["$N_\\text{IC}=12$","$N_\\text{IC}=10$","$N_\\text{IC}=8$","$N_\\text{IC}=6$","$N_\\text{IC}=4$","$N_\\text{IC}=3$"],ncol=3,prop={'size': 17},loc='upper center')

plt.xlabel(r'$k_\perp \, [h \, {\rm Mpc}^{-1}]$',fontsize=fontsize)
plt.ylabel(r'$k_\parallel \, [h \, {\rm Mpc}^{-1}]$',fontsize=fontsize)
plt.xticks([0.1,0.2,0.3])
plt.yticks([0.1,0.2,0.3])
plt.title(r'$P(k_\parallel,k_\perp)_\text{NoFG}/P(k_\parallel,k_\perp)_\text{SubFG}$',fontsize=fontsize)
plt.tick_params(labelsize=fontsize-2)
plt.subplots_adjust(top=0.91,
bottom=0.18,
left=0.14,
right=0.97,
hspace=0.22,
wspace=0.22)
plt.show()
