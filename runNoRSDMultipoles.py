import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import chisquare
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
import modellingtools
import stattools
import jackknifetools
#from nbodykit.lab import cosmology

# Dimension of data cube:
lx,ly,lz = 1000,1000,1000 #Mpc/h
nx,ny,nz = 225,225,225
lzbuff = 119 #Used to achieve a lz=762Mpc/h which is ~Deltaz=0.4 at zeff=0.8
xbins = np.linspace(0,lx,nx+1) #cartesian-z pixel bins
ybins = np.linspace(0,ly,ny+1) #cartesian-z pixel bins
lz = lz - 2*lzbuff

# Cosmological & Survey Parameters:
zeff = 0.82 #Redshift for simulation boz
Tbar = teletools.TbarModel(zeff) #mK
d_c = cosmotools.D_com(zeff)
b_HI = 1.2
P_SN = Tbar**2*124 #From https://arxiv.org/pdf/1804.09180.pdf (Table 5)
R_beam = 10 #Mpc/h
sigma_N = 0 #No instrumental noise in this test

# Script Options:
#####################################################
Jackknife = False #Set True to obtain and plot error-bars with Jackknifing method
#####################################################

#####################################################
# Read-in and Process HI and FG maps                #
#####################################################

dT_HI_nobeam = np.load('MultiDarkSims/dT_HI-MDSAGE_z_%s_NoRSD.npy'%zeff)
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
dT_clean_nobeam = teletools.FASTICAclean(dT_obs_nobeam, N_IC=4)
dT_clean = teletools.FASTICAclean(dT_obs, N_IC=4)

#####################################################
# Calculate model power spectra multipoles          #
#####################################################

ffid = cosmotools.f(zeff)
B = ffid / b_HI
#Nbkcosmo = cosmotools.getNBKcosmo()
#Pmod = cosmology.HalofitPower(Nbkcosmo, redshift=zeff)
#kmod = np.linspace(0,0.3,100)
#Pmod = Pmod(kmod)
kmod,Pmod = np.load('Inputs/Pk_z=%s.npy'%zeff,allow_pickle=True)
pk0mod,pk2mod,pk4mod = modellingtools.MultipoleExpansion(Pmod,kmod,b_HI,B,P_SN,R_beam,sig_v_loc=0,doFG=False,noRSD=True)
pk0modFG,pk2modFG,pk4modFG = modellingtools.MultipoleExpansion(Pmod,kmod,b_HI,B,P_SN,R_beam,sig_v_loc=0,doFG=True,noRSD=True)
pkmultsmods = [pk0mod,pk2mod,pk4mod]
pkmultsmodsFG = [pk0modFG,pk2modFG,pk4modFG]
pk0mod,pk2mod,pk4mod = modellingtools.MultipoleExpansion(Pmod,kmod,b_HI,B,P_SN,R_beam=0,sig_v_loc=0,doFG=False,noRSD=True)
pk0modFG,pk2modFG,pk4modFG = modellingtools.MultipoleExpansion(Pmod,kmod,b_HI,B,P_SN,R_beam=0,sig_v_loc=0,doFG=True,noRSD=True)
pkmultsmodsNoBeam = [pk0mod,pk2mod,pk4mod]
pkmultsmodsFGNoBeam = [pk0modFG,pk2modFG,pk4modFG]

#####################################################
# Measure the auto-power spectrum                   #
#####################################################

kmin,kmax = 0.02,0.3
nkbin = 29
dk = (kmax-kmin)/nkbin
k = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
W = pktools.W_alias(nx,ny,nz,lx,ly,lz,p=1)
pkspec_nobeam = pktools.getpkspec(dT_HI_nobeam,dT_HI_nobeam,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspecFG_nobeam = pktools.getpkspec(dT_clean_nobeam,dT_clean_nobeam,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspec = pktools.getpkspec(dT_HI,dT_HI,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspecFG = pktools.getpkspec(dT_clean,dT_clean,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspec_nobeam, pkspecFG_nobeam, pkspec, pkspecFG = pkspec_nobeam/W**2, pkspecFG_nobeam/W**2, pkspec/W**2, pkspecFG/W**2  # Correct for aliasing

#####################################################
# Perform multipole expansion on power spectrum     #
#####################################################

pkmults_nobeam,nmodes = pktools.binpole(pkspec_nobeam,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmultsFG_nobeam,nmodes = pktools.binpole(pkspecFG_nobeam,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmults,nmodes = pktools.binpole(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmultsFG,nmodes = pktools.binpole(pkspecFG,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)

#####################################################
# Obtain error bars                                 #
#####################################################

if Jackknife==False:
    vol_cell = lx*ly*lz / (nx*ny*nz)
    P_N = sigma_N**2 * vol_cell
    sig_err_nobeam = stattools.MultipoleError(P_N,pkmults_nobeam[0],pkmults_nobeam[1],pkmults_nobeam[2],nmodes)
    sig_errFG_nobeam = stattools.MultipoleError(P_N,pkmultsFG_nobeam[0],pkmultsFG_nobeam[1],pkmultsFG_nobeam[2],nmodes)
    sig_err = stattools.MultipoleError(P_N,pkmults[0],pkmults[1],pkmults[2],nmodes)
    sig_errFG = stattools.MultipoleError(P_N,pkmultsFG[0],pkmultsFG[1],pkmultsFG[2],nmodes)
if Jackknife==True:
    sig_err_nobeam = jackknifetools.MultipoleError(dT_HI_nobeam,dT_HI_nobeam,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
    sig_errFG_nobeam = jackknifetools.MultipoleError(dT_clean_nobeam,dT_clean_nobeam,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
    sig_err = jackknifetools.MultipoleError(dT_HI,dT_HI,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
    sig_errFG = jackknifetools.MultipoleError(dT_clean,dT_clean,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)

#####################################################
# Calculate and Display Chi-squared                 #
#####################################################

Pmod_datakbin = np.load('Inputs/Pk_z-datakbin=%s.npy'%zeff)
pk0modNoBeam,pk2modNoBeam,pk4modNoBeam = modellingtools.MultipoleExpansion(Pmod_datakbin,k,b_HI,B,P_SN,R_beam=0,sig_v_loc=0,doFG=False,noRSD=True)
pk0mod,pk2mod,pk4mod = modellingtools.MultipoleExpansion(Pmod_datakbin,k,b_HI,B,P_SN,R_beam,sig_v_loc=0,doFG=False,noRSD=True)
pk0modFG,pk2modFG,pk4modFG = modellingtools.MultipoleExpansion(Pmod_datakbin,k,b_HI,B,P_SN,R_beam,sig_v_loc=0,doFG=True,noRSD=True)
dof = len(k)
print('\nNo FG (with Beam) Chi-squared/dof:')
print('P_2 = ', stattools.ChiSquare(pkmults[1], Tbar**2*pk2mod,sig_err[1])/dof)
print('P_4 = ', stattools.ChiSquare(pkmults[2], Tbar**2*pk4mod,sig_err[2])/dof)
print('\nSub FG (with Beam) Chi-squared:')
print('P_2 = ', stattools.ChiSquare(pkmultsFG[1], Tbar**2*pk2modFG,sig_errFG[1])/dof)
print('P_4 = ', stattools.ChiSquare(pkmultsFG[2], Tbar**2*pk4modFG,sig_errFG[2])/dof)

#####################################################
# Plot No RSD Multipoles                            #
#####################################################

fontsize = 25
titles=['No RSD Quadrupole','No RSD Hexadecapole']
plt.figure(figsize=(12,5))
for i in range(2):
    plt.subplot(121+i)
    plt.plot(kmod,np.zeros(len(kmod)),color='grey',linewidth=1)
    plt.plot(kmod,kmod*Tbar**2*pkmultsmods[i+1],linestyle='dashed',color='Black')
    plt.plot(kmod,kmod*Tbar**2*pkmultsmodsFG[i+1],linestyle='dotted',linewidth=2.5,color='Red')
    plt.errorbar(k,k*pkmults_nobeam[i+1],yerr=k*sig_err_nobeam[i+1], fmt='s',capsize=5, markersize=4, label='No FG (No Beam)',color='Blue')
    plt.errorbar(k,k*pkmults[i+1],yerr=k*sig_err[i+1], fmt='x',capsize=5, markersize=8, label='No FG (with Beam)',color='Black')
    plt.errorbar(k,k*pkmultsFG[i+1],yerr=k*sig_errFG[i+1], fmt='o',capsize=5, markersize=5, label='Sub FG (with Beam)',color='Red')
    plt.ylabel(r'$k \,P_{%s}(k) \, $[mK$^2\,h^{-2} \, {\rm Mpc}^2]$'%((i+1)*2),fontsize=fontsize-3)
    plt.xlabel(r'$k \, [h \, {\rm Mpc}^{-1}]$',fontsize=fontsize-3)
    plt.xticks(np.linspace(0,0.3,4),['0.0','0.1','0.2','0.3'])
    plt.title(titles[i],fontsize=fontsize)
    plt.tick_params(labelsize=fontsize-2)
    if i==1: plt.legend(ncol=3,loc='upper center',bbox_to_anchor=[-0.2, 1.35],fontsize=fontsize-5)
    plt.xlim(left=0,right=0.3)
plt.subplots_adjust(top=0.8,
bottom=0.18,
left=0.08,
right=0.98,
hspace=0.25,
wspace=0.29)
plt.show()
