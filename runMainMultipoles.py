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
'''
Choice of redshift determines which instrumental settings are used
[please see the companion paper for further details]
'''
h = cosmotools.H(0) / 100
zeff = 0.82 #Redshift for simulation box
#zeff = 2.03 #Redshift for simulation box
Tbar = teletools.TbarModel(zeff) #mK
d_c = cosmotools.D_com(zeff) #comoving distance to effective redshift
if zeff==0.82:
    b_HI = 1.15
    P_SN = Tbar**2*124 #From https://arxiv.org/pdf/1804.09180.pdf (Table 5)
    theta_FWHM,R_beam = teletools.getbeampars(zeff,d_c)
    sig_v = 250*h
if zeff==2.03:
    b_HI = 1.95
    P_SN = Tbar**2*65 #From https://arxiv.org/pdf/1804.09180.pdf (Table 5)
    theta_FWHM,R_beam = teletools.getbeampars(zeff,d_c)
    sig_v = 0

# Script Options:
#####################################################
DoNoise = False
DoWedges = False #Set True for Clustering Wedges analysis
Jackknife = False #Set True to obtain error-bars with Jackknifing method
#####################################################

#####################################################
# Read-in and Process HI and FG maps                #
#####################################################

dT_HI = np.load('MultiDarkSims/dT_HI-MDSAGE_z_%s.npy'%zeff)
dT_FG = np.load('MultiDarkSims/dT_FG-MDSAGE_z_%s.npy'%zeff)
dT_obs = dT_HI + dT_FG
xbincentres = xbins+(xbins[1]-xbins[0])/2
xbincentres = xbincentres[:len(xbincentres)-1] #remove last value since this is outside of bins
ybincentres = ybins+(ybins[1]-ybins[0])/2
ybincentres = ybincentres[:len(ybincentres)-1] #remove last value since this is outside of bins
if DoWedges==True: R_beam = 6
for i in range(nz):
    dT_HI[:,:,i] = teletools.smoothimage(dT_HI[:,:,i],0,0,lx,ly,xbincentres,ybincentres,R_beam)
    dT_obs[:,:,i] = teletools.smoothimage(dT_obs[:,:,i],0,0,lx,ly,xbincentres,ybincentres,R_beam)
if DoNoise==True:
    dv = 1e6 #Hz (1MHz freq bin width) - use for noise map
    dT_noise1,sigma_N = teletools.ReceiverNoise(zeff,dv,d_c,nx,ny,nz,lx,ly,theta_FWHM)
    dT_noise2,sigma_N = teletools.ReceiverNoise(zeff,dv,d_c,nx,ny,nz,lx,ly,theta_FWHM)
else: dT_noise1,dT_noise2,sigma_N = np.zeros(np.shape(dT_HI)),np.zeros(np.shape(dT_HI)),0
dT_HI1 = dT_HI + dT_noise1
dT_HI2 = dT_HI + dT_noise2
dT_obs1 = dT_obs + dT_noise1
dT_obs2 = dT_obs + dT_noise2
dT_clean1 = teletools.FASTICAclean(dT_obs1, N_IC=4)
dT_clean2 = teletools.FASTICAclean(dT_obs2, N_IC=4)

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
pk0mod,pk2mod,pk4mod = modellingtools.MultipoleExpansion(Pmod,kmod,b_HI,B,P_SN,R_beam,sig_v,doFG=False,noRSD=False)
pk0modFG,pk2modFG,pk4modFG = modellingtools.MultipoleExpansion(Pmod,kmod,b_HI,B,P_SN,R_beam,sig_v,doFG=True,noRSD=False)
pkmultsmods = [pk0mod,pk2mod,pk4mod]
pkmultsmodsFG = [pk0modFG,pk2modFG,pk4modFG]

#####################################################
# Measure the auto-power spectrum                   #
#####################################################

kmin,kmax = 0.02,0.3
nkbin = 29
dk = (kmax-kmin)/nkbin
k = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
pkspec = pktools.getpkspec(dT_HI1,dT_HI2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkspecFG = pktools.getpkspec(dT_clean1,dT_clean2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
W = pktools.W_alias(nx,ny,nz,lx,ly,lz,p=1)
pkspec, pkspecFG = pkspec/W**2, pkspecFG/W**2  # Correct for aliasing

#####################################################
# Perform multipole expansion on power spectrum     #
#####################################################

pkmults,nmodes = pktools.binpole(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
pkmultsFG,nmodes = pktools.binpole(pkspecFG,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)

#####################################################
# Convert from multipoles to P(kperp,kpar) wedges   #
#####################################################

if DoWedges==True:
    nwedge = 4 #Number of wedges to produce
    pkmu = pktools.pkpoletopkmu(nwedge,pkmults)
    pkmuFG = pktools.pkpoletopkmu(nwedge,pkmultsFG)

#####################################################
# Obtain error bars                                 #
#####################################################

if Jackknife==False:
    vol_cell = lx*ly*lz / (nx*ny*nz)
    P_N = sigma_N**2 * vol_cell
    sig_pl = stattools.MultipoleError(P_N,pkmults[0],pkmults[1],pkmults[2],nmodes)
    sigFG_pl = stattools.MultipoleError(P_N,pkmultsFG[0],pkmultsFG[1],pkmultsFG[2],nmodes)
    if DoWedges==True:
        sig_wedges = stattools.WedgesError(nwedge,pkmults[0],pkmults[1],pkmults[2],sig_pl,nmodes)
        sigFG_wedges = stattools.WedgesError(nwedge,pkmultsFG[0],pkmultsFG[1],pkmultsFG[2],sigFG_pl,nmodes)
if Jackknife==True:
    if DoWedges==False:
        sig_pl = jackknifetools.MultipoleError(dT_HI1,dT_HI2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
        sigFG_pl = jackknifetools.MultipoleError(dT_clean1,dT_clean2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
    if DoWedges==True:
        sig_wedges = jackknifetools.WedgesError(dT_HI1,dT_HI2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
        sigFG_wedges = jackknifetools.WedgesError(dT_clean1,dT_clean2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)

#####################################################
# Calculate %-Residuals & Chi-squared                 #
#####################################################

if DoWedges==False:
    Pmod_datakbin = np.load('Inputs/Pk_z-datakbin=%s.npy'%zeff)
    pk0mod,pk2mod,pk4mod = modellingtools.MultipoleExpansion(Pmod_datakbin,k,b_HI,B,P_SN,R_beam,sig_v,doFG=False,noRSD=False)
    pk0modFG,pk2modFG,pk4modFG = modellingtools.MultipoleExpansion(Pmod_datakbin,k,b_HI,B,P_SN,R_beam,sig_v,doFG=True,noRSD=False)
    #Obtain model with same k-binning as data (used for residual plots)
    pkmods_datak = pk0mod,pk2mod,pk4mod
    pkmods_datakFG = pk0modFG,pk2modFG,pk4modFG
    dof = len(k)
    print('\nNo FG Chi-squared/dof:')
    print('P_0 = ', stattools.ChiSquare(pkmults[0],Tbar**2*pk0mod,sig_pl[0])/dof)
    print('P_2 = ', stattools.ChiSquare(pkmults[1],Tbar**2*pk2mod,sig_pl[1])/dof)
    print('P_4 = ', stattools.ChiSquare(pkmults[2],Tbar**2*pk4mod,sig_pl[2])/dof)
    print('\nSub FG Chi-squared/dof:')
    print('P_0 = ', stattools.ChiSquare(pkmultsFG[0],Tbar**2*pk0modFG,sigFG_pl[0])/(dof+1)) #dof + 1 for FG case
    print('P_2 = ', stattools.ChiSquare(pkmultsFG[1],Tbar**2*pk2modFG,sigFG_pl[1])/(dof+1)) #  since 1 extra fitting parameter
    print('P_4 = ', stattools.ChiSquare(pkmultsFG[2],Tbar**2*pk4modFG,sigFG_pl[2])/(dof+1))

#####################################################
# Plot Multipoles                                   #
#####################################################

if DoWedges==False:
    kmax = 0.3
    fontsize = 25
    fig = plt.figure(figsize=(12,7))
    gs=GridSpec(3,3) # 3 rows, 3 columns
    for i in range(6):
        if i<3:
            ax1=fig.add_subplot(gs[0:2,i]) # First row, first column
            ax1.errorbar(k, k*pkmults[i], yerr=k*sig_pl[i], fmt='x',capsize=5, markersize=8, label='No FG (Sim)',color='Black')
            ax1.errorbar(k, k*pkmultsFG[i], yerr=k*sigFG_pl[i], fmt='o',capsize=5, markersize=5, label='Sub FG (Sim)',color='Red')
            ax1.plot(kmod,kmod*Tbar**2*pkmultsmods[i],linestyle='dashed',color='Black',label='No FG (Mod)')
            ax1.plot(kmod,kmod*Tbar**2*pkmultsmodsFG[i],linestyle='dotted',linewidth=2.5,color='Red',label='Sub FG (Mod)')
            ax1.set_xlim(left=0,right=kmax)
            ax1.set_ylabel(r'$k \,P_{%s}(k) \, $[mK$^2\,h^{-2} \, {\rm Mpc}^2$]'%(i*2),fontsize=fontsize)
            ax1.set_xticks([0.2]) #set one tick to hide behind residual plot
            if i==0: ax1.set_yticks([0,2,4,6,8])
            if i==1: ax1.set_yticks([0,2,4,6,8,10])
            if i==2: ax1.set_yticks([-6,-3,0,3])
            ax1.tick_params(labelbottom='off',labelsize=fontsize-5)
        if i==1: ax1.legend(ncol=4,loc='upper center',bbox_to_anchor=[0.42, 1.23],fontsize=fontsize-5)
        if i>=3:
            ax2=fig.add_subplot(gs[2,i-3]) # Last row, first column
            ax2.plot([0,1],np.ones(2),linestyle='dashed',linewidth=2.5,color='Black')
            ax2.errorbar(k,-100*(1-pkmults[i-3]/(Tbar**2*pkmods_datak[i-3])), yerr=100*(sig_pl[i-3]/(Tbar**2*pkmods_datak[i-3])), fmt='x',capsize=5, markersize=8, color='Black')
            ax2.errorbar(k,-100*(1-pkmultsFG[i-3]/(Tbar**2*pkmods_datakFG[i-3])), yerr=100*(sigFG_pl[i-3]/(Tbar**2*pkmods_datakFG[i-3])), fmt='o',capsize=5, markersize=5, color='Red')
            if i<5:
                ax2.set_ylim(bottom=-30,top=30)
                ax2.set_yticks([-20,-10,0,10,20])
                ax2.set_yticklabels(['20','','0','','20'])
            else:
                ax2.set_ylim(bottom=-90,top=90)
                ax2.set_yticks([-60,-30,0,30,60])
                ax2.set_yticklabels(['60','','0','','60'])
            ax2.set_xticks([0,0.1,0.2,0.3])
            ax2.set_xlim(left=0,right=kmax)
            ax2.set_xlabel(r'$k \, [h \, {\rm Mpc}^{-1}]$',fontsize=fontsize)
            ax2.set_ylabel('Residuals [\%]', fontsize=fontsize-3)
            ax2.tick_params(labelsize=fontsize-5)
    plt.subplots_adjust(top=0.9,
    bottom=0.13,
    left=0.08,
    right=0.985,
    hspace=0,
    wspace=0.35)
    plt.show()

#####################################################
# Plot Wedges                                       #
#####################################################

if DoWedges==True:
    mulims = [0.000,0.250,0.500,0.750,1.000]
    fontsize=22
    plt.figure(figsize=(11,7))
    for i in range(nwedge):
        plt.subplot(221+i)
        plt.errorbar(k, k*pkmu[:,i], yerr=k*sig_wedges[i], fmt='x',capsize=5, markersize=8, label='No FG',color='Black')
        plt.errorbar(k, k*pkmuFG[:,i], yerr=k*sigFG_wedges[i], fmt='o',capsize=5, markersize=5, label='Sub FG',color='Red')
        plt.xlim(left=kmin,right=0.3)
        if i>1: plt.xlabel(r'$k \, [h \, {\rm Mpc}^{-1}]$',fontsize=fontsize)
        else: plt.xticks([0.1,0.2,0.3],[])
        if i==0 or i==2: plt.ylabel(r'$k \,P(k) \, $[mK$^2\,h^{-2} \, {\rm Mpc}^2]$',fontsize=fontsize)
        plt.ylim(bottom=0)
        plt.legend(fontsize=fontsize-5)
        plt.tick_params(labelsize=fontsize-2)
        plt.title('Wedge %s:'%(i+1) +' $%s$'%np.round(mulims[i],2) + '$\,<\\mu<%s$'%np.round(mulims[i+1],3), fontsize=fontsize )
        plt.tick_params(labelsize=fontsize-2)
    plt.subplots_adjust(top=0.95,
    bottom=0.12,
    left=0.08,
    right=0.985,
    hspace=0.22,
    wspace=0.13)
    plt.show()
