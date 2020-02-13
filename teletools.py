import numpy as np
import scipy
from scipy import fftpack
from sklearn.decomposition import FastICA, PCA
import cosmotools
v_21cm = 1420.405751#MHz

#D_dish = 15 #Dish diameter in metres
D_dish = 100 #Dish diameter in metres
N_dish = 1 #Number of dishes
t_obs = 200 * 60 * 60 #Observation time (hours converted to secs)

def getbeampars(zeff,d_c):
    theta_FWHM = np.degrees(1.22 * 0.21 / D_dish * (1+zeff))
    sig_beam = theta_FWHM/(2*np.sqrt(2*np.log(2)))
    R_beam = d_c * np.radians(sig_beam) #Beam sigma
    print('\nTelescope Params: Dish size =',D_dish,'m, R_beam =',np.round(R_beam,1),'Mpc/h, theta_FWHM =',np.round(theta_FWHM,2),'deg\n')
    return theta_FWHM,R_beam

def ReceiverNoise(zeff,deltav,d_c,nx,ny,nz,lx,ly,beamsize):
    '''
    Returns noise map as a function of redshift bin edges.
    Use Alonso et al. Method: eq22 in https://arxiv.org/pdf/1409.8667.pdf
    All T_sys related values from Santos et al. pg 16: https://arxiv.org/pdf/1501.03989.pdf
    '''
    v = v_21cm / (1+zeff) #MHz
    T_sky = 60 * (300/v)**2.55 #frequncies in MHz
    T_inst = 30 #in Kelvin for SKA1-MID lowest redshift range
    T_rcvr = 0.1*T_sky + T_inst
    T_sys = (T_rcvr + T_sky) * 1e3 #convert to mK for consistancy with rest of sims
    Omega_beam = 1.133 * np.radians(beamsize)**2
    deltar = np.degrees(lx / d_c)
    deltad = np.degrees(ly / d_c)
    f_sky = deltar*deltad / 41253 #deg^2
    T_sys = 10000 #mK
    sigma_N = T_sys * np.sqrt( 4*np.pi * f_sky / (deltav * t_obs * N_dish * Omega_beam) )
    dT_noise = np.random.normal( 0, sigma_N, nx*ny*nz )
    dT_noise = np.reshape(dT_noise,(nx,ny,nz))
    return dT_noise,sigma_N

def smoothimage(image,xmin,ymin,xwidth,ywidth,xbincentres,ybincentres,sig):
    '''
    Takes a flat image and runs a convolution to smooth the image. Convolution kernel
    is a Gaussian with an input sigma. Units must be consistent e.g. degrees or
    Mpc/h throughout.
    '''
    #Create Gaussian field
    x = xbincentres
    y = ybincentres[:,np.newaxis]
    x0 = xmin+xwidth/2
    y0 = ymin+ywidth/2
    gaussian = np.exp(-0.5 * (((x-x0)/sig)**2 + ((y-y0)/sig)**2))
    gaussian = np.swapaxes(gaussian,0,1)
    A = np.sum(gaussian)
    gaussian = gaussian/A #normalise gaussian so that all pixels sum to 1
    ft = fftpack.fft2(image)
    ft_gauss=fftpack.fft2(gaussian)
    smoothimage = fftpack.ifft2(ft*ft_gauss)
    smoothimage = np.real(smoothimage)
    #Now shift the quadrants around into correct position
    return fftpack.fftshift(smoothimage)

def FASTICAclean(Input, N_IC=4):
    '''
    Takes input in either [Nx,Ny,Nz] data cube form or HEALpix [Npix,Nz]
    form where Nz is number of redshift (frequency) bins. N_IC is number of
    independent components for FASTICA to try and find
    '''
    # Check shape of input array. If flat data cube, collapse (ra,dec) structure
    #    to (Npix,Nz) structure reqd for FASTICA:
    axes = np.shape(Input)
    if len(axes)==3: Input = np.reshape(Input,(axes[0]*axes[1],axes[2]))
    ica = FastICA(n_components=N_IC, whiten=True)
    S_ = ica.fit_transform(Input) # Reconstruct signals
    A_ = ica.mixing_ # Get estimated mixing matrix
    Recon_FG = np.dot(S_, A_.T) + ica.mean_ #Reconstruct foreground
    Residual = Input - Recon_FG #Residual of fastICA is HI plus any Noise
    if len(axes)==3: Residual = np.reshape(Residual,(axes[0],axes[1],axes[2])) #Rebuild if Input was 3D datacube
    return Residual

def b_HI(z):
    return 0.67 + 0.18*z + 0.05*z**2

def TbarModel(z):
    '''
    [Units of mK]
    '''
    OmegaHI = 0.43e-3 / b_HI(z) #Result from Masui 2013 GBT paper (assumes r=1)
    Hz = cosmotools.H(z) #km / Mpc s
    H0 = cosmotools.H(0) #km / Mpc s
    h = H0/100
    return 180 * OmegaHI * h * (1+z)**2 / (Hz/H0)
