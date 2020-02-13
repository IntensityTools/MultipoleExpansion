import numpy as np
import pktools
n_jack = 9 #Number of jacknife subsamples

########################################################################
#
#   Jackknife intensity maps for Pk errors.
#   n_jack is the number of subsamples you split data up into then code is run each time
#   jacknifed with an exlcluded subsample so you have (n-1) samples in each iteration.
#   From these iterations, standard deviation is found for error bars
#
########################################################################

def MultipoleError(dT_HI1,dT_HI2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin):
    '''
    Jacknife code for Pk multipoles
    '''
    pk0_jack,pk2_jack,pk4_jack = JacknifeSpectrum(dT_HI1,dT_HI2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,DoWedges=False)
    sig_pk0 = ProcessJackknife(pk0_jack)
    sig_pk2 = ProcessJackknife(pk2_jack)
    sig_pk4 = ProcessJackknife(pk4_jack)
    return sig_pk0,sig_pk2,sig_pk4

def WedgesError(dT_HI1,dT_HI2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin):
    '''
    Jacknife code for Pk wedges power specta
    '''
    pkmu_jack = JacknifeSpectrum(dT_HI1,dT_HI2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,DoWedges=True)
    nwedge = np.shape(pkmu_jack)[0]
    sig_wedges=[]
    for w in range(nwedge):
        sig_wedges.append( ProcessJackknife(pkmu_jack[w]) )
    return sig_wedges

def JacknifeSpectrum(dT_HI1,dT_HI2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,DoWedges=True):
    '''
    Jacknife code for multipoles (P_l) power specta
    '''
    Nr,Nd,Nz = np.shape(dT_HI1)
    nsqrt = np.int(np.sqrt(n_jack))
    pk0_jack=[];pk2_jack=[];pk4_jack=[];pkmu_jack=[]
    for i in range(nsqrt):
        for j in range(nsqrt):
            dT_HI_jack1,dT_HI_jack2 = np.copy(dT_HI1),np.copy(dT_HI2)
            #Create mask to jackknife data with:
            ra0_jack = i * (Nr/nsqrt)
            ra1_jack = (i+1) * (Nr/nsqrt)
            dec0_jack = j * (Nd/nsqrt)
            dec1_jack = (j+1) * (Nd/nsqrt)
            y, x = np.indices(dT_HI1[0].shape)
            jackmask = (x >= ra0_jack) & (x < ra1_jack) & (y >= dec0_jack) & (y < dec1_jack)
            for zi in range(Nz):
                dT_HI_jack1[zi][jackmask] = 0
                dT_HI_jack2[zi][jackmask] = 0
            # Measure the auto-power spectrum:
            pkspec = pktools.getpkspec(dT_HI_jack1,dT_HI_jack2,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
            Pk = pktools.binpk(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
            # Perform multipole expansion on power spectrum:
            [pk0_J,pk2_J,pk4_J],nmodes = pktools.binpole(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin)
            if DoWedges==False: pk0_jack.append(pk0_J); pk2_jack.append(pk2_J); pk4_jack.append(pk4_J)
            else:
                nwedge = 4
                pkmu_J = pktools.pkpoletopkmu(nwedge,pk0_J,pk2_J,pk4_J)
                pkmu_jack.append(pkmu_J)
            print(str(i*nsqrt + j + 1), 'out of', str(n_jack), 'subsamples jack-knifed')
    if DoWedges==False: return pk0_jack,pk2_jack,pk4_jack
    else:
        pkmu_jack = np.swapaxes(pkmu_jack,0,2)
        pkmu_jack = np.swapaxes(pkmu_jack,1,2)
        return pkmu_jack

def ProcessJackknife(pk_jack):
    '''
    Process Jackknifed data to obtain errors
    '''
    n_jack,nkbin = np.shape(pk_jack)
    sig=[]
    for i in range(nkbin):
        #Obtain each subsample value for particular k-bin:
        pk_jack_k = [val[i] for val in pk_jack]
        mu = np.mean(pk_jack_k)
        #Formula for variance of jackknived data, from wiki: https://en.wikipedia.org/wiki/Jackknife_resampling
        var = (n_jack - 1) / n_jack * np.sum( ( pk_jack_k - mu )**2 )
        sig.append( np.sqrt(var) ) #Get standard.dev for error bars
    return sig
