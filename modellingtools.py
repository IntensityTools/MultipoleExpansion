import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pktools
import cosmotools
H_0 = cosmotools.H(0)

# Damping functions for beam, frequency channel binning and alias effect:
def B_beam(mu,k,R_beam):
    return np.exp( -(1-mu**2)*k**2*R_beam**2/2 )
# Anisotropic power spectrum and Legendre polynomials:
def P(k,mu,i):
    return b**2 * (1 + B*mu**2)**2 / (1 + (k*mu*sig_v/H_0)**2) * Pk_M[i]
def Leg0(mu):
    return 1
def Leg2(mu):
    return (3*mu**2 - 1) / 2
def Leg4(mu):
    return (35*mu**4 - 30*mu**2 + 3) / 8

def MultipoleExpansion(Pk_M_loc,kmod,b_loc,B_loc,P_SN_loc,R_beam,sig_v_loc,doFG=True,noRSD=False):
    '''
    Model multipole expansion function to be used for modelling expansion of both
    IM power spectrum with observable effects and also for modelling the shot noise
    individually through expansion
    '''
    #Declare some global values/functions
    global Pk_M; global b; global B; global sig_v; global P_SN
    Pk_M = Pk_M_loc; b = b_loc; B = B_loc; sig_v = sig_v_loc; P_SN = P_SN_loc
    #Define Parameters and multipoles:
    if noRSD==True:
        PL0 = lambda mu: 1/2 * b**2 * Pk_M_i * B_beam(mu,k_i,R_beam)**2 * Leg0(mu)
        PL2 = lambda mu: 5/2 * b**2 * Pk_M_i * B_beam(mu,k_i,R_beam)**2 * Leg2(mu)
        PL4 = lambda mu: 9/2 * b**2 * Pk_M_i * B_beam(mu,k_i,R_beam)**2 * Leg4(mu)
    if noRSD==False: #Inlcude RSD e.g. Kaiser/FoG effect
        PL0 = lambda mu: 1/2 * P(k_i,mu,i) * B_beam(mu,k_i,R_beam)**2 * Leg0(mu)
        PL2 = lambda mu: 5/2 * P(k_i,mu,i) * B_beam(mu,k_i,R_beam)**2 * Leg2(mu)
        PL4 = lambda mu: 9/2 * P(k_i,mu,i) * B_beam(mu,k_i,R_beam)**2 * Leg4(mu)
    PSN0 = lambda mu: 1/2 * P_SN * B_beam(mu,k_i,R_beam)**2 * Leg0(mu)
    PSN2 = lambda mu: 5/2 * P_SN * B_beam(mu,k_i,R_beam)**2 * Leg2(mu)
    PSN4 = lambda mu: 9/2 * P_SN * B_beam(mu,k_i,R_beam)**2 * Leg4(mu)
    if doFG==True: k_FG = 0.015 #Run for z=0.82
    #if doFG==True: k_FG = 0.019 #Run for z=2.03
    else: k_FG = 0
    P_0 = np.zeros(len(kmod)); P_2 = np.zeros(len(kmod)); P_4 = np.zeros(len(kmod))
    PSN_0 = np.zeros(len(kmod)); PSN_2 = np.zeros(len(kmod)); PSN_4 = np.zeros(len(kmod))
    for i in range(len(kmod)):
        k_i = kmod[i]
        Pk_M_i = Pk_M[i]
        if k_i==0: k_i = 1e-30 #Avoid divide by k=0
        mu_FG = k_FG/k_i
        if mu_FG>=1: mu_FG=0.9999999999
        P_0[i] = integrate.quad(PL0, mu_FG, 1)[0]  + integrate.quad(PL0, -1, -mu_FG)[0]
        PSN_0[i] = integrate.quad(PSN0, mu_FG, 1)[0]  + integrate.quad(PSN0, -1, -mu_FG)[0]
        P_4[i] = integrate.quad(PL4, mu_FG, 1)[0]  + integrate.quad(PL4, -1, -mu_FG)[0]
        PSN_4[i] = integrate.quad(PSN4, mu_FG, 1)[0]  + integrate.quad(PSN4, -1, -mu_FG)[0]
        if doFG==True:
            if k_i<0.08: mu_FG = 0.16 #For z=0.82 l=2 small-k use constant mu_FG method
            #if k_i<0.08: mu_FG = 0.13 #For z=2.03 l=2 small-k use constant mu_FG method
            #run for NoRSD:
            #if k_i<0.08: mu_FG = 0.22 #For l=2 small-k use constant mu_FG method
        P_2[i] = integrate.quad(PL2, mu_FG, 1)[0]  + integrate.quad(PL2, -1, -mu_FG)[0]
        PSN_2[i] = integrate.quad(PSN2, mu_FG, 1)[0]  + integrate.quad(PSN2, -1, -mu_FG)[0]
    return P_0+PSN_0,P_2+PSN_2,P_4+PSN_4

def P_SN(kmod,Tbar,R_beam,P_HIshot):
    PSNeq = lambda mu: Tbar * P_HIshot * B_beam(mu,k_i,R_beam)**2
    P_shot = np.zeros(len(kmod))
    for i in range(len(kmod)):
        k_i = kmod[i]
        P_shot[i] = integrate.quad(PSNeq, -1, 1)[0]
    return P_shot
