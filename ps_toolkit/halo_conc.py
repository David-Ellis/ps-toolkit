# -*- coding: utf-8 -*-
"""
Module for calculating halo concentrations
"""

# Standard modules
from scipy.special import erfc
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
import numpy as np

# local modules
from ps_toolkit.cosmology import Cosmology
from ps_toolkit.hmf import PS_HMF

# Cosmological parameters
Omega_m0 = 0.3
Omega_r0 = 8.486e-5
h = 0.678 # Planck 15

z_eq = Omega_m0/Omega_r0 - 1
a_eq = 1/(1+z_eq)


def PS_Prob(z, M, z0, sig, f = 0.01, epsilon = 1.686, cosmology = "default"):
    
    '''Calculate probability of halo projenitors at z having mass greater than f*M
    where M is the mass at the final redshift z0
    
    Parameters
    ----------
    z : redshift (float) 
    M : Mass M_sol (float)
    z0 : final redshift
    siglog: interpolated function for the log of the mass variance 
    f = 0.01 (optional): fraction of total mass for progenitors (float)

    Returns
    -------
    prob : probablity (float)
    '''
    
    cosmo = Cosmology(cosmology = cosmology)
    
    sig1 = sig(f*M)
    sig2 = sig(M)
    
    
    
# test_thresh = thresh(z, epsilon)/(GrowthFactor(z)/GrowthFactor(0))-thresh(z0, epsilon)/(GrowthFactor(z0)/GrowthFactor(0))
    
    thresh1 = cosmo.thresh(z, epsilon) / ( cosmo.GrowthFactor(z) / cosmo.GrowthFactor(0)) 
    
    thresh2 = cosmo.thresh(z0, epsilon)/( cosmo.GrowthFactor(z0)/ cosmo.GrowthFactor(0))
    
    test_thresh = thresh1 - thresh2
    
    prob = erfc(test_thresh/np.sqrt(2*(sig1**2 - sig2**2)))
    
    return prob

def solve_PS_Prob(z, M, z0, siglog, f, epsilon = 1.686, 
                  cosmology = "default"):
    """ Equation to be solved in Find_zcol()
    
    Parameters
    ----------
    z : redshift (float) 
    M : Mass M_sol (float)
    z0 : final redshift
    siglog: interpolated function for the log of the mass variance 
    f (optional): fraction of total mass for progenitors (float)
    """
    return 0.5 - PS_Prob(z, M, z0, siglog, f, epsilon = epsilon, cosmology = cosmology)

def Find_zcol(Pdata, kdata, masses, z0, f = 0.01, printOutput = False, 
              krange = None, epsilon = 1.686, cosmology = "default"):
    unconverged = 0
    z_col = np.zeros(len(masses))
    
    HMF_PS, M2, f_PS, sigma0, slinedata = PS_HMF(Pdata, kdata, z=0, krange = krange, 
                                                 epsilon = epsilon, input_cosmo = cosmology)
    
    sig = interp1d(M2, sigma0, bounds_error=None, fill_value="extrapolate")
    
    if f*min(masses)<min(M2):
        print("Error: Desired mass and proj fraction too small for HMF data")
        print("min(input mass) = {} Msol".format(min(masses)))
        print("min(HMF mass) = {} Msol".format(min(M2)))
    
    for i, M in enumerate(masses):
        sol = root_scalar(solve_PS_Prob, args = (M, z0, sig, f, epsilon, cosmology), bracket=(1e7, z0))
        if sol.converged == True:
            z_col[i] = sol.root
        else:
            unconverged += 1
        if printOutput == True:
            print("{} of {} complete".format(i+1, len(masses)), end = "\r")
    if printOutput == True:
        print()
        
    return z_col

def find_conc(c, delta):
    """Equation to be solved to find concentration parameter
    as use in solve_conc().
    """
    frac = c**3/(np.log(1+c)-c/(1-c))
    return frac-delta

def solve_conc(scale_densitys):
    unconverged = 0
    concs = np.zeros(len(scale_densitys))
    for i in range(len(scale_densitys)):
        #print(i, end = ' ')
        sol = root_scalar(find_conc, args = scale_densitys[i], bracket=(1e-4, 1e8))
        if sol.converged == True:
            concs[i] = sol.root
        else:
            unconverged += 1
    return concs
    
def pred_conc(mass, C, z0, zcol = [], pspec = None, mode = "NFW",
              f = 0.01,krange = None, epsilon = 1.686, cosmology = "default"):
    
    assert (mode == "NFW") or (mode == "Bullock"), "Error: mode not recognised."
    cosmo = Cosmology(cosmology = cosmology)
    
    if zcol == []:
        assert pspec != None, \
        "Collapse redshift undefined. Therefore require power spectrum data."
        # Unpack power spectrum data
        Pdata, kdata = pspec
        
        # Estimate zcol from PS
        zcol = Find_zcol(Pdata, kdata, mass, z0, f, epsilon = epsilon, 
                         krange = krange, cosmology = cosmology)
        
    xcol = (z_eq+1)/(zcol+1); x0 = (z_eq+1)/(z0+1)
    
    if mode == "NFW":
        Omega_m = cosmo.omega_m*(1+z0)**3/(cosmo.omega_m*(1+z0**3) + cosmo.omega_r*(1+z0)**4 + cosmo.omega_l) 
        x = Omega_m - 1
        del_vir =18*np.pi**2 + 82*x - 39*x**2
     
        arg = 3*C/del_vir*(xcol**(-3) + xcol**(-4))/(x0**(-3) + x0**(-4))
        return solve_conc(arg)
        
    elif mode == "Bullock":
        return C * ((xcol**(-3) + xcol**(-4))/(x0**(-3) + x0**(-4)))**(1/3)