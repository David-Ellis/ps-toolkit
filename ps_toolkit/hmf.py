import numpy as np
from scipy.interpolate import interp1d


from ps_toolkit.cosmology import Cosmology

# TODO: Take cosmology as an input
print("Loaded PS module.\n")


class PowerSpec:
    """ Create padded power spectrum
           - Constant at small k
           - Power law at large k
    """
    def __init__(self, k_data, P_data):
        # Create spline of data
        kdata = k_data[P_data>0]
        P_data = P_data[P_data>0]
        self.Pspline = interp1d(np.log10(k_data), np.log10(P_data), kind='cubic')
        
        # Get spline range
        self.klow = min(k_data)
        self.khigh = max(k_data)
        
        # Get value of P at smallest k (to become const)
        self.Plow = P_data[0]
        
        # Use last 2 point to estimate power law
        p = np.polyfit(np.log10(k_data[-2:]), np.log10(P_data[-2:]), deg=1)
        self.power_slope = p[0]
        self.power_height = p[1]
        
    def val(self, k):
        # TODO: Add ability for this to take single values
        
        output = np.zeros(len(k))
        
        # Calculate values within spline range
        spline_mask = (k >= self.klow)*(k <= self.khigh)
        output[spline_mask] = 10**self.Pspline(np.log10(k[spline_mask]))
        
        # Power spec for k below spline range = const
        small_k_mask = k < self.klow
        output[small_k_mask] = self.Plow
        
        # Power spec for k above spline range = power law
        large_k_mask = k > self.khigh
        output[large_k_mask] = 10**(self.power_slope*np.log10(k[large_k_mask]) + self.power_height)
        
        return output

def fit_PS(sigma, z, cosmo, epsilon = 1.686):
    # Press-Schechter fit
    v = cosmo.thresh(z, epsilon)/sigma
    return np.sqrt(2/np.pi)*v*np.exp(-v**2/2)

def fit_ST(sigma, z, cosmo, epsilon = 1.686):
    # Sheth-Tormen fit
    A = 0.3222; a = 0.707; p=0.3
    v = cosmo.thresh(z, epsilon)/sigma
    return A*np.sqrt(2*a/np.pi)*(1+(1/(a*v**2))**p)*v*np.exp(-v**2*a/2)

def fit_TK(sigma, z, cosmo):
    
    # TODO: get del_vir from cosmo
    
    A0 = 1.858659e-01
    a0 = 1.466904
    b0 = 2.571104
    c0 = 1.193958
    A_exp = 0.14
    a_exp = 0.06
    
    #Tinker and Kravtsov
    A = A0 * (1 + z) ** (-A_exp)
    a = a0 * (1 + z) * (-a_exp)
    
    #Omega_m = Omega_m0*(1+z)**3/(Omega_m0*(1+z**3) + Omega_r0*(1+z)**4 + 0.6911) 
    #x = Omega_m - 1
    del_vir = 200 #18*np.pi**2 + 82*x - 39*x**2
    
    alpha = 10 ** ( - ((0.75/np.log10(del_vir / 75.0))**1.2))
    
    b = b0*(1+z)**(-alpha)
    c = c0
    
    return A*((b / sigma) ** a + 1) * np.exp(- c / sigma**2)

def mass_variance(pspec, k, radii, cosmo,
                  filter_mode = 'tophat', printOutput = False):
    '''
    Calculates mass varience from the given power spectrum and returns with 
    the mass scale in units M_sol h^3
    '''
    
    masses = cosmo.rho_m0 * 4/3 * np.pi * radii ** 3 # Msol h^3 

    if printOutput == True:
        print("mass_variance: max(P) = {:.3}".format(max(pspec.val(k))))
        
    sigma = np.zeros(len(masses))
    
    if filter_mode == 'tophat':
        for i in range(len(radii)):
            W_th = 3*(np.sin(k*radii[i])-k*radii[i]*np.cos(k*radii[i]))/(k*radii[i])**3
            sigma[i] = np.sqrt(1/(2*np.pi**2)*np.trapz(pspec.val(k)*k**2*W_th**2, x=k))
    else:
        print("ERROR: Unexpected filter mode.")
        
    return sigma, masses

    
def PS_HMF(P0, 
           k, 
           input_cosmo = "planck15", 
           z=0, 
           mode = 'PS', 
           printOutput = False, 
           epsilon = 1.686, 
           krange = None):
    
    '''
    Takes power spectrum linearly evolved to z=0 and returns the HMF as predicted by 
    Press-Schechter
    
    inputs:
        Pf - Power spectrum in units Mpc^3
        k - k modes in units Mpc^(-1)
        z - desired redshift
        mode - desired fitting function:
                > PS: Press-Schechter
                > ST: Sheth-Tormen
                > TK: Tinker-Kravtov
        epsilon - fitting parameter for threshold overdensity
                  default = 1.686
        
    returns:
        HMF - Halo mass functionin dn/dlog10(M) Mpc^-3
        M - Halo masses in Msol
        f - fitting function
        sigma - Mass spectrum
        [pspec, k] - The interpolated/extrapolated power function
    '''
    if printOutput == True:
        print("Running PS calc for z = {}".format(z))
        
    cosmo = Cosmology(cosmology = input_cosmo)
        
    d = cosmo.GrowthFactor(z) / cosmo.GrowthFactor(0)
    #d = D_plus(1/(1+z))/D_plus(1)
        
    # create power spectrum object to allow for extrapolation
    pspec = PowerSpec(k, P0)
    
    if krange == None:
        k2 = np.logspace(np.log10(pspec.klow), np.log10(pspec.khigh), int(1e5))
    else:
        k2 = np.logspace(np.log10(min(krange)), np.log10(max(krange)), int(1e5))
                     
    R = np.logspace(np.log10(2*np.pi/max(k2)),
                    np.log10(2*np.pi/min(k2)), 
                    200) # h^(-1) Mpc
    
    if printOutput == True:
        print("Min R = {}, max R = {}".format(min(R), max(R)))
        
    sigma0, M = mass_variance(pspec, k2, R, cosmo, 
                              filter_mode = 'tophat',
                              printOutput = printOutput)
    
    if printOutput == True:
        print("max sig (unev) = {}".format(max(sigma0)))
    
    sigma = sigma0*d
    
    if mode == "PS":
        f = fit_PS(sigma, z, cosmo, epsilon)
    elif mode == "ST":
        f = fit_ST(sigma, z, cosmo, epsilon)
    elif mode == "TK":
        f = fit_TK(sigma, z, cosmo)
    else:
        print("ERROR: Unexpected fitting function.")
        return 0
        
    # derivative of mass varience wrt to M
    dsig_dM = abs(np.gradient(np.log(sigma), np.log(M)))
    
    # Calculate HMF#
    
    HMF = cosmo.rho_m0 * f / M * dsig_dM * np.log(10)
    
    return HMF, M, f, sigma, [pspec.val(k2)*d**2, k2]

    
def massfrac(pspec, redshifts, Mtot, mode = "PS"):
    ''' Use PS to estimate bound mass fraction
    '''
    
    assert mode == "PS" or mode == "ST", \
    '''cal_massfrac() Error: Mode not recognised. Use PS, ST. 
    TK currently not working for mass frac calculations'''
    
    Pdata, kdata = pspec
    
    HMFs = [PS_HMF(Pdata, kdata, z=z, mode = mode)[0] for z in redshifts]
    M = PS_HMF(Pdata, kdata, z=redshifts[0], mode = 'PS')[1]
    
    fb = [np.trapz(HMFs[i]*M, np.log10(M))/Mtot for i in range(len(redshifts))]
    
    return fb

def haloNums(pspec, redshifts, boxVol, cutoffmasses=[0], mode="PS"):
    ''' Use PS to estimate number of halos
    '''
    assert mode == "PS" or mode == "ST", \
    '''cal_massfrac() Error: Mode not recognised. Use PS, ST. 
    TK currently not working for halo num calculations'''
    
    Pdata, kdata = pspec
        
    HMFs = [PS_HMF(Pdata, kdata, z=z, mode = mode)[0] for z in redshifts]    
    M = PS_HMF(Pdata, kdata, 0, mode = 'PS')[1] 
   
    HaloNums = [[np.trapz(HMFs[i][M>mass]*(1e-6)**3*boxVol,
    np.log(M[M>mass])) for i in range(len(redshifts))] for mass in cutoffmasses]
    
    return HaloNums