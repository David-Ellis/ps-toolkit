import numpy as np

planck15 = {"h" : 0.6774,
            "omega_l" : 0.6911,
            "omega_m" : 0.3089,
            "omega_r" : 9.16e-5}

units = {"G" : 6.674e-11, # m^3 kg^(-1) s^(-2)
         "Msol" : 1.988e30, # kg
         "Mpc" : 3.086e22, # m 
         "Mpcpkm" : 1e3/3.0869e22
         }
#6.674×10−11 m3⋅kg−1⋅s−2.


class Cosmology:
    def __init__(self,
                 cosmology = "planck15"):
        
        # standard constants
        if cosmology == "planck15":
            self.h = planck15["h"]
            self.omega_l = planck15["omega_l"]
            self.omega_m = planck15["omega_m"]
            self.omega_r = planck15["omega_r"]
        else:
            raise Exception("Error: cosmology not implemented.")
            
        # Calculated constants
        self.a_eq = self.omega_r / self.omega_m
        self.z_eq = 1 / self.a_eq - 1
        
        self.H0 = 100 * units["Mpcpkm"] * self.h # /s
        
        # Critical density
        rhoc_kgpm3 = 3/(8 * np.pi * units["G"])*self.H0**2 
        self.rhoc0 = rhoc_kgpm3 * units["Mpc"]**3 / units["Msol"] # Msol / Mpc^(-3)
        
        # current matter and radiation densities
        self.rho_m0 = self.rhoc0 * self.omega_m / self.h ** 2
        self.rho_r0 = self.rhoc0 * self.omega_r / self.h ** 2

    def matter_dens(self, z):
        a = 1/(1+z)
        return self.rho_m0 * a**3
    
    def radiation_dens(self, z):
        a = 1/(1+z)
        return self.rho_r0 * a**4
    
    def GrowthFactor(self, z):
        a = 1/(z+1)
        x = a/self.a_eq
        return 1+2/3*x
    
    def thresh(self, z, epsilon = 1.686):
        D = self.GrowthFactor(z)
        return epsilon*D/(D-1)