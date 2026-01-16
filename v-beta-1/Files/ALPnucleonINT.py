import numpy as np
from scipy.integrate import quad, dblquad, nquad, simpson, qmc_quad
from scipy.interpolate import RegularGridInterpolator

from constants import Units,  Constants
from SNprofiles import SNprofiles_QCDALP


class NuclStructFunc:
    def __init__(self, Cap, Can, profile, use_saved_sx=False, sx_file="ConfigFiles/SN-structurefunc/sx_CUSTOM.dat", Nsamples_class=1e7 ):

        """ use as profile the SNprofiles class,
        and choose if you want to compute and save the nuclear structure function (which takes time) or load one from a file """
    
        self.Cap = Cap
        self.Can = Can
        self.mn = Constants.mn
        self.crho = Constants.crho
        self.MeVtokg = Units.MeVtokg
        self.GeVtomminus1 = Units.GeVtomminus1
        self.kmminus1toMeV = Units.kmminus1toMeV
        self.profile = profile
        self.Nsamples_class = Nsamples_class

        
        # For the lapse function
        self.sx_file = sx_file
        if use_saved_sx == True:
        	self.sx_func = self._load_and_interpolate_2d(sx_file)
        else:
	        self.sx_func = None
        	self._generate_sx()
        
        
        
    @staticmethod
    def _load_and_interpolate_2d(file_path):
        """Load a file and interpolate linearly."""
        data = np.loadtxt(file_path)
        x1 = np.unique(data[:, 0])   # unique values
        x2 = np.unique(data[:, 1])
        f_x1x2 = data[:, 2].reshape(len(x1), len(x2))
        return RegularGridInterpolator((x1, x2), f_x1x2, method="linear", bounds_error=False, fill_value=None)
        
        
        

    def Cplus(self):
        return 0.5 * (self.Can + self.Cap)

    def Cminus(self):
        return 0.5 * (self.Can - self.Cap)


    """One pion exchange factor"""
    def y(self, rad):
        return Constants.mpi**2 / (self.mn * self.profile.T(rad))

    def Fplus(self, u, v, z, rad):
        return (u + v + 2 * (u * v)**0.5 * z) / (u + v + 2 * (u * v)**0.5 * z + self.y(rad))

    def Fminus(self, u, v, z, rad):
        return (u + v - 2 * (u * v)**0.5 * z) / (u + v - 2 * (u * v)**0.5 * z + self.y(rad))
        
    @staticmethod
    def xi(u, v, z):
        return 3 * (u - v)**2 / ((u + v)**2 - (4 * u * v * z**2))


    """Rho meson exchange"""
    def rrho(self, rad):
        return Constants.mrho**2 / (self.mn * self.profile.T(rad))

    def Gplus(self, u, v, z, rad):
        return (u + v + 2 * (u * v)**0.5 * z) / (u + v + 2 * (u * v)**0.5 * z + self.rrho(rad))

    def Gminus(self, u, v, z, rad):
        return (u + v - 2 * (u * v)**0.5 * z) / (u + v - 2 * (u * v)**0.5 * z + self.rrho(rad))

    def Fhatplus(self, u, v, z, rad):
        return self.Fplus(u, v, z, rad) - self.crho * self.Gplus(u, v, z, rad)

    def Fhatminus(self, u, v, z, rad):
        return self.Fminus(u, v, z, rad) - self.crho * self.Gminus(u, v, z, rad)
        
        
    """NO non-degenerate approx functions"""
    def Huplus(self, w, u, delta, eta):
        return 1 / (np.exp((w + u) / 2 + np.sqrt(u * w) * np.cos(delta) - eta) + 1)
    
    def Huminus(self, w, u, delta, eta):
        return 1 / (np.exp((w + u) / 2 - np.sqrt(u * w) * np.cos(delta) - eta) + 1)
    
    def Hvplus(self, w, v, delta, phi, z, eta):
        return 1 / (np.exp((w + v) / 2 + np.sqrt(v * w) * ((np.sin(delta) * np.sqrt(1 - z**2) * np.cos(phi)) + (np.cos(delta) * z)) - eta) + 1)
    
    def Hvminus(self, w, v, delta, phi, z, eta):
        return 1 / (np.exp((w + v) / 2 - np.sqrt(v * w) * ((np.sin(delta) * np.sqrt(1 - z**2) * np.cos(phi)) + (np.cos(delta) * z)) - eta) + 1)
        
        
        
    """Structure functions """
    def Fdensity(self, rad, particle):
        return (self.profile.rho(rad) * self.profile.Yparticle(rad, particle)) / (2 * self.mn) * ((2 * np.pi) / (self.mn * self.profile.T(rad)))**(3/2)
        
        
    def integrate_MC(self, function, xa, Nsamples=None):
        """ SIMPSON works better than MC here """
        """ This integration method is good enough but can be improved """
        """ Nsamples: number of samples """
        if Nsamples is None:
            Nsamples = self.Nsamples_class
            
        # Integration limits
        delta_min, delta_max = 0, np.pi
        phi_min, phi_max = 0, 2*np.pi
        w_min, w_max = np.log10(0.01), np.log10(100)
        u_min, u_max = np.log10(xa+0.01), np.log10(100)
        z_min, z_max = -1, 1
        
        Nperdim = int( Nsamples**(1/5) )  # points per dimension

	# random points
        delta_vals = np.random.uniform(delta_min, delta_max, Nperdim)
        phi_vals = np.random.uniform(phi_min, phi_max, Nperdim)
        w_vals = np.logspace(w_min, w_max, Nperdim)
        u_vals = np.logspace(u_min, u_max, Nperdim)
        z_vals = np.random.uniform(z_min, z_max, Nperdim)

	# evaluate the function in those random points
        F_vals = function(delta_vals, phi_vals, w_vals, u_vals, z_vals)

	# integration volume
        volume = (delta_max - delta_min) * (phi_max - phi_min) * (w_max - w_min) * (u_max - u_min) * (z_max - z_min)

	# Approximate the integral
        integral_value = volume * np.mean(F_vals)

        return integral_value
        
        
    def integrate_simpson(self, function, xa, Nsamples=None):
        """ SIMPSON works better than MC here """
        """ This integration method is good enough but can be improved """
        """ Nsamples: number of samples """
        if Nsamples is None:
            Nsamples = self.Nsamples_class
            
        # Integration limits
        delta_min, delta_max = 0, np.pi
        phi_min, phi_max = 0, 2*np.pi
        w_min, w_max = np.log10(0.01), np.log10(100)
        u_min, u_max = np.log10(xa+0.01), np.log10(100)
        z_min, z_max = -1, 1
        
        # Grid of points
        Nperdim = int( Nsamples**(1/5) )  # points per dimension
	
        delta_vals = np.linspace(delta_min, delta_max, Nperdim)
        phi_vals = np.linspace(phi_min, phi_max, Nperdim)
        w_vals = np.logspace(w_min, w_max, Nperdim)
        u_vals = np.logspace(u_min, u_max, Nperdim)
        z_vals = np.linspace(z_min, z_max, Nperdim)
	
        # Create a single mesh
        D, P, W, U, Z = np.meshgrid(delta_vals, phi_vals, w_vals, u_vals, z_vals, indexing='ij')

        # evaluate the function in those points
        F_vals = function(D, P, W, U, Z)

        # Apply Simpson's rule to each dimension
        integral_value = simpson(F_vals, x=delta_vals, axis=0)
        integral_value = simpson(integral_value, x=phi_vals, axis=0)
        integral_value = simpson(integral_value, x=w_vals, axis=0)
        integral_value = simpson(integral_value, x=u_vals, axis=0)
        integral_value = simpson(integral_value, x=z_vals, axis=0)
	
        return integral_value
        
        
    
    def integrate_nquad(self, function, xa):
        """ Use scipy.integrate.nquad """
    
        # Define the integration limits
        limits = [
                (0, np.pi),  # delta
                (0, 2*np.pi),  # phi
                (0.01, 100),  # w
                (xa + 0.01, 100),  # u
                (-1, 1)  # z
        ]

        # Integrate
        integral_value, error = nquad(function, limits, opts={'epsabs': 1e-1, 'epsrel': 1e-1})
    
        return integral_value
    
    
    def sk(self, xa, rad, particle1, particle2, particle3, particle4, Nsamples=None):
    	# Nsamples: number of samples
    	if Nsamples is None:
            Nsamples = self.Nsamples_class
        
    	def integrand(delta, phi, w, u, z):
            v = u - xa
            aux = (
                np.sin(delta) * np.sqrt(w) * np.sqrt(u * v) *
                np.exp(w - self.profile.etaparticle(rad, particle3)) *
                np.exp(u - self.profile.etaparticle(rad, particle4)) *
                self.Huplus(w, u, delta, self.profile.etaparticle(rad, particle1)) *
                self.Huminus(w, u, delta, self.profile.etaparticle(rad, particle2)) *
                self.Hvplus(w, v, delta, phi, z, self.profile.etaparticle(rad, particle3)) *
                self.Hvminus(w, v, delta, phi, z, self.profile.etaparticle(rad, particle4)) *
                self.Fhatminus(u, v, z, rad)**2
            )
            return (1 / (4 * np.pi * np.sqrt(np.pi))) * (self.Fdensity(rad, particle1)**-1) * (self.Fdensity(rad, particle2)**-1) * aux
        
    	#return np.nan_to_num(self.integrate_MC(integrand, xa, Nsamples), nan=0.0)
    	return np.nan_to_num(self.integrate_simpson(integrand, xa, Nsamples), nan=0.0)
    	#return np.nan_to_num(self.integrate_nquad(integrand, xa), nan=0.0)
               
               
    def sl(self, xa, rad, particle1, particle2, particle3, particle4, Nsamples=None):
    	# Nsamples: number of samples
    	if Nsamples is None:
            Nsamples = self.Nsamples_class
            
    	def integrand(delta, phi, w, u, z):
            v = u - xa
            aux = (
                np.sin(delta) * np.sqrt(w) * np.sqrt(u * v) *
                np.exp(w - self.profile.etaparticle(rad, particle3)) *
                np.exp(u - self.profile.etaparticle(rad, particle4)) *
                self.Huplus(w, u, delta, self.profile.etaparticle(rad, particle1)) *
                self.Huminus(w, u, delta, self.profile.etaparticle(rad, particle2)) *
                self.Hvplus(w, v, delta, phi, z, self.profile.etaparticle(rad, particle3)) *
                self.Hvminus(w, v, delta, phi, z, self.profile.etaparticle(rad, particle4)) *
                self.Fhatplus(u, v, z, rad)**2
            )
            return (1 / (4 * np.pi * np.sqrt(np.pi))) * (self.Fdensity(rad, particle1)**-1) * (self.Fdensity(rad, particle2)**-1) * aux
        
    	#return np.nan_to_num(self.integrate_MC(integrand, xa, Nsamples), nan=0.0)
    	return np.nan_to_num(self.integrate_simpson(integrand, xa, Nsamples), nan=0.0)
               
           
    def skl(self, xa, rad, particle1, particle2, particle3, particle4, Nsamples=None):
    	# Nsamples: number of samples
    	if Nsamples is None:
            Nsamples = self.Nsamples_class
            
    	def integrand(delta, phi, w, u, z):
            v = u - xa
            aux = (
                np.sin(delta) * np.sqrt(w) * np.sqrt(u * v) *
                np.exp(w - self.profile.etaparticle(rad, particle3)) *
                np.exp(u - self.profile.etaparticle(rad, particle4)) *
                self.Huplus(w, u, delta, self.profile.etaparticle(rad, particle1)) *
                self.Huminus(w, u, delta, self.profile.etaparticle(rad, particle2)) *
                self.Hvplus(w, v, delta, phi, z, self.profile.etaparticle(rad, particle3)) *
                self.Hvminus(w, v, delta, phi, z, self.profile.etaparticle(rad, particle4)) *
                self.Fhatplus(u, v, z, rad) * self.Fhatminus(u, v, z, rad)
            )
            return (1 / (4 * np.pi * np.sqrt(np.pi))) * (self.Fdensity(rad, particle1)**-1) * (self.Fdensity(rad, particle2)**-1) * aux
        
    	#return np.nan_to_num(self.integrate_MC(integrand, xa, Nsamples), nan=0.0)
    	return np.nan_to_num(self.integrate_simpson(integrand, xa, Nsamples), nan=0.0)
               
           
    def skxl(self, xa, rad, particle1, particle2, particle3, particle4, Nsamples=None):
    	# Nsamples: number of samples
    	if Nsamples is None:
            Nsamples = self.Nsamples_class
            
    	def integrand(delta, phi, w, u, z):
            v = u - xa
            aux = (
                np.sin(delta) * np.sqrt(w) * np.sqrt(u * v) *
                np.exp(w - self.profile.etaparticle(rad, particle3)) *
                np.exp(u - self.profile.etaparticle(rad, particle4)) *
                self.Huplus(w, u, delta, self.profile.etaparticle(rad, particle1)) *
                self.Huminus(w, u, delta, self.profile.etaparticle(rad, particle2)) *
                self.Hvplus(w, v, delta, phi, z, self.profile.etaparticle(rad, particle3)) *
                self.Hvminus(w, v, delta, phi, z, self.profile.etaparticle(rad, particle4)) *
                (self.xi(u, v, z)/3) * self.Fhatplus(u, v, z, rad) * self.Fhatminus(u, v, z, rad)
            )
            return (1 / (4 * np.pi * np.sqrt(np.pi))) * (self.Fdensity(rad, particle1)**-1) * (self.Fdensity(rad, particle2)**-1) * aux
        
    	#return np.nan_to_num(self.integrate_MC(integrand, xa, Nsamples), nan=0.0)
    	return np.nan_to_num(self.integrate_simpson(integrand, xa, Nsamples), nan=0.0)
               
               
               
               
    def snn(self, xa, rad, Nsamples=None):
    	if Nsamples is None:
            Nsamples = self.Nsamples_class
    	return (
            (1/3) * self.Can**2 * self.profile.Yn(rad)**2 * 
            (self.sk(xa, rad, "n", "n", "n", "n", Nsamples) + self.sl(xa, rad, "n", "n", "n", "n", Nsamples) + self.skl(xa, rad, "n", "n", "n", "n", Nsamples) - 3 * self.skxl(xa, rad, "n", "n", "n", "n", Nsamples))
        ) 
        
    def spp(self, xa, rad, Nsamples=None):
    	if Nsamples is None:
            Nsamples = self.Nsamples_class
    	return (
            (1/3) * self.Cap**2 * self.profile.Yp(rad)**2 * 
            (self.sk(xa, rad, "p", "p", "p", "p", Nsamples) + self.sl(xa, rad, "p", "p", "p", "p", Nsamples) + self.skl(xa, rad, "p", "p", "p", "p", Nsamples) - 3 * self.skxl(xa, rad, "p", "p", "p", "p", Nsamples))
        ) 
        
    def snp(self, xa, rad, Nsamples=None):
    	if Nsamples is None:
            Nsamples = self.Nsamples_class
    	return (
            ((4/3) * self.profile.Yn(rad) * self.profile.Yp(rad) * (self.Cplus()**2 + self.Cminus()**2) * self.sk(xa, rad, "n", "p", "n", "p", Nsamples) ) + 
            ((4/3) * self.profile.Yn(rad) * self.profile.Yp(rad) * (4 * self.Cplus()**2 + 2 * self.Cminus()**2) * self.sl(xa, rad, "n", "p", "n", "p", Nsamples) ) - 
            ((8/3) * self.profile.Yn(rad) * self.profile.Yp(rad) * ( ( (self.Cplus()**2 + self.Cminus()**2) * self.skl(xa, rad, "n", "p", "n", "p", Nsamples) ) - ( (3 * self.Cplus()**2 + self.Cminus()**2) * self.skxl(xa, rad, "n", "p", "n", "p", Nsamples) ) ) )
        ) 
    
    
    
    def s(self, xa, rad, Nsamples=None):
    	if Nsamples is None:
            Nsamples = self.Nsamples_class
    	return self.snn(xa, rad, Nsamples) + self.spp(xa, rad, Nsamples) + self.snp(xa, rad, Nsamples)
        
        

    def sER(self, Ea, rad, Nsamples=None):
    	if Nsamples is None:
            Nsamples = self.Nsamples_class
    	return self.s(Ea / self.profile.T(rad), rad, Nsamples)
        
        
    def _generate_sx(self):
        """
        Computes and returns an interpolated structure function s(Ea,r).
        The structure function is calculated for a range of energies and radii and then interpolated
        to allow continuous evaluation at any value.
        """
        
        Ea_vals = np.linspace(1, 300, 100)  # Range from Ea=1 to 300 MeV
        rad_vals = np.linspace(0, 20, 21)  # Range from r=0 to 20 km
        
        # Create a meshgrid
        Ea_grid, rad_grid = np.meshgrid(Ea_vals, rad_vals, indexing="ij")

        # Evaluate the function s(Ea, rad)
        #s_vals = self.sER(Ea_grid, rad_grid, self.Nsamples_SX)
        s_vals = np.zeros_like(Ea_grid)
        for i in range(Ea_grid.shape[0]):
        	for j in range(Ea_grid.shape[1]):
        		s_vals[i, j] = self.sER(Ea_grid[i, j], rad_grid[i, j], self.Nsamples_class)


        # Save the structure funcion
        data_tosave = np.column_stack((Ea_grid.ravel(), rad_grid.ravel(), s_vals.ravel()))
        np.savetxt(self.sx_file, data_tosave, fmt="%.6e", header="# Ea rad s(Ea,rad)", comments='')
        print("structure function s(Ea,rad) grid saved in: ", self.sx_file)

        # Interpolation of the lapse function
        self.sx_func = RegularGridInterpolator((Ea_vals, rad_vals), s_vals, method="linear", bounds_error=False, fill_value=None)
        
        
        
        
        
        

class ALP_Brems:
    def __init__(self, profile, NuclStruct):
    
        """use as profile the SNprofiles class, 
        use the nuclear structure function from the NuclStructFunc class"""
        
        self.profile = profile
        self.NuclStruct = NuclStruct
        
        self.mn = Constants.mn
        self.kmtoMeVminus1 = 1/Units.kmminus1toMeV
        self.MeVtokg = Units.MeVtokg
        self.GeVtomminus1 = Units.GeVtomminus1


    def Gsigma(self, rad):
        """ nucleon spin fluctuation rate, MeV """
        return 21.6 * (self.profile.rho(rad) / (10**14 * 10**3 * 1000**3 / (self.MeVtokg * self.GeVtomminus1**3))) * (self.profile.T(rad))**0.5
        
    def Ssigma(self, Ea, rad, g):
        """ Suppresion factor sigma """
        return self.Gsigma(rad) / (Ea**2 + (g * 0.5 * self.Gsigma(rad))**2) * self.NuclStruct.sx_func([[Ea, rad]])
        
    def dndEdt(self, Ea, rad, ma, ga):
        """ ALP spectrum per unit volume dn/dEadt, MeV**3 """
        return np.where(Ea < ma, 0, 
                        (ga**2 / (16 * np.pi**2) * self.profile.nbF(rad) / self.mn**2 * 
                         (Ea**2 - ma**2)**(3/2) * np.exp(-Ea / self.profile.T(rad)) * 
                         self.Ssigma(Ea, rad, 0.2)))
                         
                         
    def dndEdtBrem_X(self, Ea, rad, ma, ga):
        """ ALP spectrum per unit volume dn/dEadt, MeV**3 """
        return np.where(Ea < ma, 0, 
                        (ga**2 / (16 * np.pi**2) * self.profile.nbF(rad) / self.mn**2 * 
                         (Ea**2 - ma**2)**(3/2) * 
                         self.Ssigma(Ea, rad, 0.2)))



    def lambdaam1_X(self, Ea, rad, ma, ga):
        """ ALP mean free path ^-1, MeV """
        return np.where(Ea < ma, 0, 
                        (2 * np.pi**2) / (Ea**2 - ma**2) * 
                        self.dndEdtBrem_X(Ea, rad, ma, ga))
                        
    def lambdaam1(self, Ea, rad, ma, ga):
        """ ALP mean free path ^-1, MeV """
        return np.where(Ea < ma, 0, 
                        (2 * np.pi**2) / (Ea**2 - ma**2) * 
                        self.dndEdt(Ea, rad, ma, ga) *
                        np.exp(Ea / self.profile.T(rad)) )
                        
                        
                        
class ALP_Pion:
    def __init__(self, Cap, Can, profile):
    	self.Cap = Cap
    	self.Can = Can
    	self.profile = profile
    	self.mpi = Constants.mpi
    	self.GammaDelta = Constants.GammaDelta
    	self.Dm = Constants.mDelta - Constants.mn  # Delta-resonance - nucleon mass difference
    	self.gAX = Constants.gAX
    	self.mn = Constants.mn
    	self.fpi = Constants.fpi
    	self.MeVtokg = Units.MeVtokg
    	self.GeVtomminus1 = Units.GeVtomminus1



    def Cplus(self):
        return 0.5 * (self.Can + self.Cap)

    def Cminus(self):
        return 0.5 * (self.Can - self.Cap)
        
    def CapiN(self):
        """ axion-pion-nucleon coupling CapiN """
        return (self.Cap - self.Can) / (np.sqrt(2) * self.gAX)

    def CaNDelta(self):
        """ axion-delta-nucleon coupling CaNDelta """
        return -np.sqrt(3) / 2 * (self.Cap - self.Can)


        
    def Epi(self, Ea):
    	""" pion energy, 
    	pag 5 of arxiv.org/pdf/2309.14798:
    	because of energy conservation Epi = Ea """
    	return Ea
    	
    def ppi(self, Ea):
    	""" pion momentum """
    	return np.where(self.Epi(Ea) < self.mpi, 0, np.sqrt( self.Epi(Ea)**2 - self.mpi**2 ) )
    	
    def BetaAxion(self, Ea, ma):
    	""" axion beta velocity factor, adimensional """
    	return np.where(Ea < ma, 0, np.sqrt((Ea**2 - ma**2)) / Ea )



    def Daux(self, Ea):
        """ MeV**4 """
        return ((self.Dm - self.Epi(Ea))**2 + (self.GammaDelta**2) / 4) * ((self.Dm + self.Epi(Ea))**2 + (self.GammaDelta**2) / 4)

    def FapiN(self, Ea):
        """ adimensional """
        return (1 / self.Daux(Ea)) * (self.Epi(Ea)**2 * (self.Dm**2 + 2 * self.Epi(Ea)**2 + self.GammaDelta**2 / 4))

    def FaNDelta(self, Ea):
        """ adimenisonal """
        return (1 / self.Daux(Ea)) * self.Epi(Ea) * (
                (self.Dm**2 - self.Epi(Ea)**2) * (self.Cplus() * self.Dm + self.Cminus() * self.Epi(Ea)) +
                (self.GammaDelta**2 / 4) * (self.Cplus() * self.Dm - self.Cminus() * self.Epi(Ea))
                )

    def Ga(self, Ea):
        """ adimensional """
        term1 = (2 * self.gAX**2 * ((2 * self.Cplus()**2) + self.Cminus()**2)) / 3 * (self.ppi(Ea) / self.mn)**2
        term2 = self.CapiN()**2 * (self.Epi(Ea) / self.mn)**2
        term3 = (self.gAX**2 * self.CaNDelta()**2) / 9 * self.FapiN(Ea) * (self.ppi(Ea) / self.mn)**2
        term4 = (-4 * np.sqrt(3) * self.gAX**2 * self.CaNDelta()) / 9 * self.FaNDelta(Ea) * ((self.ppi(Ea) / self.mn)**2)

        return term1 + term2 + term3 + term4

    def Cmatrix(self, Ea, ma):
        """ MeV**2 """
        return (self.mn**2 / self.gAX**2) * self.BetaAxion(Ea, ma)**2 * self.Ga(Ea)
        
        

    def Gsigma(self, rad):
        """ nucleon spin fluctuation rate, MeV """
        return 21.6 * (self.profile.rho(rad) / (10**14 * 10**3 * 1000**3 / (self.MeVtokg * self.GeVtomminus1**3))) * (self.profile.T(rad))**0.5
        
        
        
    def IntegralAUX(self, rad, particleIN, particleOUT):
        """ adimensional """
        if (self.Cap == 0 and particleIN == particleOUT == "p") or (self.Can == 0 and particleIN == particleOUT == "n"):
            return 0
        etap_in, etap_out = self.profile.etaparticle(rad, particleIN), self.profile.etaparticle(rad, particleOUT)
        integrand = lambda yy: yy**2 / ( (np.exp(yy**2 - etap_in) + 1) * (np.exp(-(yy**2) + etap_out) + 1) )
        return quad(integrand, 0, np.inf, limit=100, epsrel=1e-4)[0]

    def dndEdt(self, Ea, rad, ma, ga, particleIN="p", particleOUT="n"):
        """ ALP spectrum per unit volume dn/dEadt, MeV**3 """        
        
        """ Only processes involving negatively charged pions are relevant for ALP production. 
        The abundance of pion+ and pion0 inside a SN core is strongly suppressed with respect to pion− (arxiv.org/pdf/2306.01048) """
        
        """ pion- + p -> ALP + n """
        return np.where(Ea < max(ma, self.mpi), 0, ((ga**2 * self.profile.T(rad)**1.5) / (2**1.5 * np.pi**5 * self.mn**0.5) * (self.gAX / (2 * self.fpi))**2 *
                np.sqrt(Ea**2 - ma**2) * self.Cmatrix(Ea, ma) *
                1 / (np.exp(Ea / self.profile.T(rad) - self.mpi / self.profile.T(rad) - self.profile.etapi(rad)) - 1) *
                np.sqrt(Ea**2 - self.mpi**2) * Ea**2 / (Ea**2 + (self.Gsigma(rad) / 2)**2) *
                self.IntegralAUX(rad, particleIN, particleOUT)) )


    ####  For the absorption  ####
    def dndEdtPIONminABS(self, Ea, rad, ma, ga):
        """ ALP spectrum per unit volume dn/dEadt, MeV**3 """        
        """ ALP + n -> pion- + p """
        return np.where(Ea < max(ma, self.mpi), 0, ((ga**2 * self.profile.T(rad)**1.5) / (2**1.5 * np.pi**5 * self.mn**0.5) * (self.gAX / (2 * self.fpi))**2 *
                np.sqrt(Ea**2 - ma**2) * self.Cmatrix(Ea, ma) *
                ( 1 + 1 / (np.exp(Ea / self.profile.T(rad) - self.mpi / self.profile.T(rad) - self.profile.etapi(rad)) - 1) ) *
                np.sqrt(Ea**2 - self.mpi**2) * Ea**2 / (Ea**2 + (self.Gsigma(rad) / 2)**2) *
                self.IntegralAUX(rad, "n", "p")) )
                
    def dndEdtPIONplusABS(self, Ea, rad, ma, ga):
        """ ALP spectrum per unit volume dn/dEadt, MeV**3 """        
        """ ALP + p -> pion+ + n """
        return np.where(Ea < max(ma, self.mpi), 0, ((ga**2 * self.profile.T(rad)**1.5) / (2**1.5 * np.pi**5 * self.mn**0.5) * (self.gAX / (2 * self.fpi))**2 *
                np.sqrt(Ea**2 - ma**2) * self.Cmatrix(Ea, ma) *
                1 *
                np.sqrt(Ea**2 - self.mpi**2) * Ea**2 / (Ea**2 + (self.Gsigma(rad) / 2)**2) *
                self.IntegralAUX(rad, "p", "n")) )
                
                
    def GaNeutral(self, Ea):
        """ adimensional """
        term1 = (2 * self.gAX**2 * ((2 * self.Cplus()**2) + self.Cminus()**2)) / 3 * (self.ppi(Ea) / self.mn)**2

        return (1/2) * term1

    def CmatrixNeutral(self, Ea, ma):
        """ MeV**2 """
        return (self.mn**2 / self.gAX**2) * self.BetaAxion(Ea, ma)**2 * self.GaNeutral(Ea)
                
    def dndEdtPIONneutralABS(self, Ea, rad, ma, ga):
        """ ALP spectrum per unit volume dn/dEadt, MeV**3 """        
        """ ALP + n -> pion0 + n """
        """ ALP + p -> pion0 + p """
        """ this case only compton-like diagrams -> we can consider CaNDelta = 0, CapiN = 0 (see appendix 2306.01048) """
        return np.where(Ea < max(ma, self.mpi), 0, ((ga**2 * self.profile.T(rad)**1.5) / (2**1.5 * np.pi**5 * self.mn**0.5) * (self.gAX / (2 * self.fpi))**2 *
                np.sqrt(Ea**2 - ma**2) * self.CmatrixNeutral(Ea, ma) *
                1 *
                np.sqrt(Ea**2 - self.mpi**2) * Ea**2 / (Ea**2 + (self.Gsigma(rad) / 2)**2) *
                (self.IntegralAUX(rad, "n", "n") + self.IntegralAUX(rad, "p", "p")) ) )




    def lambdaam1(self, Ea, rad, ma, ga):
        """ ALP mean free path ^-1, MeV """
        return np.where(Ea < self.mpi, 0,
        	np.array([ (2 * np.pi**2) / (Ea**2 - ma**2) * (
                self.dndEdtPIONminABS(Ea, rad, ma, ga) +
                self.dndEdtPIONplusABS(Ea, rad, ma, ga) +
                self.dndEdtPIONneutralABS(Ea, rad, ma, ga)) ]) )




class ALP_Photon:
    def __init__(self, Cap, profile):
    	self.Cap = Cap
    	self.profile = profile
    	self.ee = Constants.ee
    	
    def f_gamma(self, Egamma, rad):
        """ Bose-Einstein distribution function """
        return 1 / (np.exp(Egamma / self.profile.T(rad)) - 1)


    def M2prod(self, Egamma, Ea, rad, ma, ga):
        """ Averaged squared amplitude """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        mn_star = self.profile.mnstar(rad)  # Effective nucleon mass

        term1 = (-4 * Ea**2 * mn_star + ma**2 * (-Egamma + mn_star) + 2 * Ea * (ma**2 + Egamma * mn_star)) / (ma**2 - 2 * Ea * mn_star)**2
        term2 = (Ea * m_gamma**2) / (ma**2 - 2 * Ea * mn_star)**2
        term3 = (ma**2 * (-Egamma + mn_star)) / (2 * Egamma * mn_star + m_gamma**2)**2
        term4 = (-Ea * ma**2 - 2 * Ea**2 * mn_star + 2 * ma**2 * mn_star) / ((ma**2 - 2 * Ea * mn_star) * (2 * Egamma * mn_star + m_gamma**2))

        return (4/3) * self.Cap**2 * self.ee**2 * ga**2 * mn_star * (term1 + term2 + term3 + term4)
        

    def dsigma_dEa(self, Egamma, Ea, rad, ma, ga):
        """ Differential cross section """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        mn_star = self.profile.mnstar(rad)  # Effective nucleon mass
        return (1 / (32 * np.pi * mn_star * (Egamma**2 - m_gamma**2))) * self.M2prod(Egamma, Ea, rad, ma, ga)


    def costheta0prod(self, Egamma, Ea, rad, ma):
        """ angle """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        mn_star = self.profile.mnstar(rad)  # Effective nucleon mass
        
        num = (Ea * Egamma - (-Ea + Egamma) * mn_star + 0.5 * (-ma**2 - m_gamma**2))
        den = np.sqrt(Ea**2 - ma**2) * np.sqrt(Egamma**2 - m_gamma**2)
        return num / den
                
    def EgammaMinp(self, Ea, rad, ma):
        """ Minimum photon energy to produce and ALP with Ea and ma """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        mn_star = self.profile.mnstar(rad)  # Effective nucleon mass
        
        num = (-2 * mn_star * Ea**2 + ma**2 * Ea + 2 * mn_star**2 * Ea + m_gamma**2 * Ea - mn_star * m_gamma**2 - ma**2 * mn_star)
        #den = np.where( Ea < (ma**2 + mn_star**2)/(2 * mn_star), 2 * (ma**2 + mn_star**2 - 2 * Ea * mn_star), 2 * (ma**2 + mn_star**2 + 2 * Ea * mn_star) )
        den = (2 * abs(ma**2 + mn_star**2 - 2 * Ea * mn_star))
    
        rad = (-ma**6 + Ea**2 * ma**4 + 2 * m_gamma**2 * ma**4 + 4 * Ea * mn_star * ma**4 - 
               m_gamma**4 * ma**2 - 4 * Ea**2 * mn_star**2 * ma**2 - 2 * Ea**2 * m_gamma**2 * ma**2 + 
               4 * mn_star**2 * m_gamma**2 * ma**2 - 4 * Ea * mn_star * m_gamma**2 * ma**2 - 4 * Ea**3 * mn_star * ma**2 + 
               Ea**2 * m_gamma**4 + 4 * Ea**4 * mn_star**2 - 4 * Ea**2 * mn_star**2 * m_gamma**2 + 4 * Ea**3 * mn_star * m_gamma**2)
    
        min_aux = num / den - np.sqrt(rad) / (2 * abs(ma**2 + mn_star**2 - 2 * Ea * mn_star))
        
        return np.where(min_aux < 0, Ea, min_aux) 

    def EgammaMaxp(self, Ea, rad, ma):
        """ Maximum photon energy to produce and ALP with Ea and ma """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        mn_star = self.profile.mnstar(rad)  # Effective nucleon mass
        
        num = (-2 * mn_star * Ea**2 + ma**2 * Ea + 2 * mn_star**2 * Ea + m_gamma**2 * Ea - mn_star * m_gamma**2 - ma**2 * mn_star)
        #den = np.where( Ea < (ma**2 + mn_star**2)/(2 * mn_star), 2 * (ma**2 + mn_star**2 - 2 * Ea * mn_star), 2 * (ma**2 + mn_star**2 + 2 * Ea * mn_star) )
        den = (2 * abs(ma**2 + mn_star**2 - 2 * Ea * mn_star))
    
        rad = (-ma**6 + Ea**2 * ma**4 + 2 * m_gamma**2 * ma**4 + 4 * Ea * mn_star * ma**4 - 
               m_gamma**4 * ma**2 - 4 * Ea**2 * mn_star**2 * ma**2 - 2 * Ea**2 * m_gamma**2 * ma**2 + 
               4 * mn_star**2 * m_gamma**2 * ma**2 - 4 * Ea * mn_star * m_gamma**2 * ma**2 - 4 * Ea**3 * mn_star * ma**2 + 
               Ea**2 * m_gamma**4 + 4 * Ea**4 * mn_star**2 - 4 * Ea**2 * mn_star**2 * m_gamma**2 + 4 * Ea**3 * mn_star * m_gamma**2)
    
        max_aux = num / den + np.sqrt(rad) / (2 * abs(ma**2 + mn_star**2 - 2 * Ea * mn_star))
        
        return np.where(max_aux < 0, Ea, max_aux) 


    def dndEdt(self, Ea, rad, ma, ga):
        """ Spectrum of ALPs produced in p gamma -> p a """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        npeff = self.profile.npeff(rad)  # Effective number density of protons

        def integrand(Egamma):
            return (self.f_gamma(Egamma, rad) *
                    (Egamma**2 - m_gamma**2) *
                    self.dsigma_dEa(Egamma, Ea, rad, ma, ga))

        Egamma_min = self.EgammaMinp(Ea, rad, ma)
        Egamma_max = self.EgammaMaxp(Ea, rad, ma)

        return np.where(Ea < ma, 0, 
                        (3 / (2 * np.pi**2)) * npeff * quad(integrand, Egamma_min, Egamma_max, epsrel=1e-4)[0] )
                        
                        
    def dndEdt_ABS(self, Ea, rad, ma, ga):
        """ Spectrum of ALPs produced in p gamma -> p a """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        npeff = self.profile.npeff(rad)  # Effective number density of protons

        def integrand(Egamma):
            return ( (1 + self.f_gamma(Egamma, rad)) *
                    (Egamma**2 - m_gamma**2) *
                    self.dsigma_dEa(Egamma, Ea, rad, ma, ga))

        Egamma_min = self.EgammaMinp(Ea, rad, ma)
        Egamma_max = self.EgammaMaxp(Ea, rad, ma)

        return np.where(Ea < ma, 0, 
                        (3 / (2 * np.pi**2)) * npeff * quad(integrand, Egamma_min, Egamma_max, epsrel=1e-4)[0] )
                        
    
    def lambdaam1(self, Ea, rad, ma, ga):
        """ ALP mean free path ^-1, MeV """
        return np.where(Ea < ma, 0, 
                        np.array([ (2 * np.pi**2) / (Ea**2 - ma**2) * 
                        self.dndEdt_ABS(Ea, rad, ma, ga) ]) )
                        
                        
                        
                        
class ALP_Dipole:
    " 	https://arxiv.org/pdf/2203.15812 "
    def __init__(self, Capgamma, Cangamma, profile):
    	self.Capgamma= Capgamma
    	self.Cangamma= Cangamma
    	self.profile = profile
    	self.ee = Constants.ee
    	self.mn = Constants.mn
    	
    def gdp(self, ga):
        """ dipole coupling as a function of ga, with protons, MeV^{-2} """
        return ga * self.Capgamma / (self.mn)**2
        
    def gdn(self, ga):
        """ dipole coupling as a function of ga, with neutrons, MeV^{-2} """
        return ga * self.Cangamma / (self.mn)**2
           
    def gaa(self, gdd):
        """ ga as a function of the dipole coupling [MeV^{-2}] """
        return gdd * (self.mn)**2 / self.Cangamma
        
    
    def f_gamma(self, Egamma, rad):
        """ Bose-Einstein distribution function """
        #return 1 / (np.exp(Egamma / self.profile.T(rad)) - 1)
        return 1 / (np.exp(np.clip(Egamma / self.profile.T(rad), -100, 100)) - 1)
        
        
    def M2prod(self, Egamma, Ea, rad, ma, ga):
        """ Averaged squared amplitude """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        mn_star = self.profile.mnstar(rad)  # Effective nucleon mass
        J = 1/2

        num_p = 2 * self.Capgamma**2 * ga**2 * (-2 * Egamma * (ma**2 - 2 * Ea * mn_star) + (Ea + Egamma + 2 * mn_star) * m_gamma**2)
        num_n = 2 * self.Cangamma**2 * ga**2 * (-2 * Egamma * (ma**2 - 2 * Ea * mn_star) + (Ea + Egamma + 2 * mn_star) * m_gamma**2)
        den = 3 * (1 + 2 * J) * mn_star**3

        return (num_p + num_n) / den
        

    def dsigma_dEa(self, Egamma, Ea, rad, ma, ga):
        """ Differential cross section """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        mn_star = self.profile.mnstar(rad)  # Effective nucleon mass
        return (1 / (32 * np.pi * mn_star * (Egamma**2 - m_gamma**2))) * self.M2prod(Egamma, Ea, rad, ma, ga)


    def costheta0prod(self, Egamma, Ea, rad, ma):
        """ angle """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        mn_star = self.profile.mnstar(rad)  # Effective nucleon mass
        
        num = (Ea * Egamma - (-Ea + Egamma) * mn_star + 0.5 * (-ma**2 - m_gamma**2))
        den = np.sqrt(Ea**2 - ma**2) * np.sqrt(Egamma**2 - m_gamma**2)
        return num / den
                
    def EgammaMinp(self, Ea, rad, ma):
        """ Minimum photon energy to produce and ALP with Ea and ma """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        mn_star = self.profile.mnstar(rad)  # Effective nucleon mass
        
        num = (-2 * mn_star * Ea**2 + ma**2 * Ea + 2 * mn_star**2 * Ea + m_gamma**2 * Ea - mn_star * m_gamma**2 - ma**2 * mn_star)
        #den = np.where( Ea < (ma**2 + mn_star**2)/(2 * mn_star), 2 * (ma**2 + mn_star**2 - 2 * Ea * mn_star), 2 * (ma**2 + mn_star**2 + 2 * Ea * mn_star) )
        den = (2 * abs(ma**2 + mn_star**2 - 2 * Ea * mn_star))
    
        rad = (-ma**6 + Ea**2 * ma**4 + 2 * m_gamma**2 * ma**4 + 4 * Ea * mn_star * ma**4 - 
               m_gamma**4 * ma**2 - 4 * Ea**2 * mn_star**2 * ma**2 - 2 * Ea**2 * m_gamma**2 * ma**2 + 
               4 * mn_star**2 * m_gamma**2 * ma**2 - 4 * Ea * mn_star * m_gamma**2 * ma**2 - 4 * Ea**3 * mn_star * ma**2 + 
               Ea**2 * m_gamma**4 + 4 * Ea**4 * mn_star**2 - 4 * Ea**2 * mn_star**2 * m_gamma**2 + 4 * Ea**3 * mn_star * m_gamma**2)
    
        min_aux = num / den - np.sqrt(rad) / (2 * abs(ma**2 + mn_star**2 - 2 * Ea * mn_star))
        
        return np.where(min_aux < 0, Ea, min_aux) 

    def EgammaMaxp(self, Ea, rad, ma):
        """ Maximum photon energy to produce and ALP with Ea and ma """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        mn_star = self.profile.mnstar(rad)  # Effective nucleon mass
        
        num = (-2 * mn_star * Ea**2 + ma**2 * Ea + 2 * mn_star**2 * Ea + m_gamma**2 * Ea - mn_star * m_gamma**2 - ma**2 * mn_star)
        #den = np.where( Ea < (ma**2 + mn_star**2)/(2 * mn_star), 2 * (ma**2 + mn_star**2 - 2 * Ea * mn_star), 2 * (ma**2 + mn_star**2 + 2 * Ea * mn_star) )
        den = (2 * abs(ma**2 + mn_star**2 - 2 * Ea * mn_star))
    
        rad = (-ma**6 + Ea**2 * ma**4 + 2 * m_gamma**2 * ma**4 + 4 * Ea * mn_star * ma**4 - 
               m_gamma**4 * ma**2 - 4 * Ea**2 * mn_star**2 * ma**2 - 2 * Ea**2 * m_gamma**2 * ma**2 + 
               4 * mn_star**2 * m_gamma**2 * ma**2 - 4 * Ea * mn_star * m_gamma**2 * ma**2 - 4 * Ea**3 * mn_star * ma**2 + 
               Ea**2 * m_gamma**4 + 4 * Ea**4 * mn_star**2 - 4 * Ea**2 * mn_star**2 * m_gamma**2 + 4 * Ea**3 * mn_star * m_gamma**2)
    
        max_aux = num / den + np.sqrt(rad) / (2 * abs(ma**2 + mn_star**2 - 2 * Ea * mn_star))
        
        return np.where(max_aux < 0, Ea, max_aux) 


    def dndEdt(self, Ea, rad, ma, ga):
        """ Spectrum of ALPs produced in p gamma -> p a """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        npeff = self.profile.npeff(rad)  # Effective number density of protons

        def integrand(Egamma):
            return (self.f_gamma(Egamma, rad) *
                    (Egamma**2 - m_gamma**2) *
                    self.dsigma_dEa(Egamma, Ea, rad, ma, ga))

        Egamma_min = self.EgammaMinp(Ea, rad, ma)
        Egamma_max = self.EgammaMaxp(Ea, rad, ma)

        return np.where(Ea < ma, 0, 
                        (3 / (2 * np.pi**2)) * npeff * quad(integrand, Egamma_min, Egamma_max, epsrel=1e-4)[0] )
                        
         
    def dndEdt_ABS(self, Ea, rad, ma, ga):
        """ Spectrum of ALPs produced in p gamma -> p a """
        m_gamma = self.profile.mphoton(rad)  # Effective photon mass
        npeff = self.profile.npeff(rad)  # Effective number density of protons

        def integrand(Egamma):
            return ( (1 + self.f_gamma(Egamma, rad)) *
                    (Egamma**2 - m_gamma**2) *
                    self.dsigma_dEa(Egamma, Ea, rad, ma, ga))

        Egamma_min = self.EgammaMinp(Ea, rad, ma)
        Egamma_max = self.EgammaMaxp(Ea, rad, ma)

        return np.where(Ea < ma, 0, 
                        (3 / (2 * np.pi**2)) * npeff * quad(integrand, Egamma_min, Egamma_max, epsrel=1e-4)[0] )
                        

    def lambdaam1(self, Ea, rad, ma, ga):
        """ ALP mean free path ^-1, MeV """
        return np.where(Ea < ma, 0, 
                        np.array([ (2 * np.pi**2) / (Ea**2 - ma**2) * 
                        self.dndEdt_ABS(Ea, rad, ma, ga) ]) )


                        
                        
class SN_particle_generator:
    def __init__(self, profile, lambdaam1_all, dndEdt_all, Nsamples_class=15):
    
        """use as profile the SNprofiles class, 
    
        use the (mean free path)**-1 to include absorption from the corresponding classes,
        use the spectrum per unit volume dn/dEadt to include the source processes, for example: 
       
        	brems_X = ALP_Brems( profile=SN_func, NuclStruct=NSF_X)
        	pion_X  = ALP_Pion( profile=SN_func, NuclStruct=NSF_X)
       		
        	lambdaam1_all = [brem_X.lambdaam1, pion_X.lambdaam1]
        		or lambdaam1_all = [None] for no absorption
        	dndEdt_all=[brems_X.dndEdt, pion_X.dndEdt] """
        	
        self.Nsamples_class = Nsamples_class
        self.profile = profile    	
        self.kmtoMeVminus1 = 1/Units.kmminus1toMeV
        self.MeVtoerg = Units.MeVtoerg
        self.stoMeVminus1 = Units.stoMeVminus1
    	
    	
        # sum the ALP spectrum per unit volume dn/dEadt, MeV**3, if the user included more than 1 source
        if isinstance(dndEdt_all, list):
            self.dndEdt_all = self.sum_functions(*dndEdt_all)
        else:
            self.dndEdt_all = dndEdt_all
            
    	
        # check if the user included absorption
        # if not, return (mean free path)**-1 = 0
        if lambdaam1_all is None or (isinstance(lambdaam1_all, list) and all(f is None for f in lambdaam1_all)):
            self.lambdaam1_all = lambda *args, **kwargs: np.array([0])
        # if absorption, sum the (mean free paths)**-1
        elif isinstance(lambdaam1_all, list):
            self.lambdaam1_all = self.sum_functions(*lambdaam1_all)
        else:
            self.lambdaam1_all = lambdaam1_all
            
            
        
    	
    @staticmethod
    def sum_functions(*functions):
        """Creates a function that sums the result of multiple functions"""
        def combined_function(*args, **kwargs):
            return sum(f(*args, **kwargs) for f in functions)
        return combined_function


    def Gammaa(self, Ea, rad, ma, ga):
        """ Absorption rate, MeV """
        return self.lambdaam1_all(Ea, rad, ma, ga) * (1 - np.exp(-Ea / self.profile.T(rad)))


        
    def Ecorrected(self, Ea, rad, mu, s):
        """ ALP Energy corrected by gravitational redshift """
        return Ea * self.profile.lapse(rad) / self.profile.lapse(np.sqrt(rad**2 + s**2 + 2 * rad * s * mu))

        
        
    def intsF(self, Ea, rad, ma, ga, mu, Nsamples=None):
        """ integral inside optical depth, adimensional """
        if Nsamples is None:
            Nsamples = self.Nsamples_class
            
        radMAX = 20
        upper_limit = max(0, -rad * mu + rad * np.sqrt(mu**2 - (1 - (radMAX / rad)**2)))

        # Create grid of points
        s_vals = np.linspace(0, upper_limit, Nsamples)

        # Evaluate
        F_s = np.array([( self.kmtoMeVminus1 * self.Gammaa(self.Ecorrected(Ea, rad, mu, s), 
                                                          np.sqrt(rad**2 + s**2 + 2 * rad * s * mu), 
                                                          ma, ga))[0] for s in s_vals])

        # Integrate
        return simpson(F_s, x=s_vals)
        
        
    def intsF2(self, Ea, rad, ma, ga, mu):
        """ integral inside optical depth, adimensional """
    
        radMAX = 20
        upper_limit = max(0, -rad * mu + rad * np.sqrt(mu**2 - (1 - (radMAX / rad)**2)))

        # Define the function to integrate
        def integrand(s):
            E_corr = self.Ecorrected(Ea, rad, mu, s)
            r_s = np.sqrt(rad**2 + s**2 + 2 * rad * s * mu)
            Gamma_a = self.Gammaa(E_corr, r_s, ma, ga)[0]
            return Gamma_a

        # integrate
        integral_value, error = quad(integrand, 0, upper_limit, epsabs=1e-2, epsrel=1e-2)

        return self.kmtoMeVminus1 * integral_value
       
        
    def dNdEdt(self, Ea, ma, ga, Nsamples=None):
        """ ALP spectrum dN/dEadt, adimensional """
        if Nsamples is None:
            Nsamples = self.Nsamples_class
            
        # if Ea < ma, return 0
        return np.where(Ea < ma, 0,
                self._compute_dNdEdt(Ea, ma, ga, Nsamples) )
               
        
    def _compute_dNdEdt(self, Ea, ma, ga, Nsamples=None):
        """ Compute dN/dEadt if Ea > ma """
        if Nsamples is None:
            Nsamples = self.Nsamples_class
            
        rad_min, rad_max = 0, 20
        mu_min, mu_max = -1, 1
        
        kmtoMeVminus1_3 = self.kmtoMeVminus1**3

        def integrand(rad, mu):
            # Function to integrate over radius and mu
            lapse = self.profile.lapse(rad)
            E_corr = Ea / lapse
            term1 = rad**2 / lapse
            term2 = np.exp(-self.intsF(Ea, rad, ma, ga, mu, Nsamples))
            term3 = self.dndEdt_all(E_corr, rad, ma, ga)

            return 4 * np.pi * 0.5 * kmtoMeVminus1_3 * term1 * term2 * term3

        def integrate_over_mu(rad):
            # Integrate over mu for a fixed rad
            result, _ = quad(lambda mu: integrand(rad, mu), mu_min, mu_max, epsabs=1e-2, epsrel=1e-2)
            return result
            
	# Integrate over rad
        integral_value, _ = quad(integrate_over_mu, rad_min, rad_max, epsabs=1e-2, epsrel=1e-2)

        return np.where(Ea < ma, 0, 
        	 integral_value )
        	 
        	 
        	 
    def dNdEdt2(self, Ea, ma, ga, Nsamples=None):
        """ ALP spectrum dN/dEadt, adimensional """
        if Nsamples is None:
            Nsamples = self.Nsamples_class
            
        # if Ea < ma, return 0
        return np.where(Ea < ma, 0,
                self._compute_dNdEdt2(Ea, ma, ga, Nsamples) )
               
        
    def _compute_dNdEdt2(self, Ea, ma, ga, Nsamples=None):
        """ Compute dN/dEadt if Ea > ma """
        if Nsamples is None:
            Nsamples = self.Nsamples_class
            
        
        lower_lims = [0, -1]
        upper_lims = [20, 1]
        
        kmtoMeVminus1_3 = self.kmtoMeVminus1**3

        def integrand(x):
            # Function to integrate over radius and mu
            rad = x[0]  # Ahora accedemos correctamente a los valores
            mu = x[1]
            
            lapse = self.profile.lapse(rad)
            E_corr = Ea / lapse
            term1 = rad**2 / lapse
            term2 = np.exp(-self.intsF(Ea, rad, ma, ga, mu, Nsamples))
            term3 = self.dndEdt_all(E_corr, rad, ma, ga)

            return 4 * np.pi * 0.5 * kmtoMeVminus1_3 * term1 * term2 * term3
        
	
        result = qmc_quad(integrand, lower_lims, upper_lims, n_points = Nsamples)

        return np.where(Ea < ma, 0, 
        	 result.integral )
        
       


        
    def La(self, ma, ga, Nsamples=None):
        """ ALP spectrum luminosity, MeV**2 """
        if Nsamples is None:
            Nsamples = self.Nsamples_class
            
        return self._compute_La(ma, ga, Nsamples)
        
        
    def LaTest(self, ma, ga, L_neut=3e52, Nsamples=None):
        """ ALP spectrum luminosity, MeV**2 """
        if Nsamples is None:
            Nsamples = self.Nsamples_class
            
        La_MeV2 = self._compute_La(ma, ga, Nsamples)
        La_ergsm1 = La_MeV2 * self.MeVtoerg * self.stoMeVminus1
        rate = La_ergsm1 / L_neut
        return np.array([La_MeV2, La_ergsm1, rate])
               

        
    def _compute_La(self, ma, ga, Nsamples=None):
        """ Compute Luminosity """
        if Nsamples is None:
            Nsamples = self.Nsamples_class
    
        rad_min, rad_max = 0, 20
        mu_min, mu_max = -1, 1

        def integrand(Ea, rad, mu):
            intsF_aux = self.intsF(Ea, rad, ma, ga, mu, Nsamples)
            prefactor = rad**2 * self.profile.lapse(rad)**2 * np.exp(-intsF_aux)
            return Ea * prefactor * self.dndEdt_all(Ea / self.profile.lapse(rad), rad, ma, ga)

        def integral_over_mu(Ea, rad):
            # Integrate over mu
            return quad(lambda mu: integrand(Ea, rad, mu), mu_min, mu_max, epsabs=1e-2, epsrel=1e-2)[0]

        def integral_over_rad(rad):
            # Integrate over Ea
            Ea_min = ma / self.profile.lapse(rad) #20
            Ea_max = 250
            return quad(lambda Ea: integral_over_mu(Ea, rad), Ea_min, Ea_max, epsabs=1e-2, epsrel=1e-2, limit=100)[0]

        # Integrate over Ea
        integral_value = quad(integral_over_rad, rad_min, rad_max, epsabs=1e-2, epsrel=1e-2)[0]

        return 4 * np.pi * self.kmtoMeVminus1**3 * 0.5 * integral_value

        
