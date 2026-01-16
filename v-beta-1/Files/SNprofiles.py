import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import root_scalar

from constants import Units, Constants


class SNprofiles_QCDALP:
    def __init__(self, rho_file, Ye_file, T_file, mphoton_file, npeff_file, echempot_file, Ymu_file, Ypi_file, use_saved_lapse=False, lapse_file="ConfigFiles/SN-profiles/18MsolProgenitor-AB/lapse_PROF1.dat"):
        """
        Initialize the SN profiles loading the data files
        Args:
            rho_file (str): density vs radius file [MeV^4].
            Ye_file (str): fraction of electrons vs radius file.
            T_file (str): temperature vs radius file [MeV].
            mphoton_file (str): effective photon mass in the SN plasma [MeV].
            npeff_file (str): effective proton number density [MeV^3].
            echempot_file (str): electron chemical potential [MeV].
            Ymu_file (str): fraction of muons vs radius file.
            Ypi_file (str): fraction of pions vs radius file.
            use_saved_lapse: True uses the saved lapse function, if False computes it (TAKES TIME, use the already saved).
            lapse_file (str): lapse function file. 
        """
        self.cmminus1toMeV = Units.cmminus1toMeV
        self.GeVtomminus1 = Units.GeVtomminus1
        self.MeVtokg = Units.MeVtokg
        self.mn = Constants.mn
        self.mp = Constants.mp
        self.mpi = Constants.mpi
        self.mmu = Constants.mmu

        # Load and interpolate the profiles
        self.rho_interp = self._load_and_interpolate(rho_file)
        self.Ye_interp = self._load_and_interpolate(Ye_file)
        self.T_interp = self._load_and_interpolate(T_file)
        self.mphoton_interp = self._load_and_interpolate(mphoton_file)
        self.npeff_interp = self._load_and_interpolate(npeff_file)
        self.echempot_interp = self._load_and_interpolate(echempot_file)
        self.Ymu_interp = self._load_and_interpolate(Ymu_file)
        self.Ypi_interp = self._load_and_interpolate(Ypi_file)

	# Tables for the degeneracy parameters
        self.etan_func = None
        self.etap_func = None
        self.etamu_func = None
        self.etapi_func = None

        # Generar tablas de degeneración
        self._generate_degeneracy_tables()
        
        # For the lapse function
        self.lapse_file = lapse_file
        if use_saved_lapse == True:
        	self.lapse_func = self._load_and_interpolate(lapse_file)
        else:
	        self.lapse_func = None
        	self._generate_lapse()

        


    @staticmethod
    def _load_and_interpolate(file_path):
        """Load a file and interpolate linearly."""
        data = np.loadtxt(file_path)
        return interp1d(data[:, 0], data[:, 1], kind="linear", fill_value="extrapolate")



    def rho(self, r):
        """Density vs radius."""
        factor = (self.cmminus1toMeV**3) / (self.MeVtokg * 1000)
        return factor * self.rho_interp(r)


    def Ye(self, r):
        """Fraction of electrons (Ye) vs radius."""
        return self.Ye_interp(r)

    def Yn(self, r):
        """Fraction of neutrons (Yn) vs radius."""
        return 1 - self.Ye(r)

    def Yp(self, r):
        """Fraction of protons (Yp) vs radius."""
        return self.Ye(r)
        
    def Yparticle(self, r, particle):
    	if particle == "n":
    		Y = self.Yn(r)
    	elif particle == "p":
    		Y = self.Yp(r)
    	return Y
    	
    def Ymu(self, r):
        """Fraction of muons (Ymu) vs radius."""
        return self.Ymu_interp(r)
        
    def Ypi(self, r):
        """Fraction of pions (Ypi) vs radius."""
        return self.Ypi_interp(r)


    def npF(self, r):
        """Numerical density of protons."""
        aux = (self.Yp(r) * self.rho(r)) / self.mp
        if aux > 0:
        	return aux
        else:
	        return 0 

    def nnF(self, r):
        """Numerical density of neutrons."""
        aux = (self.Yn(r) * self.rho(r)) / self.mn
        if aux > 0:
        	return aux
        else:
	        return 0 

    def nbF(self, r):
        """Numerical density of barions."""
        aux = self.npF(r) + self.nnF(r)
        if aux > 0:
        	return aux
        else:
	        return 0
	        
    def nmuF(self, r):
        """Numerical density of muon."""
        aux = (self.Ymu(r) * self.rho(r)) / self.mp
        if aux > 0:
        	return aux
        else:
	        return 0 
	        
    def npiF(self, r):
        """Numerical density of pions."""
        aux = (self.Ypi(r) * self.rho(r)) / self.mp
        if aux > 0:
        	return aux
        else:
	        return 0


    def T(self, r):
        """Temperature vs radius."""
        return self.T_interp(r)


    def mnstar(self, r):
        """Effective nucleon mass (mn*) vs radius."""
        rho_factor = 3e14 * (1e12 / (self.MeVtokg * self.GeVtomminus1**3))  # factor
        return self.mn * ((0.58 - 1) * self.rho(r) / rho_factor + 1)
        
        
    def _degeneracy_integral(self, eta, mnstar, T):
        """Integrate the degeneracy function."""
        def integrand(p):
            return p**2 / (np.exp(p**2 / (2 * mnstar * T) - eta) + 1)
        return (1 / np.pi**2) * quad(integrand, 0, np.inf)[0]

    def _solve_degeneracy(self, n_target, mnstar, T):
        """Solve the equation to find eta."""
        def equation(eta):
            return self._degeneracy_integral(eta, mnstar, T) - n_target
        sol = root_scalar(equation, bracket=[-20, 20], method='brentq')
        return sol.root if sol.converged else None
        
        
    def _degeneracy_integralBE(self, eta, mnstar, T):
        """Integrate the degeneracy function."""
        def integrand(p):
            return (1 / np.pi**2) * p**2 / (np.exp(p**2 / (2 * mnstar * T) - eta) -1)
        return  quad(integrand, 0, 10000, epsrel=1e-4)[0]

    def _solve_degeneracyBE(self, n_target, mnstar, T):
        """Solve the equation to find eta."""
        def equation(eta):
            return self._degeneracy_integralBE(eta, mnstar, T) - n_target
        #sol = root_scalar(equation, bracket=[-100, 100], method='brentq')
        sol = root_scalar(equation, method='newton', x0=0.0)
        return sol.root if sol.converged else None
        
    # THESE ONES ARE SLOW 
    def etan_full(self, r):
        """Neutron degeneracy parameter."""
        mnstar = self.mnstar(r)
        T = self.T(r)
        nn = self.nnF(r)
        return self._solve_degeneracy(nn, mnstar, T)

    def etap_full(self, r):
        """Proton degeneracy parameter."""
        mnstar = self.mnstar(r)
        T = self.T(r)
        np_density = self.npF(r)
        return self._solve_degeneracy(np_density, mnstar, T)
        
    def eta_fullparticle(self, r, particle):
    	if particle == "n":
    		eta = etan_full.Yn(r)
    	elif particle == "p":
    		eta = etap_full.Yp(r)
    	return eta
    	
    def etapi_full(self, r):
        """Pion degeneracy parameter."""
        return ( self.eta_fullparticle(r, "n") - self.eta_fullparticle(r, "p") ) - ( Constants.mpi / self.T(r) )
        
    def mphoton(self, r):
        """photon mass (plasma) vs radius."""
        factor = 18
        return factor * self.mphoton_interp(r)
        
    def npeff(self, r):
        """effective proton number density vs radius."""
        factor = 2e37 * (self.cmminus1toMeV**3)
        return factor * self.npeff_interp(r)
        
    def echempot(self, r):
        """electron chemical potential vs radius."""
        factor = 200
        return factor * self.echempot_interp(r)


        
        
        
        
    # THESE ONES ARE FASTER
    def _generate_degeneracy_tables(self):
        """Generate tables for etan y etap."""
        radii = np.arange(0.1, 20.1, 0.1)  # Radius between 0.1 and 20.1 with step 0.1
        etaneutron_tab = []
        etaproton_tab = []
        etamu_tab = []
        etapion_tab = []

        for r in radii:
            mnstar = self.mnstar(r)
            mmu = self.mmu
            mpi = self.mpi
            T = self.T(r)
            nnF = self.nnF(r)
            npF = self.npF(r)
            nmuF = self.nmuF(r)
            npiF = self.npiF(r)

            etan = self._solve_degeneracy(nnF, mnstar, T)
            etap = self._solve_degeneracy(npF, mnstar, T)
            etamu = self._solve_degeneracy(nmuF, mmu, T)
            etapi = self._solve_degeneracyBE(npiF, mpi, T)

            if etan is not None and etap is not None:
                etaneutron_tab.append([r, etan])
                etaproton_tab.append([r, etap])
                etamu_tab.append([r, etamu])
                etapion_tab.append([r, etapi])

        # interpolation
        self.etaneutron_tab = np.array(etaneutron_tab)
        self.etaproton_tab = np.array(etaproton_tab)
        self.etamu_tab = np.array(etamu_tab)
        self.etapion_tab = np.array(etapion_tab)

        self.etan_func = interp1d(
            np.array(etaneutron_tab)[:, 0], np.array(etaneutron_tab)[:, 1], kind="linear", fill_value="extrapolate"
        )
        self.etap_func = interp1d(
            np.array(etaproton_tab)[:, 0], np.array(etaproton_tab)[:, 1], kind="linear", fill_value="extrapolate"
        )
        self.etamu_func = interp1d(
            np.array(etamu_tab)[:, 0], np.array(etamu_tab)[:, 1], kind="linear", fill_value="extrapolate"
        )
        self.etapi_func = interp1d(
            np.array(etapion_tab)[:, 0], np.array(etapion_tab)[:, 1], kind="linear", fill_value="extrapolate"
        )

    def etan(self, r):
        """Neutron degeneracy parameter."""
        return self.etan_func(r)

    def etap(self, r):
        """Proton degeneracy parameter."""
        return self.etap_func(r)
        
    def etamu(self, r):
        """Muon degeneracy parameter."""
        return self.etamu_func(r)
        
    def etapi_F(self, r):
        """Pion degeneracy parameter."""
        return self.etapi_func(r)
        
    def etaparticle(self, r, particle):
    	if particle == "n":
    		eta = self.etan(r)
    	elif particle == "p":
    		eta = self.etap(r)
    	return eta
    	
    def etapi(self, r):
        """Pion degeneracy parameter."""
        return ( self.etaparticle(r, "n") - self.etaparticle(r, "p") ) - ( Constants.mpi / self.T(r) )
        



       
    def mass_enc(self, r):
        """
        Enclosed mass as a function of radius (MeV).
        This computes the integral of the density profile over the volume up to radius r.
        """
        radii = np.linspace(0, r, 500)  # Divide the range [0, r] into 500 points
        rho_values = np.array([self.rho(rad) for rad in radii])  # Density values
        dr = radii[1] - radii[0]  # Step size for integration

        # Cumulative integral to compute enclosed mass
        integral = np.cumsum(4 * np.pi * (radii**2) * rho_values) * dr
        return integral[-1] * (Units.kmminus1toMeV**-3) # MeV

    def potential(self, r):
        """
        Gravitational potential as a function of radius.
        Computes the integral of the enclosed mass over radius.
        """

        # Integrate over the range [r, 25 km] to compute the potential
        radii = np.linspace(r, 25, 500)  # Range from r to 25 km
        mass_values = np.array([self.mass_enc(rad) for rad in radii])  # Enclosed mass
        dr = radii[1] - radii[0]  # Step size for integration

        # Cumulative integral to compute the potential
        integral = np.cumsum((mass_values / (radii**2)) * dr)
        return Constants.GMeVminus2 * integral[-1] * Units.kmminus1toMeV  # Final result adimensional

    def _generate_lapse(self):
        """
        Computes and returns an interpolated lapse function α(r).
        The lapse function is calculated for a range of radii and then interpolated
        to allow continuous evaluation at any radius.
        """
        
        radii = np.linspace(0, 25, 500)  # Range from r=0 to 25 km

        # Compute the lapse values for the radii
        lapse_values = np.array([
            np.sqrt(1 - 2 * self.potential(rad)) for rad in radii
        ])
        
        # save the lapse
        lapse_tosave = [[radii[r_it], lapse_values[r_it]] for r_it in range(len(radii))]
        np.savetxt(self.lapse_file, lapse_tosave)
        print("lapse function grid saved in: ", self.lapse_file)

        # Interpolation of the lapse function
        self.lapse_func = interp1d(
            radii, lapse_values, kind="linear", fill_value="extrapolate"
        )
        
           
    def lapse(self, r):
        """lapse function α(r)."""
        return self.lapse_func(r)
        
        
        
        
        
        
class SNprofiles_lightDM:
    def __init__(self, rho_file, Ye_file, T_file, use_saved_lapse=False, lapse_file="ConfigFiles/SN-profiles/1MsolPNS-DeRocco/lapse_DeRocco.dat"):
        """
        Initialize the SN profiles loading the data files
        Args:
            rho_file (str): density vs radius file.
            Ye_file (str): fraction of particles vs radius file.
            T_file (str): temperature vs radius file.
            use_saved_lapse: True uses the saved lapse function, if False computes it (TAKES TIME, use the already saved).
            lapse_file (str): lapse function file. 
        """
        self.cmminus1toMeV = Units.cmminus1toMeV
        self.GeVtomminus1 = Units.GeVtomminus1
        self.MeVtokg = Units.MeVtokg
        self.mn = Constants.mn
        self.mp = Constants.mp
        self.mpi = Constants.mpi
        self.mmu = Constants.mmu

        # Load and interpolate the profiles
        self.rho_interp = self._load_and_interpolate(rho_file)
        self.Ye_interp = self._load_and_interpolate(Ye_file)
        self.T_interp = self._load_and_interpolate(T_file)

        
        # For the lapse function
        self.lapse_file = lapse_file
        if use_saved_lapse == True:
        	self.lapse_func = self._load_and_interpolate(lapse_file)
        else:
	        self.lapse_func = None
        	self._generate_lapse()

        


    @staticmethod
    def _load_and_interpolate(file_path):
        """Load a file and interpolate linearly."""
        data = np.loadtxt(file_path)
        return interp1d(data[:, 0], data[:, 1], kind="linear", fill_value="extrapolate")



    def rho(self, r):
        """Density vs radius."""
        factor = (self.cmminus1toMeV**3) / (self.MeVtokg * 1000)
        return factor * self.rho_interp(r)


    def Ye(self, r):
        """Fraction of electrons (Ye) vs radius."""
        return self.Ye_interp(r)

    def Yn(self, r):
        """Fraction of neutrons (Yn) vs radius."""
        return 1 - self.Ye(r)

    def Yp(self, r):
        """Fraction of protons (Yp) vs radius."""
        return self.Ye(r)
        
    def Yparticle(self, r, particle):
    	if particle == "n":
    		Y = self.Yn(r)
    	elif particle == "p":
    		Y = self.Yp(r)
    	return Y
    	


    def npF(self, r):
        """Numerical density of protons."""
        aux = (self.Yp(r) * self.rho(r)) / self.mp
        if aux > 0:
        	return aux
        else:
	        return 0 

    def nnF(self, r):
        """Numerical density of neutrons."""
        aux = (self.Yn(r) * self.rho(r)) / self.mn
        if aux > 0:
        	return aux
        else:
	        return 0 

    def nbF(self, r):
        """Numerical density of barions."""
        aux = self.npF(r) + self.nnF(r)
        if aux > 0:
        	return aux
        else:
	        return 0



    def T(self, r):
        """Temperature vs radius."""
        return self.T_interp(r)


       
    def mass_enc(self, r):
        """
        Enclosed mass as a function of radius (MeV).
        This computes the integral of the density profile over the volume up to radius r.
        """
        radii = np.linspace(0, r, 500)  # Divide the range [0, r] into 500 points
        rho_values = np.array([self.rho(rad) for rad in radii])  # Density values
        dr = radii[1] - radii[0]  # Step size for integration

        # Cumulative integral to compute enclosed mass
        integral = np.cumsum(4 * np.pi * (radii**2) * rho_values) * dr
        return integral[-1] * (Units.kmminus1toMeV**-3) # MeV

    def potential(self, r):
        """
        Gravitational potential as a function of radius.
        Computes the integral of the enclosed mass over radius.
        """

        # Integrate over the range [r, 25 km] to compute the potential
        radii = np.linspace(r, 25, 500)  # Range from r to 25 km
        mass_values = np.array([self.mass_enc(rad) for rad in radii])  # Enclosed mass
        dr = radii[1] - radii[0]  # Step size for integration

        # Cumulative integral to compute the potential
        integral = np.cumsum((mass_values / (radii**2)) * dr)
        return Constants.GMeVminus2 * integral[-1] * Units.kmminus1toMeV  # Final result adimensional

    def _generate_lapse(self):
        """
        Computes and returns an interpolated lapse function α(r).
        The lapse function is calculated for a range of radii and then interpolated
        to allow continuous evaluation at any radius.
        """
        
        radii = np.linspace(0, 25, 500)  # Range from r=0 to 25 km

        # Compute the lapse values for the radii
        lapse_values = np.array([
            np.sqrt(1 - 2 * self.potential(rad)) for rad in radii
        ])
        
        # save the lapse
        lapse_tosave = [[radii[r_it], lapse_values[r_it]] for r_it in range(len(radii))]
        np.savetxt(self.lapse_file, lapse_tosave)
        print("lapse function grid saved in: ", self.lapse_file)

        # Interpolation of the lapse function
        self.lapse_func = interp1d(
            radii, lapse_values, kind="linear", fill_value="extrapolate"
        )
        
           
    def lapse(self, r):
        """lapse function α(r)."""
        return self.lapse_func(r)
