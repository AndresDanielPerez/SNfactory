import numpy as np
from scipy.special import spherical_jn  # BesselJ[1] ≡ spherical_jn(1, x)
from scipy.integrate import quad

from constants import Units, Constants
from SNprofiles import SNprofiles_lightDM


class DmAcross:
    def __init__(self, Z, mA, diffSNfactor, profile): 
    	self.diffSNfactor = diffSNfactor # diffuse SN factor [cm^-2 s^-1]
    	self.profile = profile # SN profiles
    
    	self.alphaEM = Constants.alphaEM # Fine-structure constant
    	self.hbar_MeV = Constants.hbar_MeV  # [MeV·s]
    	self.cms = Constants.cms       # light speed [m/s]
    	self.mp = Constants.mp # proton mass [MeV]
        
    	self.GeVtomminus1 = Units.GeVtomminus1 # GeV to m⁻¹
    	self.MeVtokg = Units.MeVtokg # MeV to kg
    	self.kpctocm = Units.kpctocm
        
    	self.Z = Z                       # Atomic number of Xenon
    	self.mA = mA                   # Nuclear mass [MeV]
    	self.prefactor = (self.hbar_MeV**2) * (self.cms**2)  # [MeV^2 m^2/s^2]


    def Q(self, Erec):
        """
        Transfered momentum Q [MeV] for Erec [MeV]
        """
        return np.sqrt(2 * self.mA * Erec)

    def F(self, Erec):
        """
        Form factor F(Erec))
        https://arxiv.org/pdf/2006.11225.pdf
        """
        
        # Nuclear parameters
        aa = 0.52e-15 # m
        ss = 0.9e-15 # m
        cc = (1.23 * (self.mA / 1e3)**(1./3) - 0.6) * 1e-15 # m

        # R00 effective nuclear constant
        R00 = np.sqrt(cc**2 + (7 / 3) * np.pi**2 * aa**2 - 5 * ss**2)

        q = self.Q(Erec)
        q_scaled = q * R00 * self.GeVtomminus1 * 1e-3

        if q_scaled == 0:
            return 1.0

        j1 = spherical_jn(1, q_scaled)
        exp_factor = np.exp(-1 * q**2 * ss**2 * (self.GeVtomminus1 * 1e-3)**2 / 2)

        return 3 * (j1 / q_scaled) * exp_factor
        

        
    def red_mass(self, m1, m2):
        """Reduced mass"""
        return (m1 * m2) / (m1 + m2)


    def Tmax(self, mchi, p):
    	""" max transfered energy [MeV] """
    	return (p**2 + 2 * mchi * p) / ( ((mchi + self.mA)**2 / (2 * self.mA)) + p )

    def sigmaDMp(self, mchi, p, y_coup):
    	""" cross-section DM-p [cm**2] """
    	factor = (8 * np.pi * y_coup * self.alphaEM) / (mchi**2 * (self.GeVtomminus1 * 1e-3)**2)
    	return factor * (2 + p**2 / mchi**2) * (100**2)

    def dsigmadErecNR(self, mchi, p, Erec, y_coup):
    	"""
        Compute the differential cross-section dσ/dE_rec for DM-Xenon scattering.
        
        NO RELATIVISTIC APPROXIMATION!!!!!

        Parameters:
        - mchi   : DM mass [MeV]
        - p      : momentum of the incoming DM particle [MeV]
        - Erec   : recoil energy of the target nucleus [MeV]
        - y_coup : coupling (dimensionless)

        Returns:
        - dσ/dE_rec [cm^2 / MeV]
        """

    	red_mass_ratio = (self.red_mass(mchi, self.mA) / self.red_mass(mchi, self.mp))**2
        
    	return (self.sigmaDMp(mchi, p, y_coup) / self.Tmax(mchi, p)) * self.Z**2 * red_mass_ratio * self.F(Erec)**2
        
        
    def dsigmadErec(self, mchi, p, Erec, y_coup):
        """
        Compute the differential cross-section dσ/dE_rec for DM-Xenon scattering.

        Parameters:
        - mchi   : DM mass [MeV]
        - p      : momentum of the incoming DM particle [MeV]
        - Erec   : recoil energy of the target nucleus [MeV]
        - y_coup : coupling (dimensionless)

        Returns:
        - dσ/dE_rec [cm^2 / MeV]
        """

        sqrt_mA2_p2 = np.sqrt(self.mA**2 + p**2)
        sqrt_mchi2_p2 = np.sqrt(mchi**2 + p**2)
        denom = mchi**4 * (sqrt_mA2_p2 + sqrt_mchi2_p2)**2

        # Numerator split into terms
        term1 = p**2 * (1 - (self.mA * Erec) / p**2) * (2 * sqrt_mA2_p2 * sqrt_mchi2_p2 + self.mA**2 + mchi**2)
        term2 = p**2 * (2 * sqrt_mA2_p2 * sqrt_mchi2_p2 + mchi**2 + 3 * p**2)
        term3 = self.mA**2 * (2 * mchi**2 + p**2)
        term4 = p**4 * (1 - (self.mA * Erec) / p**2)**2

        numerator = term1 + term2 + term3 + term4

        dsigmadE = self.prefactor * 4 * np.pi * self.alphaEM * y_coup * self.Z**2 * self.mA / p**2 * (numerator / denom) * (100**2)

        if dsigmadE > 0:
            return dsigmadE
        else:
            return 0




    def diffFlux(self, mchi, p, rad, Nchi):
        """
        Differential diffuse galactic flux of dark fermions
        
        Parameters:
        - mchi   : DM mass [MeV]
        - p      : momentum of the incoming DM particle [MeV]
        - rad    : radius [km]
        - Nchi   : number flux of DM from a SN
        
        returns dflux/dp  [cm^-2 s^-1 MeV^-1]
        """
        T_r = self.profile.T(rad)
        
        # total flux on Earth
        Sigmadiffuse = Nchi * self.diffSNfactor  # cm^-2 s^-1
        

        # aux energy
        E_aux = np.sqrt(p**2 + mchi**2)


        numerator = (p**3 / (np.exp(E_aux / T_r) + 1)) * (1./ E_aux)

        # normalization (integral)
        def integrand(En):
            return (En**2 - mchi**2) / (np.exp(En / T_r) + 1)

        normalization, _ = quad(integrand, mchi, np.inf, epsrel=1e-4)

        return Sigmadiffuse * numerator / normalization
        

        
        
    def Fpinf(self, p0, mchi, rad):
        """
        Momentum at infinity (p_inf) given p0, mchi and rad.
        """
        #pot = self.profile.potential(rad) # this takes time, use lapse if already computed
        pot = ( 1 - self.profile.lapse(rad)**2 ) /2
        return p0 * np.sqrt(1 - (2 * pot * (p0**2 + mchi**2) / p0**2))

    def Fp0(self, pinf, mchi, rad):
        """
        Momentoum at origen (p0) given p_inf, mchi and rad.
        """
        #pot = self.profile.potential(rad) # this takes time, use lapse if already computed
        pot = ( 1 - self.profile.lapse(rad)**2 ) /2
        numerator = pinf**2 + 2 * pot * mchi**2
        denominator = 1 - 2 * pot
        return np.sqrt(numerator / denominator)

    def Fp0min(self, mchi, rad):
        """
        Mimimum momentum at origen to escape the potential.
        """
        #pot = self.profile.potential(rad) # this takes time, use lapse if already computed
        pot = ( 1 - self.profile.lapse(rad)**2 ) /2
        return np.sqrt((2 * pot / (1 - 2 * pot)) * mchi**2)
        
        
        
    def tauN(self, rN, mchi, p, y_coup):
        """
        transmission factor to account for the losses due scatter outside the SN
        ∫ (rN/rad)^2 * np(rad) * σ_DM-p draad from rN until infinity
        """
        def integrand(rad):
            ratio = (rN / rad)**2
            np = self.profile.npF(rad) * 1e-9 * self.GeVtomminus1**3  # in m**-3
            sigma = self.sigmaDMp(mchi, p, y_coup) * (1/100**2) # in m**2
            return ratio * np * sigma * 1e3 # in km**-1s

        result, _ = quad(integrand, rN, np.inf, limit=100, epsabs=1e-4, epsrel=1e-4)
        return result # adimensional
        
        
        
        
    def dspectra(self, mchi, Erec, y_coup, rad, Nchi):
        """
        recoil spectra in [MeV^-1 s^-1 ton^-1]
        """
        def integrand(pinf):
            p0 = self.Fp0(pinf, mchi, rad) # MeV
            dsdE = self.dsigmadErec(mchi, pinf, Erec, y_coup) # [cm^2 / MeV]
            dflux = self.diffFlux(mchi, p0, rad, Nchi) # [cm^-2 s^-1 MeV^-1]
            return dsdE * dflux # [s^-1 MeV^-2]
    		
        p_min = np.sqrt(0.5 * self.mA * Erec)  # kinematics
        result, _ = quad(integrand, p_min, np.inf, limit=100, epsabs=1e-4, epsrel=1e-4) # [s^-1 MeV^-1]
        return result * (1000 / (self.mA * self.MeVtokg)) # [MeV^-1 s^-1 ton^-1]
        
        
        
    def spectra(self, mchi, y_coup, rad, Nchi, Mtarget, Texp, Emin, Emax):
        """
        Total recoil events
        Mtarget in kg
        Texp in seconds
        """
        def integrand_Erec(Erec):
            def integrand_pinf(pinf):
            	p0 = self.Fp0(pinf, mchi, rad) # MeV
            	dsdE = self.dsigmadErec(mchi, pinf, Erec, y_coup) # [cm^2 / MeV]
            	dflux = self.diffFlux(mchi, p0, rad, Nchi) # [cm^-2 s^-1 MeV^-1]
            	return dsdE * dflux # [s^-1 MeV^-2]
            	
            p_min = np.sqrt(0.5 * self.mA * Erec)  # kinematics
            
            result, _ = quad(integrand_pinf, p_min, np.inf, limit=100, epsabs=1e-8, epsrel=1e-8)
            return result  # [s^-1 MeV^-1]
    		
        # Integrate over Erec
        result2, _ = quad(integrand_Erec, Emin, Emax, limit=100, epsabs=1e-8, epsrel=1e-8)
           
        return result2 * (Mtarget / (self.mA * self.MeVtokg)) * Texp # adimensional
        

        
            
    def convert_dp_to_dE(self, dfunc_dp, pchi, mchi):
        """
        convert some function value as dfunc/dp to dfunc/dE
        E=sqrt(p^2 + m^2)
        dE/dp = p / sqrt(p^2 + m^2)
        dp/dE = E / p
        """
        pchi = np.asarray(pchi)
        dfunc_dp = np.asarray(dfunc_dp)

        Echi = np.sqrt(pchi**2 + mchi**2)
    
        # Initialize
        result = np.zeros_like(pchi)

        # Mask Echi > 1.05 * mchi
        mask = Echi >= 1.05 * mchi
        result[mask] = dfunc_dp[mask] * (Echi[mask] / pchi[mask])
    
        return result
        
        
    def E_to_p(self, Echi, mchi):
        return np.sqrt(Echi**2 - mchi**2)
        
    def dE_dp(self, Echi, mchi):
        return Echi / np.sqrt(Echi**2 - mchi**2)
    
            


    def flux_1sim_BINS(self, Erange, Ebins, mchi, rad, Nchi):
      """ computes how the flux would be reduced due to the distance [cm^-2] (BINNED), considering the contributions of EVERY SN in 1 SIMULATION """

      result_bins = [] # flux per bin

      for i in range(len(Ebins) - 1):
	      Ebin_min = Ebins[i]
	      Ebin_max = Ebins[i+1]

	      integral_bin = 0.0

	      for idx, (E1, E2, dSN) in enumerate(Erange):
	      	Eoverlap_min = max(E1, Ebin_min)
	      	Eoverlap_max = min(E2, Ebin_max)

	      	if Eoverlap_min < Eoverlap_max and Eoverlap_min > mchi *1.05:
	      		# dflux/dp  [MeV^-1] for diffSNfactor=1   and 1/dSN in [cm**-2]
	      		integral, _ = quad(lambda Echi: self.dE_dp(Echi, mchi) * self.diffFlux(mchi, self.Fp0(self.E_to_p(Echi, mchi), mchi, rad), rad, Nchi) * (1 / ( (4*np.pi) * (dSN * self.kpctocm)**2 ) ), 
	      			Eoverlap_min,
                		Eoverlap_max, 
                		epsrel=1e-2, epsabs=1e-2 )
	      		integral_bin += integral # cm^-2 (binwidth)MeV^-1

	      result_bins.append(integral_bin)

      return np.array(result_bins) # cm^-2 (binwidth)MeV^-1
      

        
        
    def dspectra_1sim(self, Erec, mchi, y_coup, rad, Nchi, Erange):
      """ dNγ/dEγ considering all simulated SN """
      pchi_min = np.sqrt(0.5 * self.mA * Erec)  # kinematics
      Echi_min = np.sqrt( pchi_min**2 + mchi**2)

      # If the range makes no sense, return 0
      if Echi_min == 0:
        return 0.0

      def integrand(Echi):
        if Echi < mchi*1.01:
            return 0.0

        flux_sum = 0.0
        for E1, E2, dSN in Erange:
            if E1 <= Echi <= E2:
                # dflux/dp  [MeV^-1] for diffSNfactor=1   and 1/dSN in [cm**-2]
                flux = self.dE_dp(Echi, mchi) * self.diffFlux(mchi, self.Fp0(self.E_to_p(Echi, mchi), mchi, rad), rad, Nchi) * (1 / (4 * np.pi * (dSN * self.kpctocm)**2)) #  cm⁻² MeV⁻¹
                flux_sum += flux
        return flux_sum * self.dsigmadErec(mchi, self.E_to_p(Echi, mchi), Erec, y_coup) # MeV^-2   (dsigmadErec in cm**2 MeV**-1)

      integral_result, _ = quad(
        integrand,
        Echi_min,
        np.inf,
        epsrel=1e-2,
        epsabs=1e-2
      )

      return integral_result * (1000 / (self.mA * self.MeVtokg))# [MeV^-1 ton^-1]



    def dspectra_1sim_BINS(self, Erec_bins, mchi, y_coup, rad, Nchi, Mtarget, Erange):
        """
        Total recoil events per bin
        Mtarget in kg
        """
        
        binned_vals = []
        
        for i in range(len(Erec_bins) - 1):
            Ebin_min, Ebin_max = Erec_bins[i], Erec_bins[i+1]
      	
            # Integrate dNgammadEgamma_1sim entre E1 y E2
            integral, _ = quad(lambda E: self.dspectra_1sim(E, mchi, y_coup, rad, Nchi, Erange), Ebin_min, Ebin_max, epsabs=1e-2, epsrel=1e-2)
                       			
            binned_vals.append(integral * (Mtarget/1000) / (Ebin_max - Ebin_min))
        
        return np.array(binned_vals) # adimensional
        

        
        
        
        

      
      
      
    def dNgammadEgamma_1sim_BINS(self, Egamma_bins, ma, ga, dNdEdt, Erange):
      """ dNγ/dEγ considering all simulated SN """
      
      binned_vals = []
      
      for i in range(len(Egamma_bins) - 1):
        Ebin_min, Ebin_max = Egamma_bins[i], Egamma_bins[i+1]
      	
        # Integrate dNgammadEgamma_1sim entre E1 y E2
        integral, _ = quad(lambda E: self.dNgammadEgamma_1sim(E, ma, ga, dNdEdt, Erange ), 
                       			Ebin_min, Ebin_max, epsrel=1e-2, epsabs=1e-2)
                       			
        binned_vals.append(integral / (Ebin_max - Ebin_min))
        
      return np.array(binned_vals)
