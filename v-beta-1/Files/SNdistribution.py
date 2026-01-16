import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad, nquad, simpson
from scipy.optimize import root_scalar
from scipy.stats import poisson, expon, uniform, laplace

from constants import Units, Constants


class SNdistribution:
    def __init__(self, SNrate, Rd, H, RE, zE):
        """
        Initialize the SN distribution parameters:
        SNrate: number of galactic SN per century
        Rd: exponential radial parameter [kpc]
        H: exponential z parameter [kpc]
        
        RE: r positon of the Earth wrt the Galactic Center [kpc]
        zE: z position of the Earth wrt the Galactic plane [kpc]
        """
        self.SNrate = SNrate
        self.Rd = Rd
        self.H = H
        self.RE = RE
        self.zE = zE
        self.seconds_in_year = Units.seconds_in_year
        self.kpctocm = Units.kpctocm
        self.cms = Constants.cms


    def Asn(self):
      """ computes the normalization [kpc-3 yr-1] """
      def density_integrand(r, z):
      	return np.exp(-r / self.Rd) * np.exp(-abs(z) / self.H) * r

      Asn_aux, _ = nquad(density_integrand, [[0, np.inf], [-np.inf, np.inf]])
      return self.SNrate / (100 * 2*np.pi * Asn_aux)  
      

    def rhoSN(self, r, z):
      """ computes the density of SN for a given point (r, z) centered at the Galactic Center [kpc-3 yr-1] """
      return self.Asn() * np.exp(-r/self.Rd) * np.exp(-np.abs(z)/self.H)
    	
    	
    def flux_factor(self):
      """ computes the flux factor considering the contribution of all the galactic SN [cm-2 s-1] """
      # compute the normalization factor
      Asn = self.Asn()
      
      # number of points
      Nr, Nz = 1000, 500  

      # range for r
      r_min, r_max = 0.1, 100
      r_vals = np.logspace(np.log10(r_min), np.log10(r_max), Nr)

      # range for z (it has an abs)
      z_min, z_max = 0.01, 50
      z_positive = np.logspace(np.log10(z_min), np.log10(z_max), Nz // 2)
      z_vals = np.concatenate((-z_positive[::-1], z_positive))

      # meshgrid
      R, Z = np.meshgrid(r_vals, z_vals, indexing='ij')      
      
      # flux function to be integrated in theta for each (r, z)
      def flux_theta_integrand(theta, r, z):
      	denominator = ((r * np.cos(theta) - self.RE)**2 + (r * np.sin(theta))**2 + (z - self.zE)**2)
      	
      	# option to set an exclusion radius around Earth for which we have no contribution 
      	R_excl = 0.1 # kpc
      	if denominator < R_excl**2:
            return 0.0
      	return Asn * np.exp(-r / self.Rd) * np.exp(-abs(z) / self.H) * r / denominator

      # integrate in theta for each (r, z)
      F_vals = np.zeros_like(R)

      for i in range(Nr):
      	for j in range(Nz):
      		F_vals[i, j], _ = quad(flux_theta_integrand, 0, 2*np.pi, args=(r_vals[i], z_vals[j]), epsabs=1e-3, epsrel=1e-3)

      # Simpson to r and z
      flux_z = simpson(F_vals, x=z_vals, axis=1)
      flux = simpson(flux_z, x=r_vals, axis=0)

      return flux * self.seconds_in_year**-1 * self.kpctocm**-2 * (1/(4*np.pi))
      
      



    def ALPvel(self, Ea, ma):
      """ computes v_ALP/c """
      return np.sqrt(1 - (ma/Ea)**2 )   

    def dSNearth(self, r, theta, z):
      """ computes the distance between the SN and Earth [kpc] """
      return np.sqrt( r**2 + self.RE**2 - 2*r*self.RE*np.cos(theta) + (z-self.zE)**2 )
      
    def tarrival(self, Ea, ma, r, theta, z):
      """ computes the time of arrival for an ALP produced in a SN located at (r, theta, z) [s] """
      return self.dSNearth(r,theta,z) * self.kpctocm / (self.ALPvel(Ea,ma) * 1e2 * self.cms)
      
    def tpackage(self, Eamin, Eamax, ma, r, theta, z):
      """""
      returns the package time spread (in seconds) when it passes through Earth 
      width defined as the ALPs with energies (Eamin, Eamax) [MeV]
      for a particle with mass ma [MeV]
      produced in a SN located at (r, theta, z) (in kpc and rad)
      """""
      return self.tarrival(Eamin, ma, r, theta, z) - self.tarrival(Eamax, ma, r, theta, z)
    
    def Earrival(self, ta, ma, r, theta, z):
      """ computes the ALP energy that reaches Earth, after a time ta, for a SN located at (r, theta, z) [MeV] """
      return np.max([ 0, np.real( ma / np.sqrt( 1 - ( (self.dSNearth(r,theta,z) * self.kpctocm * 1e-2) / (self.cms*ta) )**2) ) ])
      
      
      
      

    def SNsimulation(self, num_ITER, time_low, time_up):
    
      rng = np.random.default_rng(12345)
    
      
      # num_ITER: number of iterations
      iteration = np.arange(1, num_ITER + 1)

      # Time window: these ranges should be OK
      #time_low = 1e10 or 0  # in seconds (how back in time we go, lower limit)
      #time_up = 2e13   # in seconds (how back in time we go, upper limit)

      # expected number of SN in the considered timeframe: SN per year  * years
      exp_SNe = (self.SNrate / 100) * (time_up - time_low) / self.seconds_in_year  # SN expected



      # to save the results
      SN_numbers = []
      SN_sim = []

      # Generate events (r, theta, z, time) for several iterations
      for Nit in range(num_ITER):
      
      	SN_counts = rng.poisson(exp_SNe)  # Number of SN for this simulation
      	SN_numbers.append(SN_counts) # save
    
      	# Generate r, theta, z
      	SN_r = rng.exponential(scale=self.Rd, size=SN_counts)  # Exponential for r
      	SN_theta = rng.uniform(0, 2 * np.pi, size=SN_counts)  # Uniform for theta
      	SN_z = rng.laplace(scale=self.H, loc=0, size=SN_counts)  # Laplace for z
    
      	# Generate the event time (ordered from recent to the past)
      	SN_t = np.sort(rng.uniform(time_low, time_up, size=SN_counts))
    
      	# Combine in a list (r, theta, z, t)
      	events = np.column_stack((SN_r, SN_theta, SN_z, SN_t))
      	SN_sim.append(events)
      	
      
      return SN_numbers, SN_sim
      
      
      
      
      
    def Eflux1sim(self, ma, texposure, SN_sim_it):
      """ computes the energy [MeV] and distance to Earth [kpc] of the ALPs that would reach Earth for EACH SN in 1 SIMULATION """
      """ considering an exposure time, texposure [s], for each SN in a given a simulation SN_sim_it
      SN_sim_it: several SNe each one with [rSN, thetaSN, zSN, tSN] """
      
      Erange = []
      for isn, SNevent in enumerate(SN_sim_it):  
	      rSN, thetaSN, zSN, tSN = SNevent
	      E1 = self.Earrival(tSN - texposure, ma, rSN, thetaSN, zSN) # ALP energy at the end of the exposure time (lower)
	      E2 = self.Earrival(tSN, ma, rSN, thetaSN, zSN) # ALP energy at the beginning of the experiment (higher)
	      dSN = self.dSNearth(rSN, thetaSN, zSN)
	      Erange.append([E1, E2, dSN])
      
      return Erange
      

    def factor1simBINS(self, Erange, Ebins):
      """ computes how the flux would be reduced due to the distance [cm^-2] (BINNED), considering the contributions of EVERY SN in 1 SIMULATION """
      
      result_bins = [] # flux per bin
      SN_num_bins = [] # number of SN contributing in each bin
      SN_indices_bins = [] # index of the SN contributing in each bin

      for i in range(len(Ebins) - 1):
	      Ebin_min = Ebins[i]
	      Ebin_max = Ebins[i+1]
	      
	      integral_bin = 0.0
	      SN_num_aux = 0
	      contributing_indices = []  # Indices for this bin

	      for idx, (E1, E2, dSN) in enumerate(Erange):
	      	Eoverlap_min = max(E1, Ebin_min)
	      	Eoverlap_max = min(E2, Ebin_max)

	      	if Eoverlap_min < Eoverlap_max:
	      		contrib = 1 / ( (4*np.pi) * (dSN * self.kpctocm)**2 )
	      		integral = contrib * (Eoverlap_max - Eoverlap_min)
	      		integral_bin += integral
	      		SN_num_aux += 1
	      		contributing_indices.append(idx)

	      result_bins.append(integral_bin)
	      SN_num_bins.append(SN_num_aux)
	      SN_indices_bins.append(contributing_indices)

      return np.array(result_bins), SN_num_bins, SN_indices_bins
    
    

    
    
    
    
    def factorALLsimBINS(self, ma, texposure, SN_sim, Ebins):
      """
      Mean and standard deviation of the flux in each bin [cm^-2]
      considering all SN simulations.

      Args:
        ma: ALP mass [MeV]
        texposure: time of experiment [s]
        SN_sim: simulation list (each simulation has a lot of SNe)
        Ebins: energy bins

      Returns:
        mean_bins: mean of the flux in each bin
        std_bins: standard deviation of the flux in each bin
        all_bin_results: save the flux for each universe
      """
      all_bin_results = []

      for it in range(len(SN_sim)):
	      SN_sim_it = SN_sim[it]
	      Erange = self.Eflux1sim(ma, texposure, SN_sim_it)
	      bins_result, _, _ = self.factor1simBINS(Erange, Ebins)
	      all_bin_results.append(bins_result)

      all_bin_results = np.array(all_bin_results)  # shape: (Nsim, Nbines)
      mean_bins = np.mean(all_bin_results, axis=0)
      std_bins = np.std(all_bin_results, axis=0)

      return mean_bins, std_bins, all_bin_results
      
      
    def factorALLsimBINS_percentile(self, ma, texposure, SN_sim, Ebins):
      """
      Median and 16 and 84 percentiles of the flux in each bin [cm^-2]
      considering all SN simulations.

      Args:
        ma: ALP mass [MeV]
        texposure: time of experiment [s]
        SN_sim: simulation list (each simulation has a lot of SNe)
        Ebins: energy bins

      Returns:
        median_bins: median of the flux in each bin
        p16: 16 percentile
        p84: 84 percentile
        all_bin_results: save the flux for each universe
      """
      all_bin_results = []

      for it in range(len(SN_sim)):
	      SN_sim_it = SN_sim[it]
	      Erange = self.Eflux1sim(ma, texposure, SN_sim_it)
	      bins_result, _, _ = self.factor1simBINS(Erange, Ebins)
	      all_bin_results.append(bins_result)

      all_bin_results = np.array(all_bin_results)  # shape: (Nsim, Nbines)
      median_bins = np.mean(all_bin_results, axis=0)
      p16, p84 = np.percentile(all_bin_results, [16, 84], axis=0)

      return median_bins, p16, p84, all_bin_results
