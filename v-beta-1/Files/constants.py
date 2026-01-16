class Units:
    # to change units
    GeVtomminus1 = 5.076e15  # 1 GeV = 5.076 * 10^15 m^-1
    MeVtokg = 1.79e-30  # 1 MeV/c^2 = 1.79 * 10^-30 kg
    seconds_in_year = 365 * 24 * 60 * 60  # Total seconds in a year
    GeVminus1tos = 6.582e-25  # 1 GeV^-1 = 6.582 * 10^-25 s
    stoMeVminus1 = 1 / (1e3 * GeVminus1tos)  # 1 s = 1.5193 * 10^21 MeV^-1
    gtoMeV = 931.49 / (1.66e-24)  # 1 gram to MeV
    cmminus1toMeV = 1e-13 * 197.33  # 1 cm^-1 to MeV
    mminus1toMeV = cmminus1toMeV / 100  # 1 m^-1 to MeV
    kmminus1toMeV = mminus1toMeV / 1000  # 1 km^-1 to MeV
    MeVtoerg = 1.60218e-6 # 1 MeV = 1.60218e-6 erg
    kpctocm = 3.086e21; # 1kpc = 3.086e21 cm

    @staticmethod
    def describe():
        """print a description of the Units"""
        return (
            f"Units:\n"
            f"1 GeV = {Units.GeVtomminus1} m^-1\n"
            f"1 MeV/c^2 = {Units.MeVtokg} kg\n"
            f"Seconds in a year: {Units.seconds_in_year}\n"
            f"1 GeV^-1 = {Units.GeVminus1tos} s\n"
            f"1 s = {Units.stoMeVminus1} MeV^-1\n"
            f"1 gram = {Units.gtoMeV} MeV\n"
            f"1 cm^-1 = {Units.cmminus1toMeV} MeV\n"
            f"1 m^-1 = {Units.mminus1toMeV} MeV\n"
            f"1 km^-1 = {Units.kmminus1toMeV} MeV\n"
        )






class Constants:
    # Physical Constants
    mn = 938  # MeV nucleon mass
    mp = 938  # MeV proton mass
    mpi = 134.9766 # MeV neutral pion mass
    mmu = 105.7 # MeV muon mass
    mrho = 600 # MeV two-pions exchange contribution by a one-meson exchange with an effective mass
    crho = 1.67 # constant of the two-pion exchange
    
    gAX = 1.28 # axial coupling
    mDelta = 1232 # MeV, Delta-resonance mass
    GammaDelta = 117 # MeV, width of the Delta-resonance
    mpiplus = 139.57 # MeV, charged pion mass
    fpi = 92.4 # MeV
    
    ee = 0.3028
    alphaEM = 1/137.0359 # Fine-structure constant
    
    GMeVminus2 = 6.70711e-45  # Gravitational constant (MeV^-2)
    
    cms = 299792458 # Light speed (m/s)
    hbar_MeV = 6.582119569e-22  # [MeV·s]

    @staticmethod
    def describe():
        """print a description of the Constants"""
        return (
            f"Constants:\n"
            f"mn (nucleon mass): {Constants.mn} MeV\n"
            f"mp (proton mass): {Constants.mp} MeV\n"
            f"G (Gravitational constant): {Constants.GMeVminus2} MeV^-2\n"
        )
