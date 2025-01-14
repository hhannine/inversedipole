#!/usr/bin/env python3
"""
inversedipole implementation written in collaboration by H. Hänninen, A. Kykkänen and H. Schlüter
Copyright 2024

deepinelasticscattering module implements calculation of DIS cross sections in the dipole picture.
"""

import math
import numpy
import scipy.integrate as integrate
import scipy.special as special

# Universal constants
Nc = 3.0
CF = 4.0/3.0
alphaem = 1.0/137.0
lambdaqcd = 0.241 #GeV
structurefunfac = 1./((2*math.pi)**2 * alphaem)

# Calculation / scattering dependent constants
sumef_light = 6.0/9.0 # light quarks uds only.
qmass_light = 0.14 # this was the old effective light quark mass?
icx0 = 0.01 # Default initial scale for LO.


def Sq(z):
    return z**2

class Dipole:
    def __init__(self, q0sq, ec, gamma):
        self.q0sq = q0sq
        self.ec = ec
        self.gamma = gamma
    
    def scattering_S_x0(self, r):
        return math.exp(-(r**2*self.q0sq)/4 * math.log(1/(r*lambdaqcd)+self.ec*math.e))

def sigma_reduced(x,Q,y):
    """Calculate reduced cross section, standard definition."""
    fac = structurefunfac * Sq(Q)
    FL = fac * sigma_L
    FT = fac * sigma_T
    F2 = FL + FT
    fy = y**2/(1+(1-y)**2)
    sigmar = F2 - fy * FL
    return sigmar

def dipole_scattering_amplitude_S(r,x):
    S = 0
    return S

def dipole_amplitude_N(r,x):
    N = 1 - dipole_scattering_amplitude_S(r,x)
    return N

def rapidity_X(x, Qsq):
    X = x/icx0*1.0/Qsq
    return X

def bessel_K0(r):
    K0 = special.kn(0, r)
    return K0

def bessel_K1(r):
    K1 = special.kn(1, r)
    return K1

def dis_quarkantiquark_dipole_wavefunction_pol_T(Q,z,r,qmass):
    """virtual photon T splitting wf"""
    Qsq = Q**2
    # rapidity_X = rapidity_X(x,Qsq)
    
    af = math.sqrt( Sq(Q)*z*(1.0-z) + Sq(qmass) )
    impactfac = (1.0-2.0*z+2.0*Sq(z))*Sq(af * bessel_K1(af * r)) + Sq(qmass * bessel_K0(af * r))
    psi_squared = impactfac * r # r here is from the jacobian from the change to polar coordinates. Old choice to have it here.
    return psi_squared

def dis_quarkantiquark_dipole_wavefunction_pol_L(Q,z,r,qmass):
    """virtual photon L splitting wf"""
    Qsq = Q**2
    # rapidity_X = rapidity_X(x,Qsq)
    af = math.sqrt( Qsq*z*(1-z) + qmass**2 )
    impactfac = 4.0*(Q*(z)*(1.0-z)*bessel_K0(af * r))**2
    # res = (1.0-(dipole_scattering_amplitude_S(r, rapidity_X))) * impactfac * r  # Old code for reference. Includes the dipole which is now done elsewhere
    psi_squared = impactfac * r # r here is from the jacobian from the change to polar coordinates. Old choice to have it here.
    return psi_squared

def sigma_T(x,Q):
    """Calculate DIS cross section for T polarization."""
    sigmat = 0
    return sigmat

def sigma_L(x,Q):
    """Calculate DIS cross section for L polarization."""
    sigmal = 0
    return sigmal
