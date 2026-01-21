#!/usr/bin/env python3
"""
inversedipole implementation written in collaboration by H. Hänninen, A. Kykkänen and H. Schlüter
Copyright 2026

deepinelasticscattering module implements calculation of DIS cross sections in the dipole picture.
"""

import math
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import vegas
from scipy import stats
from scipy.interpolate import CubicSpline, PchipInterpolator, griddata, RegularGridInterpolator, make_interp_spline, InterpolatedUnivariateSpline

from data_manage import load_dipole, get_data
from quark_mass_schemes import *

# Universal constants
Nc = 3.0
NF = 3
CF = 4.0/3.0
alphaem = 1.0/137.0
lambdaqcd = 0.241 #GeV
structurefunfac = 1./((2*math.pi)**2 * alphaem)

# Calculation / scattering dependent constants
sumef_light = 6.0/9.0 # light quarks uds only.
qmass_light = 0.14 # this is the old effective light quark mass
qmass_charm = 1.35 # Kinda ~mean of the nlo massive fit values (1.29...1.49)
icx0 = 0.01 # Default initial scale for LO.
alpha_scaling_C2_ = 1

# r_min = 1e-6
r_min = 0
r_max = 30


def Sq(z):
    return z**2

class Dipole:
    def __init__(self, q0sq, ec, gamma):
        self.q0sq = q0sq
        self.ec = ec
        self.gamma = gamma
    
    def scattering_S_x0(self, r):
        return math.exp(-(r**2*self.q0sq)/4 * math.log(1/(r*lambdaqcd)+self.ec*math.e))

def fwd_op_sigma_reduced(Qsq,y,z,r):
    """Calculate reduced cross section, forward operator definition.
    
    Proper reduced cross section is calculated as sigma_r = fwd_op * N(r,x)."""
    # print("here Q y z r",Q,y,z,r)
    qmass=qmass_light
    sumef=6/9
    fac = structurefunfac * Qsq
    FL = fac * fwd_op_sigma_L(Qsq,z,r,qmass,sumef)
    FT = fac * fwd_op_sigma_T(Qsq,z,r,qmass,sumef)
    F2 = FL + FT
    fy = y**2/(1+(1-y)**2)
    fwd_op_sigmar = F2 - fy * FL
    return fwd_op_sigmar

def fwd_op_sigma_reduced_udsc(Qsq,y,z,r):
    """Calculate reduced cross section, forward operator definition.
    
    Proper reduced cross section is calculated as sigma_r = fwd_op * N(r,x)."""
    sumef=6/9
    efc = 4/9
    fac = structurefunfac * Qsq
    FL_uds = fac * fwd_op_sigma_L(Qsq,z,r,qmass_light,sumef)
    FT_uds = fac * fwd_op_sigma_T(Qsq,z,r,qmass_light,sumef)
    FL_c = fac * fwd_op_sigma_L(Qsq,z,r,qmass_charm,efc)
    FT_c = fac * fwd_op_sigma_T(Qsq,z,r,qmass_charm,efc)
    FL = FL_uds + FL_c
    F2 = FL_uds + FL_c + FT_uds + FT_c
    fy = y**2/(1+(1-y)**2)
    fwd_op_sigmar = F2 - fy * FL
    return fwd_op_sigmar

def fwd_op_sigma_reduced_udscb(Qsq,y,z,r,quark_masses):
    """Calculate reduced cross section, forward operator definition.
    
    Proper reduced cross section is calculated as sigma_r = fwd_op * N(r,x)."""
    fac = structurefunfac * Qsq
    quarks = quark_masses.values()
    FL = 0
    FT = 0
    for (mq, efq) in quarks:
        FL += fac * fwd_op_sigma_L(Qsq,z,r,mq,efq)
        FT += fac * fwd_op_sigma_T(Qsq,z,r,mq,efq)
    F2 = FL + FT
    fy = y**2/(1+(1-y)**2)
    fwd_op_sigmar = F2 - fy * FL
    return fwd_op_sigmar

def fwd_op_FL_LO(Qsq,z,r):
    qmass=qmass_light
    sumef=6/9
    fac = structurefunfac * Qsq
    FL = fac * fwd_op_sigma_L(Qsq,z,r,qmass,sumef)
    return FL

def fwd_op_FT_LO(Qsq,z,r):
    qmass=qmass_light
    sumef=6/9
    fac = structurefunfac * Qsq
    FT = fac * fwd_op_sigma_T(Qsq,z,r,qmass,sumef)
    return FT

def dipole_scattering_amplitude_S(r,x):
    S = 0
    return S

def dipole_amplitude_N(r,x):
    N = 1 - dipole_scattering_amplitude_S(r,x)
    return N

def alpha_bar_fixed(rsq,x):
    return 0.190986

def alpha_bar_parent(rsq,x):
    scalefactor = 4.0*alpha_scaling_C2_
    alphas_mu0=2.5    # mu0/lqcd
    alphas_freeze_c=0.2
    b0 = (11.0*Nc - 2.0*NF)/3.0

    AlphaSres = 4.0*math.pi / (b0 *
        math.log(
            math.pow(
                math.pow(alphas_mu0, 2.0/alphas_freeze_c) + math.pow(scalefactor/(rsq*lambdaqcd*lambdaqcd), 1.0/alphas_freeze_c), alphas_freeze_c
                )
            )
        )
    return Nc/math.pi*AlphaSres

def rapidity_X(x, Qsq):
    X = x/icx0*1.0/Qsq
    return X

def bessel_K0(r):
    K0 = special.kn(0, r)
    return K0

def bessel_K1(r):
    K1 = special.kn(1, r)
    return K1

def dis_quarkantiquark_dipole_wavefunction_pol_T(Qsq,z,r,qmass):
    """virtual photon T splitting wf"""
    # rapidity_X = rapidity_X(x,Qsq)
    
    af = math.sqrt( Qsq*z*(1.0-z) + qmass**2 )
    impactfac = (1.0-2.0*z+2.0*Sq(z))*Sq(af * bessel_K1(af * r)) + Sq(qmass * bessel_K0(af * r))
    psi_squared = impactfac * r # r here is from the jacobian from the change to polar coordinates. Old choice to have it here.
    return psi_squared

def dis_quarkantiquark_dipole_wavefunction_pol_L(Qsq,z,r,qmass):
    """virtual photon L splitting wf"""
    # rapidity_X = rapidity_X(x,Qsq)
    # print("af_interior", Qsq*z*(1-z) + qmass**2)
    af = math.sqrt( Qsq*z*(1-z) + qmass**2 )
    impactfac = 4.0*Qsq*((z)*(1.0-z)*bessel_K0(af * r))**2
    # res = (1.0-(dipole_scattering_amplitude_S(r, rapidity_X))) * impactfac * r  # Old code for reference. Includes the dipole which is now done elsewhere
    psi_squared = impactfac * r # r here is from the jacobian from the change to polar coordinates. Old choice to have it here.
    return psi_squared

def fwd_op_sigma_T(Qsq,z,r,qmass,sumef):
    """Calculate DIS cross section for T polarization.
    
    This corresponds to the 'LLOp', 'TLOp' etc functions of the old code calculating bare cross sections.
    Structure functions are F_T = structurefunfac*Q^2*Sigma_T(TLOp_*)

    Previously the integration was done at this stage, but now we want to do it last, so this just multiplies by the correct 
    Jacobian and other constants.
    """
    fac=4.0*Nc*alphaem/Sq(2.0*math.pi)*sumef
    integrand = dis_quarkantiquark_dipole_wavefunction_pol_T(Qsq,z,r,qmass) ### DIPOLE AMPLITUDE WOULD GO HERE, BUT NOW WE SEPARATE IT from the fwd operator
    sigmat = fac*2.0*math.pi*integrand #*nlodis_config::MAXR*integral ----- integration done last
    return sigmat #fac*2.0*M_PI*nlodis_config::MAXR*integral;

def fwd_op_sigma_L(Qsq,z,r,qmass,sumef):
    """Calculate DIS cross section for L polarization.
    
    This corresponds to the 'LLOp', 'TLOp' etc functions of the old code calculating bare cross sections.
    Structure functions are F_L = structurefunfac*Q^2*Sigma_L(LLOp_*)
    """
    fac=4.0*Nc*alphaem/Sq(2.0*math.pi)*sumef
    integrand = dis_quarkantiquark_dipole_wavefunction_pol_L(Qsq,z,r,qmass) ### DIPOLE AMPLITUDE WOULD GO HERE, BUT NOW WE SEPARATE IT from the fwd operator
    sigmal = fac*2.0*math.pi*integrand #*nlodis_config::MAXR*integral ----- integration done last
    return sigmal


class Sigmar_calc:
    def __init__(self, qsq, y, S_interp, sigma02):
        self.qsq = qsq
        self.y = y
        self.S_interp = S_interp
        self.sigma02 = sigma02
    
    def discrete_int_kernel_sigmar(self, args):
        """For each r, computes the z-integral of the DIS cross section kernel"""
        z = args
    
    def mc_integrand_sigmar(self, args):
        [r, z] = args
        try:
            res = []
            for (ri, zi) in zip(r,z):
                if zi>1:
                    print("z>1, exit")
                    exit()
                res.append(self.sigma02*fwd_op_sigma_reduced(self.qsq, self.y, zi, ri)*(1-self.S_interp(ri)))
            return np.array(res)
        except:
            return np.array([self.sigma02*fwd_op_sigma_reduced(self.qsq, self.y, z, r)*(1-self.S_interp(r)),])
        
    def mc_integrand_FL(self, args):
        [r, z] = args
        try:
            res = []
            for (ri, zi) in zip(r,z):
                if zi>1:
                    print("z>1, exit")
                    exit()
                res.append(self.sigma02*fwd_op_FL_LO(self.qsq, zi, ri)*(1-self.S_interp(ri)))
            return np.array(res)
        except:
            return np.array([self.sigma02*fwd_op_FL_LO(self.qsq, z, r)*(1-self.S_interp(r)),])
    
    def mc_integrand_FT(self, args):
        [r, z] = args
        try:
            res = []
            for (ri, zi) in zip(r,z):
                if zi>1:
                    print("z>1, exit")
                    exit()
                res.append(self.sigma02*fwd_op_FT_LO(self.qsq, zi, ri)*(1-self.S_interp(ri)))
            return np.array(res)
        except:
            return np.array([self.sigma02*fwd_op_FT_LO(self.qsq, z, r)*(1-self.S_interp(r)),])


def reduced_cross_section(sigmar_datum, r_grid, S_interp, sigma02):
    """Compute reduced cross section from given dipole amplitude data.
    Using the continuous formulation of the problem to enable data generation independently of the
    discretized formulation of the problem used to solve the inversion problem."""
    qsq, xbj, y, sqrt_s, sigmar, sig_err, theory_cpp = sigmar_datum
    r_min = r_grid[0]
    r_max = r_grid[-1]
    sigmar_py = integrate.dblquad(lambda z, r: sigma02*fwd_op_sigma_reduced(qsq, y, z, r)*(1-S_interp(r)), r_min, r_max, 0, 1, epsrel=1e-3)
    return sigmar_py[0]


# def discretize_dipole_data_linear(r_grid, S_interp_dict, x_bins):
    # TODO MAKE A LINEAR GRID IMPLEMENTATION AS A COMPARISON AND CONTROL!!! Will definitely help with evaluating the accuracy gains




def discretize_dipole_data_log(r_grid, S_interp_dict, x_bins):
    rmin=2e-3 # todo: re-test with log grid
    rmax=25 # todo: re-test with log grid
    # r_steps=256 # old linear grid step count
    # r_steps=128 # SEEMS TO WORK EXCELLENTLY!!
    # r_steps=32 # EVEN THIS SEEMS TO WORK??!?!?
    r_steps=30 # this is close to the limit for 1% numerical accuarcy for the discretization

    r=rmin
    interpolated_r_grid = []
    while len(interpolated_r_grid)<r_steps+1:
        interpolated_r_grid.append(r)
        # r+=(rmax-rmin)/r_steps # linear uniform grid step
        r*=(rmax/rmin)**(1/r_steps) # log grid

    discrete_N_list = []
    for xbin in x_bins:
        S_interp = S_interp_dict[xbin]
        discrete_N_vals = []
        for i in range(len(interpolated_r_grid)-1):
            r_mid = (interpolated_r_grid[i]+interpolated_r_grid[i+1])/2
            # r_mid = interpolated_r_grid[i] # TODO TEMP LOWER POINT TESTING
            # mid point rule interpolation
            discr_N = 1-S_interp(r_mid)
            if discr_N <= 0:
                print("DISCRETE N NOT POSITIVE!:", discr_N, r_mid)
                exit()
                # alternatively just append 0
                # discr_N = 0
            discrete_N_vals.append(discr_N)
        vec_discrete_N = np.array(discrete_N_vals)
        discrete_N_list.append(vec_discrete_N)
    
    discr_r = interpolated_r_grid
    discr_N_dict = dict(zip(x_bins, discrete_N_list))
    return discr_r, discr_N_dict


def z_inted_fw_sigmar_udscb_riem_logstep_unifieddatum(datum, r_grid, sigma02, quark_masses):
    # .rcs data: qsq, xbj, y, sqrt_s, sigmar, sig_err, theory
    qsq = datum[0]
    y = datum[2]
    z_inted_points = []

    # Rieman sum, MIDPOINT rule, LOG interval width
    # Delta_r_i = r_{i+1} - r_i
    # sum Delta_r_i * f(r_i + (r_{i+1}-r_i)/2) = sum Delta_r_i f((r_{i+1}+r_i)/2)
    for i, r in enumerate(r_grid[:-1]):
        delta_r = r_grid[i+1]-r_grid[i]
        r_midpoint = (r_grid[i+1]+r_grid[i])/2
        z_inted_fw_sigmar_val = integrate.quad(lambda z: sigma02*fwd_op_sigma_reduced_udscb(qsq, y, z, r_midpoint, quark_masses), 0, 1, epsrel=1e-4)
        z_inted_points.append((r, z_inted_fw_sigmar_val[0]*delta_r))
    return np.array(z_inted_points)


def discrete_reduced_cross_section(sigmar_datum, r_grid, discrete_N, sigma02, quark_masses = mass_scheme_standard_light):
    """Compute reduced cross section from given dipole amplitude data with discrete formulation."""
    qsq, xbj, y, sqrt_s, sigmar, sig_err, theory_cpp = sigmar_datum

    # TODO multiprocessing
    # with multiprocessing.Pool(processes=16) as pool:
        # fw_op_vals_z_int = pool.starmap(z_inted_fw_sigmar_udscb_riem_uniftrapez, ((datum, (r_grid,), sigma02, quark_masses) for datum in data_sigmar))

    # fw_op_datum_r_matrix = []
    # for array in fw_op_vals_z_int:
    #     fw_op_datum_r_matrix.append(array[:,1]) # vector value riemann/uniform trapez sum operator, also has r in 0th col
    # fw_op_datum_r_matrix = np.array(fw_op_datum_r_matrix)

    # Without parallelization
    fw_op_vals_z_int = z_inted_fw_sigmar_udscb_riem_logstep_unifieddatum(sigmar_datum, r_grid, sigma02, quark_masses)
    fw_op_datum_r_matrix = fw_op_vals_z_int[:,1]

    # dscr_sigmar = np.matmul(fw_op_datum_r_matrix, discrete_N)
    dscr_sigmar = np.dot(fw_op_datum_r_matrix, discrete_N)
    print(sigmar, dscr_sigmar)
    return dscr_sigmar


# Main

def main():
    """DIS reduced cross section computation testing and development."""

    # Load data.
    # data_sigmar = get_data("./data/simulated-lo-sigmar_DIPOLE_TAKEN.txt")
    data_sigmar = get_data("./data/simulated_lo_sigmar_with_FL_FT.dat")
    qsq_vals = data_sigmar["qsq"]
    y_vals = data_sigmar["y"]
    sigma02=48.4781

    # We need a dipole initial guess?
    # data_dipole = load_dipole("./data/readable-lo-dipolescatteringamplitude_S.txt")
    data_dipole = load_dipole("./data/readable-lo_dip_S-logstep_r.dat")
    data_dipole = np.sort(data_dipole, order=['xbj','r'])
    xbj_vals = data_dipole["xbj"]
    r_vals = data_dipole["r"]
    S_vals = data_dipole["S"]

    # Interpolator testing
    # dipole_interpolation = RegularGridInterpolator([xbj_vals, r_vals], S_vals) # THIS DOESNT WORK FOR FIXED XBJ??
    # dipole_interp_in_r = CubicSpline(r_vals, S_vals)
    # dipole_interp_in_r = PchipInterpolator(r_vals, S_vals)

    # Initialize a dipole scattering amplitude interpolator
    # S_interp = make_interp_spline(r_vals, S_vals, k=1)
    S_interp = CubicSpline(r_vals, S_vals)
    # S_interp = PchipInterpolator(r_vals, S_vals)
    # exit()


    # We need to test the forward operator acting on a dipole to get a calculation of the reduced cross section
    # 'b = Ax', i.e. sigma_r = integrate(fwd_op*N,{r,z}), where the operator needs to integrate over r and z.

    # Calculate sigma_r = fwd_op * N over a dataset, which is then compared against the simulated data, at a fixed x.
    # scipy integrate: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html#scipy.integrate.dblquad

    # Vegas
    n_itn = 20
    n_eval = 10**3
    vegas_integ = vegas.Integrator([[r_min, r_max], [0,1]])
    print("xbj,    qsq,       y,   sigmar,    FL_LO,    FT_LO,   sigmr_test[0],   sigmr_test3[0],   sigmr_test3[0]/sigmar")

    for datum in data_sigmar[-5:]:
    # for datum in data_sigmar[-10:]:
        (xbj, qsq, y, sigmar, fl, ft) = datum
        sigmr_test = integrate.dblquad(lambda z, r: sigma02*fwd_op_sigma_reduced(qsq, y, z, r)*(1-S_interp(r)), r_min, r_max, 0, 1, epsrel=1e-3)

        sig_calc = Sigmar_calc(qsq, y, S_interp, sigma02)
        sigmr_test_veg = vegas_integ(sig_calc.mc_integrand_sigmar, nitn=n_itn, neval=n_eval)
        fl_veg = vegas_integ(sig_calc.mc_integrand_FL, nitn=n_itn, neval=n_eval)
        ft_veg = vegas_integ(sig_calc.mc_integrand_FT, nitn=n_itn, neval=n_eval)

        print(xbj, qsq, y, " - ", sigmar, sigmr_test[0],  sigmr_test_veg[0] ,sigmr_test_veg[0]/sigmar , " - ", fl, ft, fl_veg, ft_veg, " - ", fl_veg/fl, ft_veg/ft )

    return 0

if __name__=="__main__":
    main()
