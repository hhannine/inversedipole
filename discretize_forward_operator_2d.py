#!/usr/bin/env python3
"""
inversedipole implementation written in collaboration by H. Hänninen, H. Schlüter
Copyright 2026

Implements discretization in 2D (r,xbj), and export of the forward operator for inclusive DIS in the dipole picture.
"""

# import math
import os
from pathlib import Path
import multiprocessing
import scipy.integrate as integrate
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy.io import savemat
from timeit import default_timer as timer

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

from data_manage import load_dipole, get_data, read_sigma02
from deepinelasticscattering import fwd_op_sigma_reduced, fwd_op_sigma_reduced_udscb
from quark_mass_schemes import *




def z_inted_fw_sigmar_udscb_riem_uniftrapez(datum, r_grid, sigma02, quark_masses):
    qsq = datum["qsq"]
    y = datum["y"]
    r_grid=r_grid[0]
    z_inted_points = []

    # Trapezoid rule, uniform interval width
    for i, r in enumerate(r_grid[:-1]):
        delta_r = r_grid[i+1]-r_grid[i]
        z_inted_fw_sigmar_val = integrate.quad(lambda z: sigma02*fwd_op_sigma_reduced_udscb(qsq, y, z, r, quark_masses), 0, 1, epsrel=1e-4)
        if ((i==0) or (i==len(r_grid[:-1])-1)):
            z_inted_points.append((r, z_inted_fw_sigmar_val[0]*delta_r/2))
        else:
            z_inted_points.append((r, z_inted_fw_sigmar_val[0]*delta_r))
    return np.array(z_inted_points)


def z_inted_fw_sigmar_udscb_riem_logstep(datum, r_grid, sigma02, quark_masses):
    qsq = datum["qsq"]
    y = datum["y"]
    r_grid=r_grid[0]
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




def export_discrete_riemann_log(dipfile, mass_scheme, xbj_bin, data_sigmar, parent_data_name, sigma02=1, include_dipole=True):
    interpolated_r_grid = []
    rmin=1e-3 # Nice round lower limit to aim for
    rmax=25 #
    # r_steps=256 # Paper 1 grid size
    # r_steps=128 #
    r_steps=64 # Good enough with log step!


    r=rmin
    while len(interpolated_r_grid)<r_steps+1:
        interpolated_r_grid.append(r)
        r*=(rmax/rmin)**(1/r_steps) # log grid

    if dipfile:
        # Including discretized reference dipole for closure testing / comparison
        # TODO NEEDS TO BE UPDATED FOR THE UNIFIED DIPOLE FILE FORMAT

        data_dipole = load_dipole(dipfile)
        data_dipole = np.sort(data_dipole, order=['xbj','r'])
        xbj_vals = data_dipole["xbj"]
        if xbj_vals[0] != xbj_bin:
            print("xbj bin mismatch in export!.")
            print(xbj_bin, xbj_vals[0])
            print(xbj_vals)
        r_vals = data_dipole["r"]
        S_vals = data_dipole["S"]

        S_interp = InterpolatedUnivariateSpline(r_vals, S_vals, k=1, ext=3)
        discrete_N_vals = []
        # for r in interpolated_r_grid[:-1]:
        for i in range(len(interpolated_r_grid)-1):
            r_mid = (interpolated_r_grid[i]+interpolated_r_grid[i+1])
            # mid point rule interpolation
            discr_N = 1-S_interp(r_mid)
            if discr_N <= 0:
                print("DISCRETE N NOT POSITIVE!:", discr_N, r_mid)
                exit()
            discrete_N_vals.append(discr_N)
        vec_discrete_N = np.array(discrete_N_vals)

    with multiprocessing.Pool(processes=16) as pool:
        fw_op_vals_z_int = pool.starmap(z_inted_fw_sigmar_udscb_riem_uniftrapez, ((datum, (interpolated_r_grid,), sigma02, quark_masses) for datum in data_sigmar))

    fw_op_datum_r_matrix = []
    for array in fw_op_vals_z_int:
        fw_op_datum_r_matrix.append(array[:,1]) # vector value riemann sum operator, also has r in 0th col
    fw_op_datum_r_matrix = np.array(fw_op_datum_r_matrix)

    return fw_op_datum_r_matrix




def export_discrete_2d(mass_scheme, data_sigmar, data_name, ground_truth=None, reference_dip=None):
    """
    2D discretization routine.
    Needs to run 1D discretization in r for each Bjorken-x, and then contstruct the sparce forward operator.
    If a reference dipole is included, N(r,x) needs to be reshaped into a stacked column vector: [N(r,x1),...,N(r,xn)].
    """
    qm_scheme_name, quark_masses = mass_scheme


    Process in bins of xbj and use the old code to build those matrices?
    And then here build them into the sparce matrix?


    # Export
    exp_folder = "./export_hera_data_2d/"
    base_name = exp_folder
    
    if closure_testing:
        # Closure testing simulated data and ground truth dipole
        base_name += "ctest_"
        dscr_check = True
        if dscr_check:
            # Test that discretization agrees with simulated data
            dscr_sigmar = np.matmul(fw_op_datum_r_matrix, vec_discrete_N)
            for d, s in zip(data_sigmar, dscr_sigmar):
                print(d, d["theory"], s, s/d["theory"])
        mat_dict = {
            "forward_op_A": fw_op_datum_r_matrix,
            "discrete_dipole_N": vec_discrete_N, 
            "r_grid": interpolated_r_grid,
            "rcs_data_table": rcs_data_table,
            }
    else:
        # Real data fwd operator with comparison reference CKM MV4 dipole
        mat_dict = {
            "forward_op_A": fw_op_datum_r_matrix,
            "r_grid": interpolated_r_grid,
            "rcs_data_table": rcs_data_table,
            "ref_dipole": ref_dipole_discrete
            }
    
    base_name+="exp2dlog_fwdop_qms_hera_"
    if mass_scheme == "mass_scheme_heracc_charm_only":
        base_name += "CC_charm_only_"
    savename = base_name+data_name+"_" + qm_scheme_name + "_"+"_r_steps"+str(r_steps)+".mat"

    savemat(savename, mat_dict)
    
    return 0



def run_export(mass_scheme, closure_testing, ct_groundtruth_dip=None):
    """Recostruct the dipole amplitude N from simulated reduced cross section data."""
    ###################################
    ### SETTINGS ######################
    ###################################
    charm_only=False

    ####################
    # Reading data files
    data_path = "./data/paper2/" # s binned data in .rcs format
    data_path_cc = data_path
    dipole_path = "./data/paper2/dipoles_unified2d/"
    dipole_files = [i for i in os.listdir(dipole_path) if os.path.isfile(os.path.join(dipole_path, i)) and 'dipole_fit_'+fitname+"_" in i]

    ref_fit_bayesMV4_dip = "dip_amp_evol_data_bayesMV4_sigma0_included_r256.edip"
    # ref_fit_bayesMV4_dip = "dipole_modeffect_evol_data_dip_amp_evol_data_bayesMV4_r256_large_x_extension_MVfreeze_r256.edip"

    # Load sigma_r .rcs file for the specified closure testing dipole amplitude and effect
    # sigmar filenames for ref fits: heraII_reference_dipoles_filtered_bayesMV4-strict_Q_cuts_s318.1_all_xbj_bins.rcs
    # TODO actual closure testing data has not been generated yet, naming scheme not decided yet TODO
    closure_name_base = "TODO"
    if closure_testing:
        closure_testing_sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
            closure_name_base in i and ct_groundtruth_dip in i]    

    s_bins = [318.1, 300.3, 251.5, 224.9]
    s_bin = s_bins[0]
    s_str = "s" + str(s_bin)

    # Load unified HERA II sigma_r .rcs file(s) with specified sqrt(s)
    hera_sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and "heraII_filtered" in i and s_str in i]
    print(hera_sigmar_files)

    if qm_scheme == "mass_scheme_heracc_charm_only":
        # HERA II charm only data settings
        data_path = data_path_cc
        charm_only=True
        hera_sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and "heraII_CC_filtered" in i]
        print("Discreticing for charm production data.")

    
    #############################################
    # Discretizing and exporting forward problems
    ret = -1
    if closure_testing:
        # Discretization for CLOSURE TESTING
        # Generate forward operator discretization for data from a known dipole amplitude
        # "reference" dipole here is the ground truth
        # TODO LOAD dipole and sigmar files in pairs
        print("Closure testing: Discretizing with reference dipole sigma_r data in HERA II bins.")
        for sig_file in sigmar_files:
            ground_truth_dip = load_edip(TODO)
            print("Loading sigma_r rcs file: ", sig_file)
            data_sigmar = load_rcs(sig_file)
            print("Discretizing forward problem for sigmar data file: ", sig_file, mass_scheme)
            ret = export_discrete_2d(mass_scheme, data_sigmar, Path(sig_file).stem, ground_truth=ground_truth_dip)
        print("Export done with:", mass_scheme, closure_testing)
    elif not closure_testing:
        # Discretizing forward operator for reconstruction from real data (HERA II)
        # Including the Casuga-Karhunen-Mäntysaari Bayesian MV4 fit dipole as reference
        print("Discretizing with HERA II data.")
        g_truth = None
        reference_fit_dip = load_edip(ref_fit_bayesMV4_dip)
        for sig_file in hera_sigmar_files:
            print("Loading data file: ", sig_file)
            data_sigmar = load_rcs(sig_file)
            print("Discretizing forward problem for real data file: ", sig_file, mass_scheme)
            ret = export_discrete_2d(mass_scheme, data_sigmar, Path(sig_file).stem, ground_truth=g_truth, reference_dip=reference_fit_dip)
        print("Export done with:", mass_scheme, use_real_data)


    if ret==-1:
        print("loop error: export not done?")
    return ret


# Run multiple settings:
if __name__ == '__main__':
    multiprocessing.freeze_support()
    i=0
    r0=0

    run_settings=[
        ("standard", mass_scheme_standard,),
        ("standard_light", mass_scheme_standard_light,),
        ("mqMpole", mass_scheme_mq_Mpole,),
        ("mqMcharm", mass_scheme_mcharm,),
        ("mqMbottom", mass_scheme_mbottom,),
        ("mqMW", mass_scheme_mW,),
        ("mass_scheme_heracc_charm_only", mass_scheme_heracc_charm_only),
    ]

    test_set=[run_settings[0]]
    run_settings=test_set

    # closure_testing = False
    closure_testing = True

    ctesting_dipoles=[
        "large_x_extension",
    ]

    for setting in run_settings:
        qm_scheme = setting
        if closure_testing:
            ct_groundtruth_dip = ctesting_dipoles[0]
        else:
            ct_groundtruth_dip = None

        try:
            r0 += run_export(qm_scheme, closure_testing, ct_groundtruth_dip=ct_groundtruth_dip)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            i+=1
            print("error occured", i)
            raise
    
    if i==0 and r0==0:
        print("Export runs finished, No errors", i==0)
    else:
        print("Error occured during exports.", i, r0)
    
    exit()
