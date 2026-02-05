#!/usr/bin/env python3
"""
inversedipole implementation written in collaboration by H. Hänninen, H. Schlüter
Copyright 2026

Implements discretization in 2D (r,xbj), and export of the forward operator for inclusive DIS in the dipole picture.
"""

# import math
import os
import sys
from pathlib import Path
import multiprocessing
import scipy.integrate as integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.io import savemat
from timeit import default_timer as timer

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

from data_manage_sigmar import load_rcs, sigmar_rcs_cnt_xbj_points
from dipole_amplitude_managetool import load_edip, edip_dipole_xbins
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
    # .rcs format: qsq, xbj, y, sqrt_s, sigmar, sig_err, theory
    qsq, xbj, y, sqrt_s, sigmar, sig_err, theory_cpp = datum
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


def r_grid_log(conventional=False):
    interpolated_r_grid = []
    if conventional:
        rmin=2e-3 # Nice round lower limit to aim for
        rmax=25 #
    else:
        rmin=1e-2 # Nice round lower limit to aim for
        rmax=100 #
    # r_steps=256 # Paper 1 grid size
    r_steps=127 #
    # r_steps=64 # Good enough with log step!
    r=rmin
    while len(interpolated_r_grid)<r_steps+1:
        interpolated_r_grid.append(r)
        r*=(rmax/rmin)**(1/r_steps) # log grid
    return interpolated_r_grid


def discretize_1D_dipole(interpolated_r_grid, r_vals, S_vals):
    S_interp = InterpolatedUnivariateSpline(r_vals, S_vals, k=1, ext=3)
    S_max = max(S_vals)
    discrete_N_vals = []
    for i in range(len(interpolated_r_grid)-1):
        r_mid = (interpolated_r_grid[i]+interpolated_r_grid[i+1])/2
        # mid point rule interpolation
        discr_N = S_max-S_interp(r_mid)
        if discr_N <= 0:
            print("DISCRETE N NOT POSITIVE!:", discr_N, r_mid)
            raise Exception
        discrete_N_vals.append(discr_N)
    # vec_discrete_N = np.array(discrete_N_vals)
    return discrete_N_vals


def build_discrete_dipole_stack(dipole_edip):
    """Reshape dipole edip into a single column vector, N(r) grouped by xbj bins."""
    interpolated_r_grid = r_grid_log()
    dip_mat = load_edip(dipole_edip)
    x_bins = dip_mat[:,0,0]
    r_vals = dip_mat[0,:,1]
    
    # Dipole scattering amplitude S(r,x) is accessed by index of x:
    # S_x = dip_mat[i_x,:,2]

    # Loop over xbj indices and stack DISCRETIZED DIPOLE AMPLITUDE N
    stacked_dipole_amplitude = []
    for ix, x in enumerate(x_bins):
        S_x = dip_mat[ix,:,2]
        discretized_N = discretize_1D_dipole(interpolated_r_grid, r_vals, S_x)
        stacked_dipole_amplitude += discretized_N

    stacked_dipole_amplitude = np.array(stacked_dipole_amplitude)
    return stacked_dipole_amplitude


def prune_1D_fwdop_array(array):
    # Pruning the z-integrated fwd op data:
    fw_op_datum_r_matrix = []
    fw_op_datum_r_matrix.append(array[:,1]) # vector value riemann sum operator, also has r in 0th col
    fw_op_datum_r_matrix = np.array(fw_op_datum_r_matrix)
    return fw_op_datum_r_matrix


def export_discrete_2d(mass_scheme, data_sigmar_rcs, data_name, ground_truth=None, reference_dip=None):
    """
    2D discretization routine.
    Needs to run 1D discretization in r for each Bjorken-x, and then contstruct the sparce forward operator.
    If a reference dipole is included, N(r,x) needs to be reshaped into a stacked column vector: [N(r,x1),...,N(r,xn)].
    """
    conventional_rgrid = False
    interpolated_r_grid = r_grid_log(conventional_rgrid)
    r_steps = len(interpolated_r_grid)
    print("r_steps", r_steps)
    qm_scheme_name, quark_masses = mass_scheme
    closure_testing = False

    # Build stacked N(r,x) vector from ground_truth .edip data. None if input is None.
    if ground_truth:
        closure_testing = True
        discrete_ground_truth_stack = build_discrete_dipole_stack(ground_truth)
    if reference_dip:
        discrete_refdip_stack = build_discrete_dipole_stack(reference_dip)
        ref_xbj_bins = edip_dipole_xbins(reference_dip) # reference fit dipole might not have all the same bins!

    # Reading Bjorken-x bins and point counts from data for structuring the forward operator
    xbj_bins_and_points_sigmar = sigmar_rcs_cnt_xbj_points(data_sigmar_rcs) # xbj bins and number of datapoints in each is needed to build the sparce fwd operator.
    xbj_ordinal_dict = dict([(xbj, ix) for ix, (xbj, nx) in enumerate(xbj_bins_and_points_sigmar)])
    # print(xbj_bins_and_points_sigmar)
    print([float(x) for (x,n) in xbj_bins_and_points_sigmar])
    n_x_bins = len(xbj_bins_and_points_sigmar)
    # print(n_x_bins)
    qsq_points_total = sum([nx for xbj, nx in xbj_bins_and_points_sigmar])
    # print(qsq_points_total)

    # Discretization loop over datapoints in sigma_r
    sigma02 = 1 # Normalization sigma0/2 is now integrated into the dipole amplitude data, not used here anymore.
    with multiprocessing.Pool(processes=16) as pool:
        fw_op_vals_z_int = pool.starmap(z_inted_fw_sigmar_udscb_riem_logstep, ((datum, (interpolated_r_grid,), sigma02, quark_masses) for datum in data_sigmar_rcs))

    # Building the sparce unified forward operator.
    # If sorting is in order, the starting point is a block column matrix of the 1D operators stacked one after another.
    # Task is to place them in the large stacked operator of size (r_steps*n_x_bins, qsq_points_total)
    sparce_stacked_fwdop = np.zeros((qsq_points_total, (r_steps-1)*n_x_bins))
    for i, array in enumerate(fw_op_vals_z_int):
        fwd_op_datum_row = prune_1D_fwdop_array(array)
        # xbj of datum determines the column block this goes into, and Q^2 of datum the row
        datum = data_sigmar_rcs[i]
        xbj = datum[1]
        ix = xbj_ordinal_dict[xbj] # ordinal of xbj bin
        # inserting the data
        # print(ix, (r_steps-1)*ix, (r_steps-1)*(ix+1), fwd_op_datum_row.shape, sparce_stacked_fwdop.shape)
        sparce_stacked_fwdop[i,(r_steps-1)*ix:(r_steps-1)*(ix+1)] = fwd_op_datum_row

    # Test stacked forward operator
    testing_stack = True
    if testing_stack:
        print(sparce_stacked_fwdop.shape)
        print("Sparce operator rows expected is qsq_points_total = ", qsq_points_total) # each row corresponds to a real data point at some xbj, Q^2
        print("ref dip shape", discrete_refdip_stack.shape, "expected: (nr-1) * nxbins = ", (r_steps-1)*len(ref_xbj_bins))
        if n_x_bins != len(ref_xbj_bins):
            print("Mismatch in number of xbins in rcs data selected, and those available for the reference dipole!", n_x_bins, len(ref_xbj_bins))
        # compute stack op * stack dipole products and compare against sigmar in datum
        calcs = 0
        # rcs format: qsq, xbj, y, sqrt_s, sigmar, sig_err, theory
        dscr_sigmar = np.matmul(sparce_stacked_fwdop, discrete_refdip_stack)
        print(dscr_sigmar.shape)
        for d, s in zip(data_sigmar_rcs, dscr_sigmar):
            # if d[1] < 1e-4 or d[1] > 0.01:
                # continue
            print(calcs, d, d[4], s, s/d[4])
            calcs+=1
        print("When comparing discretization against reference fit dipole, remember to use light quarks only mass scheme!")

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
            for d, s in zip(data_sigmar_rcs, dscr_sigmar):
                print(d, d["theory"], s, s/d["theory"])
        mat_dict = {
            "forward_op_A": sparce_stacked_fwdop,
            "discrete_ground_truth_stack": discrete_ground_truth_stack, 
            "r_grid": interpolated_r_grid,
            "data_sigmar_rcs": data_sigmar_rcs,
            }
    else:
        # Real data fwd operator with comparison reference CKM MV4 dipole
        mat_dict = {
            "forward_op_A": sparce_stacked_fwdop,
            "r_grid": interpolated_r_grid,
            "data_sigmar_rcs": data_sigmar_rcs,
            "discrete_refdip_stack": discrete_refdip_stack,
            "ref_xbj_bins": ref_xbj_bins
            }
    
    base_name+="rtesting_exp2dlog_fwdop_qms_hera_"
    if mass_scheme == "mass_scheme_heracc_charm_only":
        base_name += "CC_charm_only_"
    if conventional_rgrid:
        r_grid_name = "_"+"conventional_r_steps"+str(r_steps)
    else:
        r_grid_name = "_"+"new_r_steps"+str(r_steps)
    savename = base_name+data_name+"_" + qm_scheme_name + r_grid_name +".mat"

    savemat(savename, mat_dict)

    print("Saving export to:", savename)
    
    return 0



def run_export(mass_scheme, ct_groundtruth_dip=None):
    """Construct and export discretized forward operator for the 2D dipole inverse problem."""
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

    ref_fit_bayesMV4_dip = "dipmod_amp_evol_data_bayesMV4_sigma0_inc_large_x_extfrz_r256.edip"
    # ref_fit_bayesMV4_dip = "dipole_modeffect_evol_data_dip_amp_evol_data_bayesMV4_r256_large_x_extension_MVfreeze_r256.edip"

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
    # Discretizing forward operator for reconstruction from real data (HERA II)
    # Including the Casuga-Karhunen-Mäntysaari Bayesian MV4 fit dipole as reference
    print("Discretizing with HERA II data.")
    g_truth = None
    reference_fit_dip = ref_fit_bayesMV4_dip
    for sig_file in hera_sigmar_files:
        sig_file = data_path + sig_file
        print("Loading data file: ", sig_file)
        data_sigmar = load_rcs(sig_file)
        print("Discretizing forward problem for real data file: ", sig_file, mass_scheme[0])
        ret = export_discrete_2d(mass_scheme, data_sigmar, Path(sig_file).stem, ground_truth=g_truth, reference_dip=reference_fit_dip)
    print("Export done with:", mass_scheme[0])

    if ret==-1:
        print("loop error: export not done?")
    return ret


def run_export_closuretesting(mass_scheme, ctest_rcs_edip):
    """Construct and export discretized forward operator for the 2D dipole inverse problem.
    
    Closure testing uses data generated from a known ground truth dipole amplitude, and the
    ground truth is stored together with the testing reduced cross section data."""
    # Load sigma_r .rcs file for the specified closure testing dipole amplitude and effect
    # Closure testing .rcs file also includes the ground truth dipole amplitude .edip data!

    ct_files = [ctest_rcs_edip]
    
    #############################################
    # Discretizing and exporting forward problems
    ret = -1

    # Discretization for CLOSURE TESTING
    # Generate forward operator discretization for data from a known groundtruth dipole amplitude
    print("Closure testing: Discretizing with reference dipole sigma_r data in HERA II bins.")
    for ct_file in ct_files:
        ct_name = Path(ct_file).stem
        ground_truth_rcs_dip = load_edip(TODO)
        print("Discretizing forward problem for sigmar data file: ", ct_file, mass_scheme)
        ret = export_discrete_2d(mass_scheme, data_sigmar, ct_name, ground_truth=ground_truth_dip)
    print("Export done with:", mass_scheme[0], ct_groundtruth_dip)
    
    if ret==-1:
        print("loop error: export not done?")
    return ret


def qmass_scheme_from_dataname(data_name):
    qscheme = None
    if "CKMlightonly" in data_name:
        qscheme = [("standard_light", mass_scheme_standard_light,)]
    return qscheme

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
        # ("mass_scheme_heracc_charm_only", mass_scheme_heracc_charm_only),
    ]

    test_set=[run_settings[1]] # compare with standard light!
    # test_set=[run_settings[0], run_settings[2], run_settings[3]]
    # run_settings=test_set

    # closure_testing = False
    closure_testing = True

    if closure_testing:
        try:
            ct_dip_file = sys.argv[1]
            if os.path.isfile(ct_dip_file):
                print("loading input dipole file: ", ct_dip_file)
            else:
                print("invalid file: ", ct_dip_file)
        except:
            print("Need to give the input dipole file as argument!")
        qm_set = qmass_scheme_from_dataname(ct_dip_file)
        if not qm_set:
            run_settings = test_set
        else:
            run_settings = qm_set


    for setting in run_settings:
        qm_scheme = setting
        try:
            if closure_testing:
                ctest_rcs_edip = ct_dip_file
                r0 += run_export_closuretesting(qm_scheme, ctest_rcs_edip)
            else:
                r0 += run_export(qm_scheme)
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
