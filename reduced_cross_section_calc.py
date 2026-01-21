#!/usr/bin/env python3
"""
inversedipole implementation written in collaboration by H. Hänninen, H. Schlüter
Copyright 2026

Computes reduced DIS cross sections in the dipole picture from arbitrary evolved dipole amplitude.

Calculations are implemented for the standard continuous problem using MC integration, to enable
the generation of reduced cross section data from a dipole amplitude independently from the 
discretized problem and forward operator. This prevents inverse crime.

This module has two core functionalities:
    - verification of the accuracy of this new Python implementation of the reduced cross section calculation
      against the older C++ implementation.
    - generation of reduced cross section datasets for closure testing of the reconstruction method(s).
"""

import os
import sys

import numpy as np

from pathlib import Path
from scipy.io import loadmat, savemat
from scipy.interpolate import InterpolatedUnivariateSpline

from deepinelasticscattering import reduced_cross_section, discrete_reduced_cross_section, discretize_dipole_data_log, discretize_dipole_data_linear


def test_sigmar():
    """Run a preset accuracy test against an established dataset."""
    # Load the dip_file as a mat file
    dip_file = "./dip_amp_evol_data_bayesMV4_r256.edip"
    ref_dip_name = Path(dip_file).stem
    dip_mat = loadmat(dip_file)["dip_array"]
    x_bins = dip_mat[:,0,0]
    print("Input xbj bins: ", x_bins, len(x_bins))
    r_grid = dip_mat[0,:,1] # same for each xbj
    S_data_list = [dip_mat[i,:,2] for i in range(len(x_bins))]
    S_interp_list = [InterpolatedUnivariateSpline(r_grid, S_vals, ext=3) for S_vals in S_data_list]
    S_interp_dict = dict(zip(x_bins, S_interp_list))
    print("Test interp", S_interp_dict[x_bins[0]](r_grid[0])) # test the first interpolator: call by the key of xbj value, which is then an interpolator function S(r)

    # get reference sigma_r data
    # .rcs data: qsq, xbj, y, sqrt_s, sigmar, sig_err, theory
    ref_sigr_file = "./data/paper2/heraII_reference_dipoles_filtered_bayesMV4-strict_Q_cuts_s318.1_all_xbj_bins.rcs"
    # ref_sigr_file = "./data/paper2/heraII_reference_dipoles_filtered_bayesMV4-wide_Q_cuts_s318.1_all_xbj_bins.rcs"
    ref_sigr_data = loadmat(ref_sigr_file)["sigma_r_data"]
    print("Loaded N=", len(ref_sigr_data), "data points.")

    # settings: standard mass scheme, no charm

    # Reference sigma02 from the Bayesian LO fit
    sigma02=37.0628 # LO Bayes MV 4 refit, strict cuts
    # sigma02=36.8195 # LO Bayes MV 4 refit, wide cuts
    # sigma02=36.3254 # LO Bayes MV 5 refit, strict cuts
    # sigma02=36.0176 # LO Bayes MV 5 refit, wide cuts

    test_continuous = False
    test_discrete = True
    use_log = True
    # use_log = False

    if test_discrete:
        # Prepare discrete computation
        if use_log:
            discr_r, discr_N_dict = discretize_dipole_data_log(r_grid, S_interp_dict, x_bins)
        else:
            discr_r, discr_N_dict = discretize_dipole_data_linear(r_grid, S_interp_dict, x_bins)

    for datum in ref_sigr_data:
    # for datum in ref_sigr_data[100:110]:
        xbj = datum[1]
        S_interp = S_interp_dict[xbj]
        sigmar_theory_cont = 0
        if test_continuous:
            sigmar_theory_cont = reduced_cross_section(datum, r_grid, S_interp, sigma02)
        sigmar_theory_discr = 0
        if test_discrete:
            discr_N = discr_N_dict[xbj]
            sigmar_theory_discr = discrete_reduced_cross_section(datum, discr_r, discr_N, sigma02)
        sigmar = datum[4]
        sigmar_cpp = datum[6]
        if test_continuous:
            print(datum, sigmar, sigmar_cpp, sigmar_theory_cont, sigmar_theory_cont/sigmar_cpp, abs(sigmar_theory_cont/sigmar_cpp-1) < 1e-2)
        elif test_discrete:
            print(datum, sigmar, sigmar_cpp, sigmar_theory_discr, sigmar_theory_discr/sigmar_cpp, abs((sigmar_theory_discr/sigmar_cpp)-1) < 1e-2, abs((sigmar_theory_discr/sigmar_cpp)-1) < 1e-3)


def generate_sigmar(dip_file):
    """Generate reduced cross section data from a dipole amplitude data file, using HERA data points."""
    # Load dipole data, make interpolators
    ref_dip_name = Path(dip_file).stem
    print("Generate sigma_r data from: ", ref_dip_name)
    dip_mat = loadmat(dip_file)["dip_array"]
    x_bins = dip_mat[:,0,0]
    r_grid = dip_mat[0,:,1] # same for each xbj
    S_data_list = [dip_mat[i,:,2] for i in range(len(x_bins))]
    S_interp_list = [InterpolatedUnivariateSpline(r_grid, S_vals, ext=3) for S_vals in S_data_list]
    S_interp_dict = dict(zip(x_bins, S_interp_list))

    # Read HERA data to have the points to compute at
    hera_data_file = "./data/paper2/heraII_filtered_s318.1_all_xbj_bins.rcs"
    hera_sigr_data = loadmat(hera_data_file)["sigma_r_data"]
    print("Loaded N=", len(hera_sigr_data), "HERA II data points from: ", hera_data_file)

    # Multiply sigma02 into dipole amplitude data so that it mathces the real data reconstruction in kind, and we don't need to have it here.
    sigma02 = 1

    generated_sigmar_data = []
    for datum in hera_sigr_data:
        qsq, xbj, y, sqrt_s, sigmar, sig_err, theory_cpp = datum
        try:
            S_interp = S_interp_dict[xbj]
        except KeyError:
            print("skipping at xbj=", xbj)
            continue
        sigmar_theory_cont = reduced_cross_section(datum, r_grid, S_interp, sigma02)
        generated_sigmar_data.append([qsq, xbj, y, sqrt_s, sigmar, sig_err, sigmar_theory_cont])
        print(len(generated_sigmar_data))
    generated_sigmar_data = np.array(generated_sigmar_data)

    # Export output

    save_to_file = False
    save_to_file = True
    if save_to_file:
        out_name = "generated_sigmar_heraIIbins_" + ref_dip_name + "_all_xbj_bins.rcs"
        data_dict = {
        "sigma_r_data": generated_sigmar_data,
        }
        savemat(out_name, data_dict)
        print("Saved to file: ", out_name)
    else:
        print("Not saving output!")



if __name__ == "__main__":
    acc_testing = True
    # acc_testing = False

    if acc_testing:
        print("Testing accuracy of MC integration against C++ implementation data.")
        test_sigmar()
    else:
        # get args
        try:
            dip_file = sys.argv[1]
            if os.path.isfile(dip_file):
                print("loading input dipole file: ", dip_file)
            else:
                print("invalid file: ", dip_file)
        except:
            print("Need to give the input dipole file as argument!")
            exit()
        generate_sigmar(dip_file)
