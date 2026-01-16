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

from pathlib import Path
from scipy.io import loadmat, savemat
from scipy.interpolate import InterpolatedUnivariateSpline

from deepinelasticscattering import reduced_cross_section


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

    # Reference sigma02 from the Bayesian LO fit
    sigma02=37.0628 # LO Bayes MV 4 refit, strict cuts
    # sigma02=36.8195 # LO Bayes MV 4 refit, wide cuts
    # sigma02=36.3254 # LO Bayes MV 5 refit, strict cuts
    # sigma02=36.0176 # LO Bayes MV 5 refit, wide cuts

    # settings: standard mass scheme, no charm

    for datum in ref_sigr_data:
    # for datum in ref_sigr_data[100:110]:
        xbj = datum[1]
        S_interp = S_interp_dict[xbj]
        sigmar_theory_cont = reduced_cross_section(datum, r_grid, S_interp, sigma02)
        sigmar = datum[4]
        sigmar_cpp = datum[6]
        print(datum, sigmar, sigmar_cpp, sigmar_theory_cont, sigmar_theory_cont/sigmar_cpp, abs(sigmar_theory_cont/sigmar_cpp-1) < 1e-2)


def generate_sigmar(todo):
    """Generate reduced cross section data from a dipole amplitude data file, using HERA data points."""
    todo


if __name__ == "__main__":
    acc_testing = True

    if acc_testing:
        print("Testing accuracy of MC integration against C++ implementation data.")
        test_sigmar()
    else:
        # get args
        generate_sigmar()
