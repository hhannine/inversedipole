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

from scipy.io import loadmat, savemat



def acc_testing():
    """Run a preset accuracy test against an established dataset."""
    # Load the dip_file as a mat file
    dip_file = "./dip_amp_evol_data_bayesMV4_r256.edip"
    ref_dip_name = Path(dip_file).stem
    dip_mat = loadmat(dip_file)["dip_array"]

    # get reference sigma_r data
    dependency: need to rewrite the sigma_r data files.

    # settings: standard mass scheme, no charm

    loop over reference data points (or some subsample)


def generate_sigmar(todo):
    """Generate reduced cross section data from a dipole amplitude data file, using HERA data points."""
    todo


if __name__ == "__main__":
    acc_testing = True

    if acc_testing:
        test_sigmar()
    else:
        # get args
        generate_sigmar()
