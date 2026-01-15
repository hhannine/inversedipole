#!/usr/bin/env python3
"""
inversedipole implementation written by H. Hänninen, University of Jyväskylä.
Copyright 2026

data_manage_sigmar module has the functionalities:
    - read filtered and binned HERA reduced cross section data and export it in a unified format, which tracks xbj, Q^2, y, sqrt(s), sigma_r, errors
    - analogously with closure testing reduced cross section data in the split binned file format
"""

import os
import sys

import numpy as np

from pathlib import Path
from scipy.io import savemat

if __name__=="__main__":
    dip_types = [
        "heraII_filtered", 
        "heraII_CC_filtered", 
        "heraII_reference_dipoles_filtered_bayesMV4-strict_Q_cuts", # heraII_reference_dipoles_filtered_bayesMV4-strict_Q_cuts_s300.3_xbj0.0002
        "heraII_reference_dipoles_filtered_bayesMV4-wide_Q_cuts",
        "heraII_reference_dipoles_filtered_bayesMV5-strict_Q_cuts",
        "heraII_reference_dipoles_filtered_bayesMV5-wide_Q_cuts",
        ]
    # sqrt_s = 318.1
    sqrt_s = 300.3
    dip_name = dip_types[0] + "_s" + str(sqrt_s)
    
    data_path = "./data/paper2/separate binned files"
    # Load sigma_r data files, filter by name and sqrt(s)
    sigr_data_files = [os.path.join(data_path, i) for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and dip_name in i]
    # sort files by xbj
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in sigr_data_files]
    sigr_data_files = [x for _, x in sorted(zip(xbj_bins, sigr_data_files))]
    sigr_data = [np.loadtxt(file) for file in sigr_data_files]
    print("Data files loaded: ", len(sigr_data))

    # Data files store different columns:
    # - heraII:                     qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err
    # - heraII_CC:                  qsq, xbj, y, sigmar, sig_err -- (files wrongly have the same header as heraII inclusive)
    # - reference_dipoles_filtered: qsq, xbj, y, sigmar, sig_err, theory -- (all but one wrongly have the same header as heraII)

    # Filter and store the data as: qsq, xbj, y, sqrt_s, sigmar, sig_err, theory

    # Build data into the new array:
    if "heraII_filtered" in dip_name:
        # input columns are: qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err
        new_data = []
        for x_bin_data in sigr_data:
            for datum in x_bin_data:
                # print(datum) # list of 8 elements
                qsq = datum[0]
                xbj = datum[1]
                y = datum[2]
                root_s = sqrt_s
                sigmar = datum[3]
                sig_err = datum[4]
                theory = 0
                new_data.append([qsq, xbj, y, root_s, sigmar, sig_err, theory])
        new_data = np.array(new_data)
        print("Combined data in (N, 7) array:", new_data.shape)
    elif "heraII_CC_filtered" in dip_name:
        # input columns are: qsq, xbj, y, sigmar, sig_err
        new_data = []
        for x_bin_data in sigr_data:
            for datum in x_bin_data:
                # print(datum) # list of 5 elements
                qsq = datum[0]
                xbj = datum[1]
                y = datum[2]
                root_s = sqrt_s
                sigmar = datum[3]
                sig_err = datum[4]
                theory = 0
                new_data.append([qsq, xbj, y, root_s, sigmar, sig_err, theory])
        new_data = np.array(new_data)
        print("Combined data in (N, 7) array:", new_data.shape)
    elif "heraII_reference_dipoles_filtered" in dip_name:
        # input columns are: qsq, xbj, y, sigmar, sig_err, theory
        new_data = []
        for x_bin_data in sigr_data:
            for datum in x_bin_data:
                # print(datum) # list of 6 elements
                qsq = datum[0]
                xbj = datum[1]
                y = datum[2]
                root_s = sqrt_s
                sigmar = datum[3]
                sig_err = datum[4]
                theory = datum[5]
                new_data.append([qsq, xbj, y, root_s, sigmar, sig_err, theory])
        new_data = np.array(new_data)
        print("Combined data in (N, 7) array:", new_data.shape)
    else:
        print("Data type not identified from dip_name: ", dip_name)


    # Export output

    save_to_file = False
    save_to_file = True
    if save_to_file:
        out_name = dip_name + "_all_xbj_bins.rcs"
        data_dict = {
        "sigma_r_data": new_data,
        }
        savemat(out_name, data_dict)
        print("Saved to file: ", out_name)
    else:
        print("Not saving output!")
