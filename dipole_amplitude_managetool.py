#!/usr/bin/env python3
"""
inversedipole implementation written by H. Hänninen, University of Jyväskylä.
Copyright 2026

dipole_amplitude_managetool module has two core functionalities:
    - reading older dipole amplitude data files to consolidate them into a new unified format
    - reading unified dipole amplitude data files and applying artificial effects for closure testing of the reconstruction
"""

import os
import sys

import numpy as np

from scipy.io import savemat
from data_manage import load_dipole




if __name__=="__main__":
    if len(sys.argv)==1:
        print("Need to use argument: reformat / dipmod")
        exit()
    elif sys.argv[1]=="reformat":
        # reformat old split data format to unified data file
        # ref_dip_name = "bayesMV4"
        ref_dip_name = "bayesMV5"
        dipole_path = "./data/paper2/dipoles/"
        dipole_files = [i for i in os.listdir(dipole_path) if os.path.isfile(os.path.join(dipole_path, i)) and 'dipole_fit_'+ref_dip_name+"_" in i]
        print(dipole_files)
        n_xbj = len(dipole_files)
        print(n_xbj)
        dipole_data = [load_dipole(dipole_path + i) for i in dipole_files]
        concatenate = False
        test_formatting = True
        if concatenate:
            # concatenates dipole data arrays into one common array of (xbj, r, S,) tuples: [(xbj, r, S,), ...]
            # this is simpler to store, but will require re-binning of the data for usage
            dip_array = np.concatenate(dipole_data)
        else:
            # keep dipole dataset arrays in a list of dipole data arrays: [[(xbj1, r, S,),..], [(xbj2, r, S,),..],..]
            dip_array = np.array(dipole_data)
            if test_formatting:
                # test formatting
                print(dip_array.shape)
                print(dip_array[0][0])
                print(dip_array[1][0])

        # save_to_file = False
        save_to_file = True
        if save_to_file:
            outfilename = "dip_amp_evol_data_"+ref_dip_name+"_r256.edip"
            # Matlab compatible data format is probably preferrable, so it would be more straightforward to work on the data in there as well.
            data_dict = {
            "dip_array": dip_array,
            }
            savemat(outfilename, data_dict)
            print("Saved to file: ", outfilename)
    elif sys.argv[1]=="dipmod":
        # apply a modification to a dipole data file (ideally a reference dipole)
        TODO

