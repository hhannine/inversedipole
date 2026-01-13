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

from pathlib import Path
from scipy.io import loadmat, savemat
from data_manage import load_dipole




if __name__=="__main__":
    if len(sys.argv)==1:
        print("Need to use argument: reformat / dipmod")
        exit()
    elif sys.argv[1]=="reformat":
        # reformat old split data format to unified data file
        ref_dip_name = "bayesMV4"
        # ref_dip_name = "bayesMV5"
        dipole_path = "./data/paper2/dipoles/"
        dipole_files = [i for i in os.listdir(dipole_path) if os.path.isfile(os.path.join(dipole_path, i)) and 'dipole_fit_'+ref_dip_name+"_" in i]
        print(dipole_files)
        n_xbj = len(dipole_files)
        print(n_xbj)
        dipole_data = np.array([load_dipole(dipole_path + i, use_dtype=False) for i in dipole_files])
        # # With use_dtype=False:
        # print(dipole_data[0][0]) # [xbj, r, S]
        # print(dipole_data[0][:,0]) # xbj
        # print(dipole_data[0][:,1]) # r
        # print(dipole_data[0][:,2]) # S

        concatenate = False
        # concatenate = True
        test_formatting = True
        if concatenate:
            # concatenates dipole data arrays into one common array of (xbj, r, S,) tuples: [(xbj, r, S,), ...]
            # this is simpler to store, but will require re-binning of the data for usage
            dip_array = np.concatenate(dipole_data)
            reshape_data = np.reshape(dip_array, (-1, 254, 3))
            print(reshape_data.shape)
            if test_formatting:
                # test formatting
                print("Testing formatting")
                print(dip_array.shape)
                print(dip_array[0])
                print(dip_array[1])
        else:
            # keep dipole dataset arrays in a list of dipole data arrays: [[(xbj1, r, S,),..], [(xbj2, r, S,),..],..]
            dip_array = dipole_data

            if test_formatting:
                print("Testing formatting")
                print(dip_array.shape)

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
        else:
            print("Not saving output!")
        
        # Reformat end
        #
    elif sys.argv[1]=="dipmod":
        # apply a modification to a dipole data file (ideally a reference dipole)
        try:
            dip_file = sys.argv[2]
            if os.path.isfile(dip_file):
                print("loading input dipole file: ", dip_file)
            else:
                print("invalid file: ", dip_file)
        except:
            print("Need to give the input dipole file as argument!")
        
        # Load the dip_file as a mat file
        ref_dip_name = Path(dip_file).stem
        dip_mat = loadmat(dip_file)["dip_array"]

        # print(dip_mat.shape) # (14, 254, 3)

        # need to initialize the (xbj, r) grid to then be able to calculate effects on top of it
        x_bins = dip_mat[:,0,0]
        r_grid = dip_mat[0,:,1]
        print("Input xbj bins: ", x_bins, len(x_bins))

        # implement different effect types to add, file naming scheme
            # - extension to xbj > 0.01
            # - waviness on the saturation front / in x / in r
            # - gaussian peaks here and there
            # - an arbitrary perturbation to lay on top like the shepp--logan phantom?
        opt = "large_x_extension"
        # opt = "wave0" # 0, 1, 2 ~ ?
        # opt = "gaussian" # 0 ~ Gaussian(s), try to reconstruct a number of peaks located in different regimes (simultaneously? 3x3 grid of peaks ~ {small r, mid r, large r} x {small x, mid x, large x})
        # opt = "prescribed_sigma0" # define some logarithmic growth of sigma0(xbj) to try to reconstruct in the closure test
        # opt = "shepplogan" # play with a completely arbitrary overlay if things are working really well?

        if opt == "large_x_extension":
            ext_type = "MVfreeze" # just freeze the IC!! the evolution will not be the same anyway! And this is clearly the easiest to do!
            # ext_type = "MVlike"
            # ext_type = "GBW"
            opt+="_"+ext_type

            bins_to_extend = [0.013, 0.02, 0.032, 0.05, 0.08, 0.13, 0.18, 0.25] # HERA bins above typical IC at 0.01
            x_max = max(x_bins)
            x_bins_to_extend = [x for x in bins_to_extend if x > x_max]
            if not x_bins_to_extend:
                print("x_max in dipole data higher than any extension x_bin?", x_max)
            print("Expected extension bin count: ", len(x_bins)+len(x_bins_to_extend))

            if ext_type == "MVfreeze":
                # COPY IC dipole data bin to extend
                # i_x_ic = x_bins.index(x_max) # not valid for numpy array
                i_x_ic, = np.where(x_bins == x_max)
                print("xmax ", x_max, " at ", i_x_ic[0])
                S_ic = dip_mat[i_x_ic[0],:,2]
                # Need to write the data in the correct format into the new array
                for xbj in x_bins_to_extend:
                    x_ar = np.array([xbj]*len(r_grid))
                    S_extension = [np.array([x_ar, r_grid, S_ic]).T]
                    dip_mat = np.append(dip_mat, S_extension, axis=0)
                # Verify extension success
                x_bins = dip_mat[:,0,0]
                print("xbj bins after extension: ", x_bins, len(x_bins))

                # extension done, jump to exporting to file.
        elif opt == "wave":
            pass
        elif opt == "gaussian":
            pass
            # need to parametrize a set of gaussians and add them on top of the dipole data
        elif opt == "prescribed_sigma0":
            pass
            # need to prescribe a log growth for the dipole in Bjorken-x. Perhaps in can just be a multiplicative scaling that grows with log(1/x)?
            # Define a size at some scale and "evolve" from there, log(1/x) is probably the only simple reasonable option? Maybe double log(log(1/x)) for even slower growth?
            # or log(Q^2)*log(1/x) like the summer plot suggested?
        elif opt == "shepplogan":
            pass
            # TBD whether this is implemented
        
        save_to_file = False
        save_to_file = True
        if save_to_file:
            outfilename = "dipole_modeffect_evol_data_"+ref_dip_name+"_"+opt+"_r256.edip"
            data_dict = {
            "dip_array": dip_mat,
            }
            savemat(outfilename, data_dict)
            print("Saved to file: ", outfilename)
        else:
            print("Not saving output!")
