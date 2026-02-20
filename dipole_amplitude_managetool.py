#!/usr/bin/env python3
"""
inversedipole implementation written by H. Hänninen, University of Jyväskylä.
Copyright 2026

dipole_amplitude_managetool module has two core functionalities:
    - reading older dipole amplitude data files to consolidate them into a new unified format
    - reading unified dipole amplitude data files and applying artificial effects for closure testing of the reconstruction
"""

import math
import os
import sys

import scipy
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.io import loadmat, savemat
from data_manage import load_dipole


def load_edip(file):
    """Load unified dipole data file into a numpy array."""
    dip_mat = loadmat(file)["dip_array"]
    return dip_mat

def edip_dipole_xbins(file):
    dip_mat = load_edip(file)
    x_bins = dip_mat[:,0,0]
    return x_bins

def gaussian_peak(x, loc=0, scale=1):
    return np.exp(-(x-loc)**2/(2*scale**2))

def gaussian_features(peaks, r_grid, x_bins, log_r_x_grid, print_peaks = False):
    # construct the functions for these features on the log_r_x_grid
    peak_outp = np.zeros((len(r_grid),len(x_bins)))
    x_min = min(x_bins)
    lr, lx = log_r_x_grid
    lx_min = min(lx)
    for peak in peak_features:
        print(peak)
        r_mean, r_stdev, x_mean, x_stdev, amp = peak
        x_mean = x_mean / x_min
        x_stdev = x_stdev / x_min
        peak_f_r = gaussian_peak(lr, loc=math.log(r_mean), scale=math.log(r_stdev))
        peak_f_x = gaussian_peak(lx - lx_min, loc=math.log(x_mean), scale=math.log(x_stdev))
        pdf_prod = amp*np.outer(peak_f_r, peak_f_x)
        # print(np.max(pdf_prod))
        peak_outp += pdf_prod
    if print_peaks:
        print(peak_outp.shape)
        for i in range(len(x_bins)):
            # check max and min values
            max_r = max(peak_outp[:,i])
            min_r = min(peak_outp[:,i])
            max_ri, = np.where(peak_outp[:,i] == max_r)
            min_ri, = np.where(peak_outp[:,i] == min_r)
            print(x_bins[i], r_grid[max_ri], r_grid[min_ri], max_r, min_r)
        plt.imshow(peak_outp, aspect="auto")
        plt.colorbar()
        plt.show()
        print("exit")
        exit()
    return peak_outp


if __name__=="__main__":
    if len(sys.argv)==1:
        print("Need to use argument: reformat / dipmod")
        exit()
    elif sys.argv[1]=="reformat":
        # reformat old split data format to unified data file
        ref_dip_name = "bayesMV4"
        # ref_dip_name = "bayesMV5"
        # dipole_path = "./data/paper2/dipoles/"
        dipole_path = "./data/paper2/dipoles/patch/"
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
            # outfilename = "dip_amp_evol_data_"+ref_dip_name+"_r256.edip"
            outfilename = "dip_amp_evol_data_"+ref_dip_name+"patch_r256.edip"
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
        closure_testing = False
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
        # print(r_grid)
        print("Input xbj bins: ", x_bins, len(x_bins))
        # r_x_grid = (r_grid, x_bins)
        log_r_x_grid = (np.log(r_grid), np.log(x_bins))
        # print(log_r_x_grid[0])
        # print(log_r_x_grid[1])

        # implement different effect types to add, file naming scheme
            # - extension to xbj > 0.01
            # - waviness on the saturation front / in x / in r
            # - gaussian peaks here and there
            # - an arbitrary perturbation to lay on top like the shepp--logan phantom?
        # opt = "patch" # run both sigma0_included and large_x_extension with MVfreeze
        # opt = "large_x_extension"
        # opt = "small_x_extension"
        # opt = "sigma0_included"
        # opt = "hera_mimic" # mimic some gaussian-like peaks that might be present in the preliminary reconstruction
        # opt = "gaussian"
        # opt = "gaussian_single"
        # opt = "gaussian_front"
        # opt = "gaussian_front2"
        # opt = "linear_sigma0"
        opt = "wave" # 0, 1, 2 ~ ?
        # opt = "shepplogan" # play with a completely arbitrary overlay if things are working really well?

        if opt == "patch":
            # Multiply sigma02 into the dipole amplitude data to make it self contained for data generation tasks
            strict_Q_cuts = True
            if "bayesMV4" in ref_dip_name:
                if strict_Q_cuts:
                    sigma02=37.0628 # LO Bayes MV 4 refit, strict cuts
                else:
                    sigma02=36.8195 # LO Bayes MV 4 refit, wide cuts
            elif "bayesMV5" in ref_dip_name:
                if strict_Q_cuts:
                    sigma02=36.3254 # LO Bayes MV 5 refit, strict cuts
                else:
                    sigma02=36.0176 # LO Bayes MV 5 refit, wide cuts
            else:
                print("CKM reference dipole not recognized, others not supported, ref_dip_name: ", ref_dip_name)
            if not strict_Q_cuts:
                opt+="_wideQcuts"
            # Loop over xbj bins of dipole data, and multiply S by sigma02
            print("Multiplying sigma02 into dipole data.")
            for i, x in enumerate(x_bins):
                dip_mat[i,:,2] *= sigma02
            
            bins_to_extend = [0.013, 0.02, 0.032, 0.05, 0.08, 0.13, 0.18, 0.25, 0.4, 0.65] # HERA bins above typical IC at 0.01
            x_max = max(x_bins)
            x_bins_to_extend = [x for x in bins_to_extend if x > x_max]
            if not x_bins_to_extend:
                print("x_max in dipole data higher than any extension x_bin?", x_max)
            print("Expected extension bin count: ", len(x_bins)+len(x_bins_to_extend))

            # COPY IC dipole data bin to extend
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
        
        elif opt == "large_x_extension":
            ext_type = "MVfreeze" # just freeze the IC!! the evolution will not be the same anyway! And this is clearly the easiest to do!
            # ext_type = "MVlike"
            # ext_type = "GBW"
            opt+="_"+ext_type

            bins_to_extend = [0.013, 0.02, 0.032, 0.05, 0.08, 0.13, 0.18, 0.25, 0.4, 0.65] # HERA bins above typical IC at 0.01
            x_max = max(x_bins)
            x_bins_to_extend = [x for x in bins_to_extend if x > x_max]
            if not x_bins_to_extend:
                print("x_max in dipole data higher than any extension x_bin?", x_max)
            print("Expected extension bin count: ", len(x_bins)+len(x_bins_to_extend))

            if ext_type == "MVfreeze":
                # COPY IC dipole data bin to extend
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
        elif opt == "small_x_extension":
            # Freeze evolution at small-x to quickly patch the reference data for the full HERA range.
            # Would be more accurate to compute the dipole with the BK equation in cpp
            ext_type = "smallxfreeze" # just freeze the IC!! the evolution will not be the same anyway! And this is clearly the easiest to do!
            opt+="_"+ext_type
            bins_to_extend = [3.98e-05, 8e-05] # small x HERA bins
            x_min = min(x_bins)
            # x_bins_to_extend = [x for x in bins_to_extend if x < x_min]
            x_bins_to_extend = bins_to_extend
            if not x_bins_to_extend:
                print("x_min in dipole data lower than any extension x_bin?", x_min)
            print("Expected extension bin count: ", len(x_bins)+len(x_bins_to_extend))

            if ext_type == "smallxfreeze":
                # COPY smallest x dipole data bin to extend
                i_x_low, = np.where(x_bins == x_min)
                print("x_min ", x_min, " at ", i_x_low[0])
                S_low = dip_mat[i_x_low[0],:,2]
                # Need to write the data in the correct format into the new array
                for xbj in reversed(x_bins_to_extend):
                    # loop over reversed list since insertion has to be done in the beginning of the data array
                    # and increasing order of xbj must be preserved
                    x_ar = np.array([xbj]*len(r_grid))
                    S_extension = [np.array([x_ar, r_grid, S_low]).T]
                    # dip_mat = np.insert(dip_mat, 0, S_extension, axis=0) # insert at the begining
                    dip_mat = np.insert(dip_mat, 1, S_extension, axis=0) # insert AFTER THE SMALLEST BIN (1)
                # Verify extension success
                x_bins = dip_mat[:,0,0]
                print("xbj bins after extension: ", x_bins, len(x_bins))
                print("INSERTION AFTER THE SMALLEST XBJ, DOUBLE CHECK SORTING!!")

                # extension done, jump to exporting to file.
        elif opt == "sigma0_included":
            # Multiply sigma02 into the dipole amplitude data to make it self contained for data generation tasks
            strict_Q_cuts = True
            if "bayesMV4" in ref_dip_name:
                if strict_Q_cuts:
                    sigma02=37.0628 # LO Bayes MV 4 refit, strict cuts
                else:
                    sigma02=36.8195 # LO Bayes MV 4 refit, wide cuts
            elif "bayesMV5" in ref_dip_name:
                if strict_Q_cuts:
                    sigma02=36.3254 # LO Bayes MV 5 refit, strict cuts
                else:
                    sigma02=36.0176 # LO Bayes MV 5 refit, wide cuts
            else:
                print("CKM reference dipole not recognized, others not supported, ref_dip_name: ", ref_dip_name)
            if not strict_Q_cuts:
                opt+="_wideQcuts"
            # Loop over xbj bins of dipole data, and multiply S by sigma02
            print("Multiplying sigma02 into dipole data.")
            for i, x in enumerate(x_bins):
                dip_mat[i,:,2] *= sigma02
            # sigma02 inclusion mod done, jump to exporting
        elif opt == "hera_mimic":
            closure_testing = True
            # there are 4 peak-like features to add, 3 additive, 1 subtractive
            peak_features = [
                # (r_center, r_width, x_center, x_width, +- amplitude * pdf_scaling_factor)
                (0.85, 2, 8e-5, 9e-5, 2.5),
                (3.9, 2, 8e-5, 6e-5, 5),
                (3.9, 2, 0.0005, 8e-5, 7),
                (5.5, 4.5, 5e-3, 1e-4, (-5.75)), # dip down at large x
            ]
            # construct the functions for these features on the log_r_x_grid
            peak_outp = np.zeros((len(r_grid),len(x_bins)))
            x_min = min(x_bins)
            lr, lx = log_r_x_grid
            lx_min = min(lx)
            for peak in peak_features:
                r_mean, r_stdev, x_mean, x_stdev, amp = peak
                x_mean = x_mean / x_min
                x_stdev = x_stdev / x_min
                peak_f_r = gaussian_peak(lr, loc=math.log(r_mean), scale=math.log(r_stdev))
                peak_f_x = gaussian_peak(lx - lx_min, loc=math.log(x_mean), scale=math.log(x_stdev))
                pdf_prod = amp*np.outer(peak_f_r, peak_f_x)
                # print(np.max(pdf_prod))
                peak_outp += pdf_prod
            # sum combined peak output onto the dipole S data, just taking it on the original grid
            #   sign of effect flips from N to S! (pos. effect for N is neg. for S): N = S_max - S(r,x) - S_mod
            #   S_i = dip_mat[x[i],:,2]
            print_peaks = False
            if print_peaks:
                print(peak_outp.shape)
                for i in range(len(x_bins)):
                    # check max and min values
                    max_r = max(peak_outp[:,i])
                    min_r = min(peak_outp[:,i])
                    max_ri, = np.where(peak_outp[:,i] == max_r)
                    min_ri, = np.where(peak_outp[:,i] == min_r)
                    print(x_bins[i], r_grid[max_ri], r_grid[min_ri], max_r, min_r)
                plt.imshow(peak_outp, aspect="auto")
                plt.colorbar()
                plt.show()
                exit()
            # Loop over dipole data by xbj bin, and add mod effect output
            for i in range(len(x_bins)):
                mod_i = peak_outp[:,i]
                # S_i = dip_mat[x_bins[i],:,2]
                dip_mat[i,:,2] += -mod_i
        elif opt == "gaussian":
            closure_testing = True
            peak_features = [
                # (r_center, r_width, x_center, x_width, +- amplitude * pdf_scaling_factor)
                (0.1, 0.2, 8e-5, 6e-5, 5),
                (0.1, 0.2, 8e-4, 20e-5, 5),
                (0.1, 0.2, 8e-3, 100e-5, 5),
                (1, 0.8, 8e-5, 6e-5, 5),
                (1, 0.8, 8e-4, 10e-5, 15),
                (1, 0.8, 8e-3, 15e-5, 5),
                (10, 3, 8e-5, 6e-5, 15),
                (10, 3, 8e-4, 10e-5, 15),
                (10, 3, 8e-3, 15e-5, 15),
            ]
            peak_outp = gaussian_features(peak_features, r_grid, x_bins, log_r_x_grid, print_peaks=False)
            # Loop over dipole data by xbj bin, and add mod effect output
            for i in range(len(x_bins)):
                mod_i = peak_outp[:,i]
                # S_i = dip_mat[x_bins[i],:,2]
                dip_mat[i,:,2] += -mod_i
        elif opt == "gaussian_single":
            closure_testing = True
            peak_features = [
                # (r_center, r_width, x_center, x_width, +- amplitude * pdf_scaling_factor)
                (3, 1.5, 8e-4, 15e-5, 12),
            ]
            peak_outp = gaussian_features(peak_features, r_grid, x_bins, log_r_x_grid, print_peaks=False)
            # Loop over dipole data by xbj bin, and add mod effect output
            for i in range(len(x_bins)):
                mod_i = peak_outp[:,i]
                # S_i = dip_mat[x_bins[i],:,2]
                dip_mat[i,:,2] += -mod_i
        elif opt == "gaussian_front":
            closure_testing = True
            peak_features = [
                # (r_center, r_width, x_center, x_width, +- amplitude * pdf_scaling_factor)
                (3, 3, 8e-5, 8e-5, 40),
                (3, 3, 8e-4, 8e-5, 40),
                (3, 3, 8e-3, 8e-5, 40),
            ]
            peak_outp = gaussian_features(peak_features, r_grid, x_bins, log_r_x_grid, print_peaks=False)
            # Loop over dipole data by xbj bin, and add mod effect output
            for i in range(len(x_bins)):
                mod_i = peak_outp[:,i]
                # S_i = dip_mat[x_bins[i],:,2]
                dip_mat[i,:,2] = -mod_i
        elif opt == "gaussian_front2":
            closure_testing = True
            peak_features = [
                # (r_center, r_width, x_center, x_width, +- amplitude * pdf_scaling_factor)
                (0.8, 0.8, 2.5e-5, 8e-5, 25),
                (0.8, 0.8, 2.5e-4, 8e-5, 25),
                (0.8, 0.8, 2.5e-3, 8e-5, 25),
                (6, 3, 8e-5, 8e-5, 40),
                (6, 3, 8e-4, 8e-5, 40),
                (6, 3, 8e-3, 8e-5, 40),
            ]
            peak_outp = gaussian_features(peak_features, r_grid, x_bins, log_r_x_grid, print_peaks=False)
            # Loop over dipole data by xbj bin, and add mod effect output
            for i in range(len(x_bins)):
                mod_i = peak_outp[:,i]
                # S_i = dip_mat[x_bins[i],:,2]
                dip_mat[i,:,2] = -mod_i
        elif opt == "linear_sigma0":
            # Linear multiplicative scaling that grows with log(1/x)
            #   another with log(1/Q^2)*log(1/x) like the summer plot suggested? 
            #   (both xbj and Q^2 were decreasing in unison)
            x1=0.01
            x2=3.98e-5
            y1=0.75
            y2=1.25
            k = (y2 - y1)/(math.log(1/x2) - math.log(1/x1))
            b = y1 - k*math.log(1/x1)
            linear_xbj_sigma0_scalings = -k * np.log(x_bins) + b
            # print(linear_xbj_sigma0_scalings)
            for i in range(len(x_bins)):
                mod_i = linear_xbj_sigma0_scalings[i]
                # S_i = dip_mat[x_bins[i],:,2]
                dip_mat[i,:,2] *= mod_i
        elif opt == "wave":
            # wave perturbation to the dipole
            # (1 - amp * sin(log(r)/wavelen)**2) * S
            amp = 1.5
            wavelen = 0.3
            wave_fac = 1 - amp * r_grid/5 * np.sin(np.log(r_grid)/wavelen)**2
            for i in range(len(x_bins)):
                # S_i = dip_mat[x_bins[i],:,2]
                dip_mat[i,:,2] *= wave_fac
            # plt.plot(np.log(r_grid), wave_fac)
            # plt.plot(r_grid, wave_fac)
            # plt.show()
            # print("exit")
            # exit()
        elif opt == "shepplogan":
            pass
            # plt.imshow(peak_outp, aspect="auto")
            # plt.colorbar()
            # plt.show()
            # print("exit")
            # exit()


        if closure_testing:
            outpath = "./data/paper2/closure_testing/"
        else:
            outpath = ""

        # save_to_file = False
        save_to_file = True
        if save_to_file:
            if closure_testing:
                outfilename = "ctest_dipeff_" + opt + "_" + ref_dip_name + ".edip"
            else:
                outfilename = "dipole_modeffect_evol_data_"+ref_dip_name+"_"+opt+".edip"
            data_dict = {
            "dip_array": dip_mat,
            }
            savemat(outpath + outfilename, data_dict)
            print("Saved to file: ", outpath + outfilename)
            print("For closure testing, next run: python reduced_cross_section_calc.py", outpath + outfilename)
        else:
            print("Not saving output!")
    else:
        print("Need to give argument: reformat / dipmod !")
