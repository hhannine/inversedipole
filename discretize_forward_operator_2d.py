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

    qm_scheme_name, quark_masses = mass_scheme

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

    # Filename settings
    str_id_qmass = "_" + qm_scheme_name + "_"
    
    qsq_vals = data_sigmar["qsq"]
    sigmar_vals = data_sigmar["sigmar"]
    sigmar_errs = data_sigmar["sigmarerr"]

    if "reference" in parent_data_name:
        sigmar_theory = data_sigmar["theory"]
    else:
        sigmar_theory = []

    # Export
    exp_folder = "./export_hera_data_2d/"
    base_name = exp_folder+"exp2dlog_fwdop_qms_hera_"
    if mass_scheme == "mass_scheme_heracc_charm_only":
        base_name += "CC_charm_only_"
    if include_dipole:
        # Real data, reference theory simulated data, and dipole
        dscr_sigmar = np.matmul(fw_op_datum_r_matrix, vec_discrete_N)
        for d, s in zip(data_sigmar, dscr_sigmar):
            print(d, d["theory"], real_sigma*s, real_sigma*s/d["theory"])
        # if "reference" in parent_data_name:
        mat_dict = {
            "forward_op_A": fw_op_datum_r_matrix,
            "discrete_dipole_N": vec_discrete_N, 
            "r_grid": interpolated_r_grid,
            "qsq_vals": qsq_vals,
            "sigmar_vals": sigmar_vals,
            "sigmar_errs": sigmar_errs,
            "sigmar_theory": sigmar_theory,
            # "real_sigma": real_sigma
            }
        savemat(base_name+parent_data_name+str_id_qmass+str_unity_sigma02+"_r_steps"+str(r_steps)+"_xbj"+str(xbj_bin)+".mat", mat_dict)
    else:
        # Real data without dipole
        mat_dict = {
            "forward_op_A": fw_op_datum_r_matrix,
            "r_grid": interpolated_r_grid,
            "qsq_vals": qsq_vals,
            "sigmar_vals": sigmar_vals,
            # TODO CORRELATED UNCERTAINTIES ALSO!
            "sigmar_errs": sigmar_errs,
            }
        savemat(base_name+parent_data_name+str_id_qmass+str_unity_sigma02+"_r_steps"+str(r_steps)+".mat", mat_dict)
    
    return 0


# TODO
def export_discrete_2d(TODO):
    """
    2D discretization routine.
    Needs to run 1D discretization in r for each Bjorken-x, and then contstruct the sparce forward operator.
    If a reference dipole is included, N(r,x) needs to be reshaped into a stacked column vector: [N(r,x1),...,N(r,xn)].
    """



def run_export(mass_scheme, use_real_data, fitname_i=None):
    """Recostruct the dipole amplitude N from simulated reduced cross section data."""

    ###################################
    ### SETTINGS ######################
    ###################################

    use_unity_sigma0 = True
    charm_only=False
    use_charm = False

    # #        0        1        2        3           4
    fits = ["MV", "MVgamma", "MVe", "bayesMV4", "bayesMV5"]
    if not fitname_i:
        fitname = fits[3]
    # else:
    #     fitname = fits[fitname_i]

    ####################
    # Reading data files
    data_path = "./data/paper2/" # s binned data in .rcs format
    data_path_cc = data_path
    dipole_path = "./data/paper2/dipoles_unified2d/"
    dipole_files = [i for i in os.listdir(dipole_path) if os.path.isfile(os.path.join(dipole_path, i)) and 'dipole_fit_'+fitname+"_" in i]

    if use_charm:
        sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
            'sigmar_'+fitname+"_dipole-lightpluscharm" in i]
    else:
        sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
            'sigmar_'+fitname+"_dipole." in i]
    # print(sigmar_files)
    

    s_bins = [318.1, 300.3, 251.5, 224.9]
    s_bin = s_bins[0]
    s_str = "s" + str(s_bin)

    hera_sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and "heraII_filtered" in i and s_str in i]

    use_ref_dip = False
    # use_ref_dip = True
    # hera_sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and "heraII_reference_dipoles_filtered_bayesMV4-strict_Q_cuts" in i and s_str in i]
    # hera_sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and "heraII_reference_dipoles_filtered_bayesMV4-wide_Q_cuts" in i and s_str in i]
    print(hera_sigmar_files)

    if qm_scheme == "mass_scheme_heracc_charm_only":
        data_path = data_path_cc
        charm_only=True
        hera_sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and "heraII_CC_filtered" in i]
        print("Discreticing for charm production data.")

    
    #############################################
    # Discretizing and exporting forward problems
    ret = -1
    if use_real_data:
        if use_ref_dip:
            print("Discretizing with reference dipole sigma_r data in HERA II bins.")
            xbj_bin_vals = [float(Path(i).stem.split("xbj")[1]) for i in hera_sigmar_files]
            print(xbj_bin_vals)
            for sig_file in hera_sigmar_files:
                print("Loading reference file: ", sig_file)
                # sigma02=read_sigma02(data_path +sig_file)
                # sigma02=13.9/(1/2.56819) # the binned files don't save the sigma02. convert 13.9 mb to GeV^-2
                sigma02=37.0628 # own refit of the normalization! (difference from correlated uncertainties?)
                data_sigmar = get_data(data_path + sig_file, simulated="reference-dipole", charm=charm_only)
                xbj_bin = float(Path(sig_file).stem.split("xbj")[1])
                if xbj_bin>0.01:
                    print("Reference dipole not available!")
                    inc_dip = False
                    ref_dipole = None
                else:
                    inc_dip = True
                    if [i for i in dipole_files if str(xbj_bin) in i][0]:
                        ref_dipole = dipole_path+[i for i in dipole_files if str(xbj_bin) in i][0]
                    else:
                        print("No ref_dipole file found for xbj=", xbj_bin)
                        continue
                    print("inc_dipole", xbj_bin, str(xbj_bin) in ref_dipole, str(xbj_bin) in sig_file)
                print("Discretizing forward problem for real data file: ", sig_file, " at xbj=", xbj_bin, mass_scheme)
                # continue
                # export_discrete(None, xbj_bin, data_sigmar, Path(sig_file).stem, sigma02, include_dipole=False, use_charm=use_charm)
                ret = export_discrete_riemann_log(ref_dipole, mass_scheme, xbj_bin, data_sigmar, Path(sig_file).stem, sigma02=sigma02, include_dipole=inc_dip, use_unity_sigma0=use_unity_sigma0)
            print("Export done with:", mass_scheme, use_real_data)
        else:
            print("Discretizing with HERA II data.")
            xbj_bin_vals = [float(Path(i).stem.split("xbj")[1]) for i in hera_sigmar_files]
            print(xbj_bin_vals)
            for sig_file in hera_sigmar_files:
                print("Loading data file: ", sig_file)
                data_sigmar = get_data(data_path + sig_file, simulated=False, charm=charm_only)
                xbj_bin = float(Path(sig_file).stem.split("xbj")[1])
                print("Discretizing forward problem for real data file: ", sig_file, " at xbj=", xbj_bin, mass_scheme)
                # export_discrete(None, xbj_bin, data_sigmar, Path(sig_file).stem, sigma02, include_dipole=False, use_charm=use_charm)
                ret = export_discrete_riemann_log(None, mass_scheme, xbj_bin, data_sigmar, Path(sig_file).stem, include_dipole=False, use_unity_sigma0=use_unity_sigma0)
            print("Export done with:", mass_scheme, use_real_data)
        # exit()
    else:
        # Simulated data
        print("Simulated data with quark mass schemes not implemented! Use STANDARD ONLY for now!") # Fits only use the standard scheme so need to use that for 1-to-1 comparison!
        mass_scheme = "standard"
        for dip_file in dipole_files:
            xbj_bin = float(Path(dip_file).stem.split("xbj")[1].split("r")[0])
            data_sigmar = get_data(data_path + sig_file, simulated="reference-dipole", charm=charm_only)
            data_sigmar_binned = data_sigmar[data_sigmar["xbj"]==xbj_bin]
            if data_sigmar_binned.size==0:
                print("NO DATA FOUND IN THIS BIN: ", xbj_bin, " in file: ", sig_file)
                continue
            print("Discretizing forward problem for dipole file: ", dip_file, " at xbj=", xbj_bin, ", Datapoints N=", data_sigmar_binned.size)
            # export_discrete(data_path+dip_file, xbj_bin, data_sigmar_binned, Path(sigmar_files[0]).stem, sigma02, use_charm=use_charm)
            ret = export_discrete_riemann_log(data_path+dip_file, mass_scheme, xbj_bin, data_sigmar_binned, Path(sigmar_files[0]).stem, use_unity_sigma0=use_unity_sigma0)
        # exit()

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

    fitname_i = None
    # use_real_data = False
    use_real_data = True

    for setting in run_settings:
        qm_scheme = setting 

        try:
            r0 += run_export(qm_scheme, use_real_data)
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
