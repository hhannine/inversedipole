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


def z_inted_fw_sigmar(datum, r_grid, sigma02):
    if len(datum)==6:
        (xbj, qsq, y, sigmar, fl, ft) = datum
    else:
        (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = datum
    r_grid=r_grid[0]
    z_inted_points = []
    # Riemann sum, lower boundary evaluation
    # for i, r in enumerate(r_grid[:-1]):
    #     delta_r = r_grid[i+1]-r_grid[i]
    #     z_inted_fw_sigmar_val = integrate.quad(lambda z: sigma02*fwd_op_sigma_reduced(qsq, y, z, r), 0, 1, epsrel=1e-4)
    #     z_inted_points.append((r, z_inted_fw_sigmar_val[0]*delta_r))
    # return np.array(z_inted_points)
    
    # Calculate integrand at interval end points.
    for i, r in enumerate(r_grid[:-1]):
        z_inted_fw_sigmar_val = integrate.quad(lambda z: sigma02*fwd_op_sigma_reduced(qsq, y, z, r), 0, 1, epsrel=1e-4)
        z_inted_points.append([r, z_inted_fw_sigmar_val[0]])    
    # Trapezoid rule of discrete integration
    fwd_op_trapez = np.zeros((len(r_grid[:-1]), len(r_grid[:-1])))
    for i, r in enumerate(r_grid[:-2]):
        delta_r = r_grid[i+1]-r_grid[i]
        z_int_i = z_inted_points[i][1]
        z_int_ii = z_inted_points[i+1][1]
        # print(z_int_i, z_int_ii)
        # trap_op = [z_int_i/2.*delta_r, z_int_ii/2.*delta_r]
        fwd_op_trapez[i][i] = z_int_i/2.*delta_r
        fwd_op_trapez[i][i+1] = z_int_ii/2.*delta_r
    return fwd_op_trapez

def z_inted_fw_sigmar_riem_uniftrapez(datum, r_grid, sigma02):
    if len(datum)==6:
        (xbj, qsq, y, sigmar, fl, ft) = datum
    else:
        (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = datum
    r_grid=r_grid[0]
    z_inted_points = []
    # Riemann sum, lower boundary evaluation
    # for i, r in enumerate(r_grid[:-1]):
    #     delta_r = r_grid[i+1]-r_grid[i]
    #     z_inted_fw_sigmar_val = integrate.quad(lambda z: sigma02*fwd_op_sigma_reduced(qsq, y, z, r), 0, 1, epsrel=1e-4)
    #     z_inted_points.append((r, z_inted_fw_sigmar_val[0]*delta_r))
    # return np.array(z_inted_points)

    # Trapezoid rule, uniform interval width
    for i, r in enumerate(r_grid[:-1]):
        delta_r = r_grid[i+1]-r_grid[i]
        z_inted_fw_sigmar_val = integrate.quad(lambda z: sigma02*fwd_op_sigma_reduced(qsq, y, z, r), 0, 1, epsrel=1e-4)
        if ((i==0) or (i==len(r_grid[:-1])-1)):
            z_inted_points.append((r, z_inted_fw_sigmar_val[0]*delta_r/2))
        else:
            z_inted_points.append((r, z_inted_fw_sigmar_val[0]*delta_r))
    return np.array(z_inted_points)


def z_inted_fw_sigmar_udsc(datum, r_grid, sigma02):
    if len(datum)==6:
        (xbj, qsq, y, sigmar, fl, ft) = datum
    else:
        (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = datum
    r_grid=r_grid[0]
    z_inted_points = []

    # Calculate integrand at interval end points.
    for i, r in enumerate(r_grid[:-1]):
        z_inted_fw_sigmar_val = integrate.quad(lambda z: sigma02*fwd_op_sigma_reduced_udsc(qsq, y, z, r), 0, 1, epsrel=1e-4)
        z_inted_points.append([r, z_inted_fw_sigmar_val[0]])    
    # Trapezoid rule of discrete integration
    fwd_op_trapez = np.zeros((len(r_grid[:-1]), len(r_grid[:-1])))
    for i, r in enumerate(r_grid[:-2]):
        delta_r = r_grid[i+1]-r_grid[i]
        z_int_i = z_inted_points[i][1]
        z_int_ii = z_inted_points[i+1][1]
        # print(z_int_i, z_int_ii)
        # trap_op = [z_int_i/2.*delta_r, z_int_ii/2.*delta_r]
        fwd_op_trapez[i][i] = z_int_i/2.*delta_r
        fwd_op_trapez[i][i+1] = z_int_ii/2.*delta_r
    return fwd_op_trapez

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



def export_discrete(dipfile, xbj_bin, data_sigmar, parent_data_name, sigma02, include_dipole=True, use_charm=False):
    interpolated_r_grid = []
    rmin=1e-6
    rmax=30
    r_steps=100
    # rmin=8e-3
    # rmax=15
    # r_steps=15

    r=rmin
    while r<=rmax:
        interpolated_r_grid.append(r)
        r*=(rmax/rmin)**(1/r_steps)

    if dipfile:
        data_dipole = load_dipole(dipfile)
        data_dipole = np.sort(data_dipole, order=['xbj','r'])
        xbj_vals = data_dipole["xbj"]
        if xbj_vals[0] != xbj_bin:
            print("xbj bin mismatch in export!.")
            print(xbj_bin, xbj_vals[0])
            print(xbj_vals)
        r_vals = data_dipole["r"]
        S_vals = data_dipole["S"]

        # S_interp = CubicSpline(r_vals, S_vals)
        S_interp = InterpolatedUnivariateSpline(r_vals, S_vals, ext=3)
        discrete_N_vals = []
        for r in interpolated_r_grid[:-1]:
            discrete_N_vals.append(1-S_interp(r))
        vec_discrete_N = np.array(discrete_N_vals)

    print("Generating discrete forward operator. use_charm=", use_charm)
    with multiprocessing.Pool(processes=16) as pool:
    # with multiprocessing.Pool(processes=1) as pool:
        if use_charm:
            fw_op_vals_z_int = pool.starmap(z_inted_fw_sigmar_udsc, ((datum, (interpolated_r_grid,), sigma02) for datum in data_sigmar))
        else:
            fw_op_vals_z_int = pool.starmap(z_inted_fw_sigmar, ((datum, (interpolated_r_grid,), sigma02) for datum in data_sigmar))

    fw_op_datum_r_matrix = []
    for array in fw_op_vals_z_int:
        # fw_op_datum_r_matrix.append(array[:,1]) # vector value riemann sum operator, also has r in 0th col
        fw_op_datum_r_matrix.append(array) # Array only has operator elements
    fw_op_datum_r_matrix = np.array(fw_op_datum_r_matrix)

    str_id_charm = ""
    if include_dipole:
        # Simulated data and dipole
        # dscr_sigmar = np.matmul(fw_op_datum_r_matrix, vec_discrete_N) # Riemann sum just has a vector dot product
        # dscr_sigmar = np.sum(np.matvec(fw_op_datum_r_matrix, vec_discrete_N)) # Trapezoid needs a summation over the resulting vector to perform the sum of the integral
        dscr_sigmar = np.sum(np.matmul(fw_op_datum_r_matrix, vec_discrete_N), axis=1) # Trapezoid needs a summation over the resulting vector to perform the sum of the integral
        for d, s in zip(data_sigmar, dscr_sigmar):
            # print(d["sigmar"])
            print(d, d["sigmar"], s, s/d["sigmar"])
        mat_dict = {"forward_op_A": fw_op_datum_r_matrix, "discrete_dipole_N": vec_discrete_N, "r_grid": interpolated_r_grid}
        savemat("exp_fwdop_"+parent_data_name+str_id_charm+"_r_steps"+str(r_steps)+"_xbj"+str(xbj_bin)+".mat", mat_dict)
        # exit()
    else:
        # Real data without dipole
        mat_dict = {"forward_op_A": fw_op_datum_r_matrix, "r_grid": interpolated_r_grid}
        savemat("exp_fwdop_"+parent_data_name+str_id_charm+"_r_steps"+str(r_steps)+".mat", mat_dict)



def export_discrete_uniform(dipfile, mass_scheme, xbj_bin, data_sigmar, parent_data_name, sigma02=None, include_dipole=True, use_unity_sigma0=False):
    interpolated_r_grid = []
    # rmin=5e-3
    rmin=2e-3 # beta1 testing
    # rmax=25 # tightening rmin and rmax help a little with the discretization precision
    # r_steps=500 # 500 by default for simulated!
    # rmin=5e-2
    rmax=25 # ALPHA2 testing, lower limit at 5e-3 seemed much more important than the upperlimit.
    # rmax=30 # beta1 testing # doesn't seem to help at all compared to 25
    # r_steps=256 # still good for simulated! (maybe, might be a bit too bad at large Q^2)
    r_steps=256+128 # testing for a little bit better accuracy at large Q^2 # not quite good enough? 0.5% errors seen?
    # r_steps=512 # increasing this alone doesn't seem to improve the discretization error??
    # r_steps=128 # this leads to >2%, maybe up to 4-5%, errors at worst. Not good enough.

    if mass_scheme == "standard":
        quark_masses = mass_scheme_standard
    elif mass_scheme == "standard_light":
        quark_masses = mass_scheme_standard_light
    elif mass_scheme == "pole":
        quark_masses = mass_scheme_pole
    elif mass_scheme == "mqMpole":
        quark_masses = mass_scheme_mq_Mpole
    elif mass_scheme == "mqmq":
        quark_masses = mass_scheme_mq_mq
    elif mass_scheme == "mqMcharm":
        quark_masses = mass_scheme_mcharm
    elif mass_scheme == "mqMbottom":
        quark_masses = mass_scheme_mbottom
    elif mass_scheme == "mqMW":
        quark_masses = mass_scheme_mW
    elif mass_scheme == "mass_scheme_heracc_charm_only":
        quark_masses = mass_scheme_heracc_charm_only
    else:
        print("BAD MASS SCHEME")
        exit()

    r=rmin
    while len(interpolated_r_grid)<r_steps+1:
        interpolated_r_grid.append(r)
        r+=(rmax-rmin)/r_steps

    if dipfile:
        data_dipole = load_dipole(dipfile)
        data_dipole = np.sort(data_dipole, order=['xbj','r'])
        xbj_vals = data_dipole["xbj"]
        if xbj_vals[0] != xbj_bin:
            print("xbj bin mismatch in export!.")
            print(xbj_bin, xbj_vals[0])
            print(xbj_vals)
        r_vals = data_dipole["r"]
        S_vals = data_dipole["S"]

        # S_interp = CubicSpline(r_vals, S_vals)
        S_interp = InterpolatedUnivariateSpline(r_vals, S_vals, k=1, ext=3)
        discrete_N_vals = []
        for r in interpolated_r_grid[:-1]:
            discr_N = 1-S_interp(r)
            if discr_N <= 0:
                print("DISCRETE N NOT POSITIVE!:", discr_N, r)
                exit()
            discrete_N_vals.append(discr_N)
        vec_discrete_N = np.array(discrete_N_vals)

    real_sigma = sigma02
    if use_unity_sigma0:
        sigma02 = 1

    with multiprocessing.Pool(processes=16) as pool:
        fw_op_vals_z_int = pool.starmap(z_inted_fw_sigmar_udscb_riem_uniftrapez, ((datum, (interpolated_r_grid,), sigma02, quark_masses) for datum in data_sigmar))


    fw_op_datum_r_matrix = []
    for array in fw_op_vals_z_int:
        fw_op_datum_r_matrix.append(array[:,1]) # vector value riemann/uniform trapez sum operator, also has r in 0th col
    fw_op_datum_r_matrix = np.array(fw_op_datum_r_matrix)

    # Filename settings
    str_id_qmass = "_" + mass_scheme + "_"
    if use_unity_sigma0:
        str_unity_sigma02 = "_unitysigma"
    else:
        str_unity_sigma02 = "_realsigma"
    
    qsq_vals = data_sigmar["qsq"]
    sigmar_vals = data_sigmar["sigmar"]
    sigmar_errs = data_sigmar["sigmarerr"]

    if "reference" in parent_data_name:
        sigmar_theory = data_sigmar["theory"]
    else:
        sigmar_theory = []

    # Export
    exp_folder = "./export_hera_data/"
    # base_name = exp_folder+"exp_fwdop_qms_hera_"
    base_name = exp_folder+"beta1_exp_fwdop_qms_hera_"
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



def run_export(mass_scheme, use_real_data, fitname_i=None):
    """Recostruct the dipole amplitude N from simulated reduced cross section data."""

    ###################################
    ### SETTINGS ######################
    ###################################

    use_unity_sigma0 = True
    charm_only=False

    # #        0        1        2        3           4
    fits = ["MV", "MVgamma", "MVe", "bayesMV4", "bayesMV5"]
    if not fitname_i:
        fitname = fits[3]
    # else:
    #     fitname = fits[fitname_i]

    ####################
    # Reading data files
    # data_path = "./data/paper1/"
    data_path = "./data/paper2/" # s binned data
    data_path_cc = "./data/paper2/"
    dipole_path = "./data/paper2/dipoles/"
    dipole_files = [i for i in os.listdir(dipole_path) if os.path.isfile(os.path.join(dipole_path, i)) and 'dipole_fit_'+fitname+"_" in i]
    # print(dipole_files)
    # xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in dipole_files]
    # print(xbj_bins)

    # if use_charm:
    #     sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
    #         'sigmar_'+fitname+"_dipole-lightpluscharm" in i]
    # else:
    #     sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
    #         'sigmar_'+fitname+"_dipole." in i]
    # print(sigmar_files)
    
    # loading data for the corret fit setup
    # sig_file = data_path + sigmar_files[0]
    # data_sigmar = get_data(sig_file)
    # if len(sigmar_files)>1:
    #     print("MORE THAN ONE SIGMAR_FILE AT LOAD TIME??")
    #     print("Using first file.")
    #     print(sigmar_files)
    #     exit(1)
    # qsq_vals = data_sigmar["qsq"]
    # y_vals = data_sigmar["y"]
    # sigma02=read_sigma02(sig_file)
    # print("sigma02 read as: ", sigma02, isinstance(sigma02, float))
    # if use_unity_sigma0:
    #     print("Using unity sigma02 with real sigma02 = ", sigma02)

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
                ret = export_discrete_uniform(ref_dipole, mass_scheme, xbj_bin, data_sigmar, Path(sig_file).stem, sigma02=sigma02, include_dipole=inc_dip, use_unity_sigma0=use_unity_sigma0)
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
                ret = export_discrete_uniform(None, mass_scheme, xbj_bin, data_sigmar, Path(sig_file).stem, include_dipole=False, use_unity_sigma0=use_unity_sigma0)
            print("Export done with:", mass_scheme, use_real_data)
        # exit()
    # else:
    #     # Simulated data
    #     print("Simulated data with quark mass schemes not implemented! Exit!")
    #     exit()
    #     for dip_file in dipole_files:
    #         xbj_bin = float(Path(dip_file).stem.split("xbj")[1])
    #         data_sigmar_binned = data_sigmar[data_sigmar["xbj"]==xbj_bin]
    #         if data_sigmar_binned.size==0:
    #             print("NO DATA FOUND IN THIS BIN: ", xbj_bin, " in file: ", sig_file)
    #             continue
    #         print("Discretizing forward problem for dipole file: ", dip_file, " at xbj=", xbj_bin, ", Datapoints N=", data_sigmar_binned.size)
    #         # export_discrete(data_path+dip_file, xbj_bin, data_sigmar_binned, Path(sigmar_files[0]).stem, sigma02, use_charm=use_charm)
    #         ret = export_discrete_uniform(data_path+dip_file, xbj_bin, data_sigmar_binned, Path(sigmar_files[0]).stem, use_unity_sigma0=use_unity_sigma0)
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
        "standard",
        "standard_light",
        # "pole",
        "mqMpole", # this is the more accurate alternative to 'standard'
        # "mqmq",
        "mqMcharm",
        "mqMbottom",
        "mqMW",
        # "mass_scheme_heracc_charm_only"
    ]

    fitname_i = None
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
