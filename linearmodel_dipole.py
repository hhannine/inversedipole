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
from deepinelasticscattering import fwd_op_sigma_reduced, fwd_op_sigma_reduced_udsc



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

def z_inted_fw_sigmar_udsc_riem_uniftrapez(datum, r_grid, sigma02):
    if len(datum)==6:
        (xbj, qsq, y, sigmar, fl, ft) = datum
    else:
        (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = datum
    r_grid=r_grid[0]
    z_inted_points = []
    # Riemann sum, lower boundary evaluation
    # for i, r in enumerate(r_grid[:-1]):
    #     delta_r = r_grid[i+1]-r_grid[i]
    #     z_inted_fw_sigmar_val = integrate.quad(lambda z: sigma02*fwd_op_sigma_reduced_udsc(qsq, y, z, r), 0, 1, epsrel=1e-4)
    #     z_inted_points.append((r, z_inted_fw_sigmar_val[0]*delta_r))
    # return np.array(z_inted_points)

    # Trapezoid rule, uniform interval width
    for i, r in enumerate(r_grid[:-1]):
        delta_r = r_grid[i+1]-r_grid[i]
        z_inted_fw_sigmar_val = integrate.quad(lambda z: sigma02*fwd_op_sigma_reduced_udsc(qsq, y, z, r), 0, 1, epsrel=1e-4)
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



def export_discrete_uniform(dipfile, xbj_bin, data_sigmar, parent_data_name, sigma02, include_dipole=True, use_charm=False, use_unity_sigma0=False):
    interpolated_r_grid = []
    rmin=5e-3
    rmax=25 # tightening rmin and rmax help a little with the discretization precision
    # r_steps=500 # 500 by default for simulated!
    r_steps=256 # still good for simulated!
    # r_steps=128 # this leads to >2%, maybe up to 4-5%, errors at worst. Not good enough.

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

    # return 0 # use this breakpoint for interpolator testing.

    real_sigma = sigma02
    if use_unity_sigma0:
        # if dipfile:
        #     print("Using unity sigma, multiplying sigma02 into the dipole amplitude")
        #     vec_discrete_N = vec_discrete_N * sigma02
        sigma02 = 1

    print("Generating discrete forward operator using uniform intervals. use_charm=", use_charm)
    with multiprocessing.Pool(processes=16) as pool:
        if use_charm:
            fw_op_vals_z_int = pool.starmap(z_inted_fw_sigmar_udsc_riem_uniftrapez, ((datum, (interpolated_r_grid,), sigma02) for datum in data_sigmar))
        else:
            fw_op_vals_z_int = pool.starmap(z_inted_fw_sigmar_riem_uniftrapez, ((datum, (interpolated_r_grid,), sigma02) for datum in data_sigmar))

    fw_op_datum_r_matrix = []
    for array in fw_op_vals_z_int:
        fw_op_datum_r_matrix.append(array[:,1]) # vector value riemann/uniform trapez sum operator, also has r in 0th col
    fw_op_datum_r_matrix = np.array(fw_op_datum_r_matrix)

    # Filename settings
    if use_charm:
        str_id_charm = "_lightpluscharm"
    else:
        str_id_charm = "_lightonly"
    if use_unity_sigma0:
        str_unity_sigma02 = "_unitysigma"
    else:
        str_unity_sigma02 = "_realsigma"
    
    qsq_vals = data_sigmar["qsq"]
    sigmar_vals = data_sigmar["sigmar"]
    if not include_dipole:
        sigmar_errs = data_sigmar["sigmarerr"]
    else:
        sigmar_errs = []

    # Export
    exp_folder = "./export_fwd_IUSinterp_fix/"
    base_name = exp_folder+"exp_fwdop_v3-1+data_"
    if include_dipole:
        # Simulated data and dipole
        dscr_sigmar = np.matmul(fw_op_datum_r_matrix, vec_discrete_N)
        for d, s in zip(data_sigmar, dscr_sigmar):
            print(d, d["sigmar"], real_sigma*s, real_sigma*s/d["sigmar"])
        mat_dict = {
            "forward_op_A": fw_op_datum_r_matrix, 
            "discrete_dipole_N": vec_discrete_N, 
            "r_grid": interpolated_r_grid,
            "qsq_vals": qsq_vals,
            "sigmar_vals": sigmar_vals,
            "real_sigma": real_sigma
            }
        savemat(base_name+parent_data_name+str_id_charm+str_unity_sigma02+"_r_steps"+str(r_steps)+"_xbj"+str(xbj_bin)+".mat", mat_dict)
    else:
        # Real data without dipole // use_real_data !== include_dipole (opposite)
        mat_dict = {
            "forward_op_A": fw_op_datum_r_matrix,
            "r_grid": interpolated_r_grid,
            "qsq_vals": qsq_vals,
            "sigmar_vals": sigmar_vals,
            "sigmar_errs": sigmar_errs,
            "real_sigma": real_sigma
            }
        savemat(base_name+parent_data_name+str_id_charm+str_unity_sigma02+"_r_steps"+str(r_steps)+".mat", mat_dict)
    
    return 0



def run_export(use_charm, use_real_data, fitname_i=None):
    """Recostruct the dipole amplitude N from simulated reduced cross section data."""

    ###################################
    ### SETTINGS ######################
    ###################################

    # use_charm = False
    # use_charm = True
    # use_real_data = False
    # use_real_data = True
    use_unity_sigma0 = True

    #        0        1        2        3           4
    fits = ["MV", "MVgamma", "MVe", "bayesMV4", "bayesMV5"]
    if not fitname_i:
        fitname = fits[3]
    else:
        fitname = fits[fitname_i]
    # fitname = fits[3]

    ####################
    # Reading data files
    data_path = "./data/paper1/"
    dipole_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
        'dipole_fit_'+fitname+"_" in i]
    print(dipole_files)
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in dipole_files]
    print(xbj_bins)

    if use_charm:
        sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
            'sigmar_'+fitname+"_dipole-lightpluscharm" in i]
    else:
        sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
            'sigmar_'+fitname+"_dipole." in i]
    print(sigmar_files)
    hera_sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
        "heraII_filtered" in i]
    
    # loading data for the corret fit setup
    sig_file = data_path + sigmar_files[0]
    data_sigmar = get_data(sig_file)
    if len(sigmar_files)>1:
        print("MORE THAN ONE SIGMAR_FILE AT LOAD TIME??")
        print("Using first file.")
        print(sigmar_files)
        exit(1)
    qsq_vals = data_sigmar["qsq"]
    y_vals = data_sigmar["y"]
    sigma02=read_sigma02(sig_file)
    print("sigma02 read as: ", sigma02, isinstance(sigma02, float))
    if use_unity_sigma0:
        # sigma02=1 ## Instead multiply the sigma into the fit dipole to train lambda?
        print("Using unity sigma02 with real sigma02 = ", sigma02)

    #############################################
    # Discretizing and exporting forward problems
    if use_real_data:
        print("Discretizing with HERA II data.")
        print(hera_sigmar_files)
        xbj_bin_vals = [float(Path(i).stem.split("xbj")[1]) for i in hera_sigmar_files]
        print(xbj_bin_vals)
        # exit()
        # sigma02=35.6952 #bayesMV4 fit value
        # sigma02=1 # TODO TEST IF WE CAN RECOVER THE OVERALL SIZE CORRECTLY -> independent xbj dependence of sigma02
        # sigma02=42.0125*0.4 #Manual scaling
        for sig_file in hera_sigmar_files:
            print("Loading data file: ", sig_file)
            data_sigmar = get_data(data_path + sig_file, simulated=False)
            xbj_bin = float(Path(sig_file).stem.split("xbj")[1])
            print("Discretizing forward problem for real data file: ", sig_file, " at xbj=", xbj_bin)
            # export_discrete(None, xbj_bin, data_sigmar, Path(sig_file).stem, sigma02, include_dipole=False, use_charm=use_charm)
            ret = export_discrete_uniform(None, xbj_bin, data_sigmar, Path(sig_file).stem, sigma02, include_dipole=False, use_charm=use_charm, use_unity_sigma0=use_unity_sigma0)
        print("Export done with:", use_charm, use_real_data)
        # exit()
    else:
        # Simulated data
        for dip_file in dipole_files:
            xbj_bin = float(Path(dip_file).stem.split("xbj")[1])
            data_sigmar_binned = data_sigmar[data_sigmar["xbj"]==xbj_bin]
            if data_sigmar_binned.size==0:
                print("NO DATA FOUND IN THIS BIN: ", xbj_bin, " in file: ", sig_file)
                continue
            print("Discretizing forward problem for dipole file: ", dip_file, " at xbj=", xbj_bin, ", Datapoints N=", data_sigmar_binned.size)
            # export_discrete(data_path+dip_file, xbj_bin, data_sigmar_binned, Path(sigmar_files[0]).stem, sigma02, use_charm=use_charm)
            ret = export_discrete_uniform(data_path+dip_file, xbj_bin, data_sigmar_binned, Path(sigmar_files[0]).stem, sigma02, use_charm=use_charm, use_unity_sigma0=use_unity_sigma0)
        # exit()

    return ret


# Run multiple settings:
if __name__ == '__main__':
    multiprocessing.freeze_support()
    i=0

    run_settings=[
        # (False, False, 3),
        (True, False, 3),
        (False, False, 4),
        (True, False, 4),
        (False, True, 3),
        (True, True, 3),
        (False, True, 4),
        (True, True, 4),
    ]

    for setting in run_settings:
        use_charm, use_real_data, fitname_i = setting

        try:
            r0 = run_export(use_charm,use_real_data,fitname_i)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            i+=1
            print("error occured", i)
            raise
            exit()
    
    if i==0:
        print("Export runs finished, No errors", i==0)
    else:
        print("Error occured during exports.", i, r0)
    
    exit()
