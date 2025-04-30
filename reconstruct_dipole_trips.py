# import math
import os
from pathlib import Path
import multiprocessing
import scipy.integrate as integrate
from scipy.interpolate import CubicSpline
from scipy.io import savemat
from timeit import default_timer as timer

# import torch
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# from cuqi.model import LinearModel
# import cuqi
from trips.solvers.Tikhonov import Tikhonov
from trips.solvers.GK_Tikhonov import Golub_Kahan_Tikhonov
from trips.solvers.A_Tikhonov import Arnoldi_Tikhonov
from trips.solvers.GKS import *
from trips.solvers.tSVD import *
from trips.solvers.tGSVD import *

from data_manage import load_dipole, get_data, read_sigma02
from deepinelasticscattering import fwd_op_sigma_reduced, fwd_op_sigma_reduced_udsc

# Need to formulate and construct the linear forward operator as a matrix.
# For this, we need to discretize the integration into a sum of products.



def z_inted_fw_sigmar(datum, r_grid, sigma02):
    if len(datum)==6:
        (xbj, qsq, y, sigmar, fl, ft) = datum
    else:
        (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = datum
    r_grid=r_grid[0]
    z_inted_points = []
    
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

def z_inted_fw_sigmar_udsc(datum, r_grid, sigma02):
    if len(datum)==6:
        (xbj, qsq, y, sigmar, fl, ft) = datum
    else:
        (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = datum
    r_grid=r_grid[0]
    z_inted_points = []
    for i, r in enumerate(r_grid[:-1]):
        delta_r = r_grid[i+1]-r_grid[i]
        z_inted_fw_sigmar_val = integrate.quad(lambda z: sigma02*fwd_op_sigma_reduced_udsc(qsq, y, z, r), 0, 1, epsrel=1e-4)
        z_inted_points.append((r, z_inted_fw_sigmar_val[0]*delta_r))

def gen_discrete(dipfile, xbj_bin, data_sigmar, parent_data_name, sigma02, include_dipole=True):
    interpolated_r_grid = []
    # rmin=1e-6
    # rmax=30
    # r_steps=100

    # This is good! 2025-04-16
    rmin=1e-3
    rmax=20
    r_steps=100

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

        S_interp = CubicSpline(r_vals, S_vals)
        discrete_N_vals = []
        for r in interpolated_r_grid[:-1]:
            discrete_N_vals.append(1-S_interp(r))
        vec_discrete_N = np.array(discrete_N_vals)

    print("Generating discrete forward operator.")
    with multiprocessing.Pool(processes=16) as pool:
    # with multiprocessing.Pool(processes=1) as pool:
        fw_op_vals_z_int = pool.starmap(z_inted_fw_sigmar, ((datum, (interpolated_r_grid,), sigma02) for datum in data_sigmar))

    fw_op_datum_r_matrix = []
    for array in fw_op_vals_z_int:
        # fw_op_datum_r_matrix.append(array[:,1]) # vector value riemann sum operator, also has r in 0th col
        fw_op_datum_r_matrix.append(array) # Array only has operator elements
    fw_op_datum_r_matrix = np.array(fw_op_datum_r_matrix)

    if include_dipole:
        # Simulated data and dipole
        dscr_sigmar = np.sum(np.matmul(fw_op_datum_r_matrix, vec_discrete_N), axis=1) # Trapezoid needs a summation over the resulting vector to perform the sum of the integral
        for d, s in zip(data_sigmar, dscr_sigmar):
            # print(d["sigmar"])
            print(d, d["sigmar"], s, s/d["sigmar"])
        # mat_dict = {"forward_op_A": fw_op_datum_r_matrix, "discrete_dipole_N": vec_discrete_N}
        return (fw_op_datum_r_matrix, vec_discrete_N)
    else:
        # Real data without dipole
        # mat_dict = {"forward_op_A": fw_op_datum_r_matrix}
        return (fw_op_datum_r_matrix, None)
    
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
    
def gen_discrete_uniform(dipfile, xbj_bin, data_sigmar, parent_data_name, sigma02, include_dipole=True, n_points=None):
    interpolated_r_grid = []
    # this works well for log interval
        # not that good with uniform intval..
        # rmin=1e-3
        # rmax=20
        # r_steps=100
    rmin=0.001
    rmax=20
    r_steps=200

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

        S_interp = CubicSpline(r_vals, S_vals)
        discrete_N_vals = []
        for r in interpolated_r_grid[:-1]:
            discrete_N_vals.append(1-S_interp(r))
        vec_discrete_N = np.array(discrete_N_vals)

    print("Generating discrete forward operator.")
    with multiprocessing.Pool(processes=16) as pool:
    # with multiprocessing.Pool(processes=1) as pool:
        fw_op_vals_z_int = pool.starmap(z_inted_fw_sigmar_riem_uniftrapez, ((datum, (interpolated_r_grid,), sigma02) for datum in data_sigmar))

    fw_op_datum_r_matrix = []
    for array in fw_op_vals_z_int:
        fw_op_datum_r_matrix.append(array[:,1]) # vector value riemann sum operator, also has r in 0th col
    fw_op_datum_r_matrix = np.array(fw_op_datum_r_matrix)

    if include_dipole:
        # Simulated data and dipole
        dscr_sigmar = np.matmul(fw_op_datum_r_matrix, vec_discrete_N)
        for d, s in zip(data_sigmar, dscr_sigmar):
            print(d, d["sigmar"], s, s/d["sigmar"])
        # mat_dict = {"forward_op_A": fw_op_datum_r_matrix, "discrete_dipole_N": vec_discrete_N}
        return (fw_op_datum_r_matrix, vec_discrete_N)
    else:
        # Real data without dipole
        # mat_dict = {"forward_op_A": fw_op_datum_r_matrix}
        return (fw_op_datum_r_matrix, None)

# def main():
if __name__=="__main__":
    """Recostruct the dipole amplitude N from simulated reduced cross section data."""

    # Load data.
    # data_sigmar = get_data("./data/simulated-lo-sigmar_WITH_DIPOLE_PRINT_higher_resolution_in_R_Q.dat")
    # data_sigmar = get_data("./data/hera_II_binned_s_318.1_xbj_0.013.dat", simulated=False)
    # data_sigmar = get_data("./data/hera_II_binned_s_318.1_xbj_0.002.dat", simulated=False)

    # Automating file IO
    fits = ["MV", "MVgamma", "MVe"]
    fitname = fits[0]

    data_path = "./data/paper1/"
    dipole_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
        'dipole_fit_'+fitname+"_" in i]
    print(dipole_files)
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in dipole_files]
    print(xbj_bins)

    sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
        'sigmar_'+fitname+"_" in i]
    print(sigmar_files)
    hera_sigmar_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
        "heraII_filtered" in i]
    
    sig_file = data_path + sigmar_files[0]
    data_sigmar = get_data(sig_file)
    if len(sigmar_files)>1:
        print("MORE THAN ONE SIGMAR_FILE AT LOAD TIME??")
        print("Using first file.")
        print(sigmar_files)
    qsq_vals = data_sigmar["qsq"]
    y_vals = data_sigmar["y"]
    sigma02=read_sigma02(sig_file)
    print("sigma02 read as: ", sigma02, isinstance(sigma02, float))

    # Discretizing and exporting forward problems
    use_real_data = False
    # use_real_data = True
    use_uniform_trapez = True
    # use_uniform_trapez = False
    if use_real_data:
        if use_uniform_trapez:
            print("real data + uniform trapez not implemented! Exit")
            exit()
        print("Discretizing with HERA II data.")
        print(hera_sigmar_files)
        sigma02=42.0125 #MVe fit value
        # sigma02=42.0125*0.4 #Manual scaling
        for sig_file in hera_sigmar_files:
            print("Loading data file: ", sig_file)
            data_sigmar = get_data(data_path + sig_file, simulated=False)
            xbj_bin = float(Path(sig_file).stem.split("xbj")[1])
            print("Discretizing forward problem for real data file: ", sig_file, " at xbj=", xbj_bin)
            gen_discrete(None, xbj_bin, data_sigmar, Path(sig_file).stem, sigma02, include_dipole=False)
        print("Export done. Exit.")
    else:
        # Simulated data
        # for dip_file in dipole_files[1:]:
        for dip_file in dipole_files:
            xbj_bin = float(Path(dip_file).stem.split("xbj")[1])
            data_sigmar_binned = data_sigmar[data_sigmar["xbj"]==xbj_bin]
            if data_sigmar_binned.size==0:
                print("NO DATA FOUND IN THIS BIN: ", xbj_bin, " in file: ", sig_file)
                continue
            if use_uniform_trapez:
                print("UNIFORM TRAPEZ Discretizing forward problem for dipole file: ", dip_file, " at xbj=", xbj_bin, ", Datapoints N=", data_sigmar_binned.size)
                fw_op_datum_r_matrix, vec_discrete_N = gen_discrete_uniform(data_path+dip_file, xbj_bin, data_sigmar_binned, Path(sigmar_files[0]).stem, sigma02, n_points=data_sigmar_binned.size)
            else:
                print("Discretizing forward problem for dipole file: ", dip_file, " at xbj=", xbj_bin, ", Datapoints N=", data_sigmar_binned.size)
                fw_op_datum_r_matrix, vec_discrete_N = gen_discrete(data_path+dip_file, xbj_bin, data_sigmar_binned, Path(sigmar_files[0]).stem, sigma02)
            # Trips py reconstruction
            x_true = vec_discrete_N
            print(len(vec_discrete_N))
            b_sigmar = data_sigmar_binned["sigmar"]
            print(len(b_sigmar))
            A_fwd_op = fw_op_datum_r_matrix
            print(A_fwd_op.shape)
            # exit()
            continue

            print("Solving inverse problem with Tikhonov regularization.")
            L = np.eye(int(A_fwd_op.shape[1]))
            x_Tikh, lambda_Tikh = Tikhonov(A_fwd_op, b_sigmar, L, x_true, regparam=1e-4)
            x_Tikh_GK, lambda_Tikh_gk = Golub_Kahan_Tikhonov(A_fwd_op, b_sigmar, n_iter = 3, regparam = 'gcv')
            x_Tikh_A, lambda_Tikh_a = Arnoldi_Tikhonov(A_fwd_op, b_sigmar, n_iter = 3, regparam = 'gcv')
            # x_Tikh_dp, lambda_Tikh_dp = Tikhonov(A_fwd_op, b_sigmar, L, x_true, regparam = 'dp', delta=1e-3)
            # print("dp lambda: ", lambda_Tikh_dp)
            # x_Tikh, lambda_Tikh = Tikhonov(A_fwd_op, b_sigmar, L, x_true, regparam=1e-2)
            # x_Tikh2, lambda_Tikh2 = Tikhonov(A_fwd_op, b_sigmar, L, x_true, regparam=5e-9)

            # showing the naive solution, along with the exact one
            plt.plot(x_true, "r-", label='x_true')
            # plt.plot(x_tsvd, label='tSVD')
            # plt.plot(x_Tikh_dp, label='Tikhonov, discr princip')
            plt.plot(x_Tikh, label='Tikhonov')
            plt.plot(x_Tikh_GK, label='Tikhonov GK')
            plt.plot(x_Tikh_A, label='Tikhonov A')
            # plt.plot(x_Tikh2, label='Tikh lambda 5e-9')
            # plt.plot(x_TikhL, label='general form Tikhonov') # this needs first derivative of something
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                    ncol=2, mode="expand", borderaxespad=0., fontsize=20)
            plt.show()

    ###########
    # Main cont.
    #
    




    # torch.mv(a,b) matrix vector product
    # Note that for the future, you may also find torch.matmul() useful. torch.matmul() infers the dimensionality of your arguments and accordingly performs either dot products between vectors, matrix-vector or vector-matrix multiplication

