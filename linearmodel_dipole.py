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
# from trips.solvers.Tikhonov import Tikhonov
# from trips.solvers.GKS import *
# from trips.solvers.tSVD import *
# from trips.solvers.tGSVD import *

from data_manage import load_dipole, get_data, read_sigma02
from deepinelasticscattering import fwd_op_sigma_reduced, fwd_op_sigma_reduced_udsc

# Need to formulate and construct the linear forward operator as a matrix.
# For this, we need to discretize the integration into a sum of products.


# cuqi model stuff:
# model = LinearModel(forward,
#                     adjoint=adjoint,
#                     range_geometry=1,
#                     domain_geometry=1)


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

        S_interp = CubicSpline(r_vals, S_vals)
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

    # if use_charm:
    #     str_id_charm = "light_plus_charm"
    # else:
    #     str_id_charm = "light_only"
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

def export_discrete_uniform(dipfile, xbj_bin, data_sigmar, parent_data_name, sigma02, include_dipole=True, use_charm=False):
    interpolated_r_grid = []
    # rmin=0.001
    # rmax=20
    rmin=1e-4
    rmax=30
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

    str_id_charm = ""
    qsq_vals = data_sigmar["qsq"]
    sigmar_vals = data_sigmar["sigmar"]
    if include_dipole:
        # Simulated data and dipole
        dscr_sigmar = np.matmul(fw_op_datum_r_matrix, vec_discrete_N)
        for d, s in zip(data_sigmar, dscr_sigmar):
            print(d, d["sigmar"], s, s/d["sigmar"])
        mat_dict = {
            "forward_op_A": fw_op_datum_r_matrix, 
            "discrete_dipole_N": vec_discrete_N, 
            "r_grid": interpolated_r_grid,
            "qsq_vals": qsq_vals,
            "sigmar_vals": sigmar_vals
            }
        savemat("exp_fwdop_"+parent_data_name+str_id_charm+"_r_steps"+str(r_steps)+"_xbj"+str(xbj_bin)+".mat", mat_dict)
        # exit()
    else:
        # Real data without dipole
        if use_charm:
            str_id_charm = "_lightpluscharm"
        else:
            str_id_charm = "_lightonly"
        mat_dict = {
            "forward_op_A": fw_op_datum_r_matrix,
            "r_grid": interpolated_r_grid,
            "qsq_vals": qsq_vals,
            "sigmar_vals": sigmar_vals
            }
        savemat("exp_fwdop_"+parent_data_name+str_id_charm+"_r_steps"+str(r_steps)+".mat", mat_dict)



# def main():
if __name__=="__main__":
    """Recostruct the dipole amplitude N from simulated reduced cross section data."""

    ###################################
    ### SETTINGS ######################
    ###################################

    # use_charm = False
    use_charm = True
    # use_real_data = False
    use_real_data = True

    fits = ["MV", "MVgamma", "MVe"]
    fitname = fits[1]

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
            'sigmar_'+fitname+"_" in i]
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
    qsq_vals = data_sigmar["qsq"]
    y_vals = data_sigmar["y"]
    sigma02=read_sigma02(sig_file)
    print("sigma02 read as: ", sigma02, isinstance(sigma02, float))

    #############################################
    # Discretizing and exporting forward problems
    if use_real_data:
        print("Discretizing with HERA II data.")
        print(hera_sigmar_files)
        sigma02=42.0125 #MVe fit value
        # sigma02=42.0125*0.4 #Manual scaling
        for sig_file in hera_sigmar_files:
            print("Loading data file: ", sig_file)
            data_sigmar = get_data(data_path + sig_file, simulated=False)
            xbj_bin = float(Path(sig_file).stem.split("xbj")[1])
            print("Discretizing forward problem for real data file: ", sig_file, " at xbj=", xbj_bin)
            # export_discrete(None, xbj_bin, data_sigmar, Path(sig_file).stem, sigma02, include_dipole=False, use_charm=use_charm)
            export_discrete_uniform(None, xbj_bin, data_sigmar, Path(sig_file).stem, sigma02, include_dipole=False, use_charm=use_charm)
        print("Export done. Exit.")
        exit()
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
            export_discrete_uniform(data_path+dip_file, xbj_bin, data_sigmar_binned, Path(sigmar_files[0]).stem, sigma02, use_charm=use_charm)
        exit()


    ########################
    # Main cont.
    #

    # Loading dipole
    data_dipole = load_dipole("./data/readable-lo-dipole_S-log_step_r_denser.dat")
    data_dipole = np.sort(data_dipole, order=['xbj','r'])
    xbj_vals = data_dipole["xbj"]
    r_vals = data_dipole["r"]
    S_vals = data_dipole["S"]

    interpolated_r_grid = []
    rmin=1e-6
    rmax=30
    r=rmin
    while r<=rmax:
        interpolated_r_grid.append(r)
        r*=(rmax/rmin)**(1/1000.)

    S_interp = CubicSpline(r_vals, S_vals)
    discrete_N_vals = []
    for r in interpolated_r_grid[:-1]:
        discrete_N_vals.append(1-S_interp(r))
    vec_discrete_N = np.array(discrete_N_vals)

    # We need to test the forward operator acting on a dipole to get a calculation of the reduced cross section
    # 'b = Ax', i.e. sigma_r = integrate(fwd_op*N,{r,z}), where the operator needs to integrate over r and z.

    print("Generating discrete forward operator.")
    with multiprocessing.Pool(processes=16) as pool:
        fw_op_vals_z_int = pool.starmap(z_inted_fw_sigmar, ((datum, (interpolated_r_grid,), sigma02) for datum in data_sigmar))

    # Single threaded implem. of DISCRETIZATION:
    if False:
        for datum in data_sigmar:
            (xbj, qsq, y, sigmar, fl, ft) = datum
            z_inted_points = []
            for i, r in enumerate(interpolated_r_grid[:-1]):
                delta_r = interpolated_r_grid[i+1]-interpolated_r_grid[i]
                z_inted_fw_sigmar = z_inted_fw_sigmar(qsq, y, r, sigma02)
                z_inted_points.append((r, z_inted_fw_sigmar[0]*delta_r))
            fw_op_vals_z_int.append(np.array(z_inted_points)) # each element in fw_op_vals_z_int is a list of z-integrated operator points as the function of r, domain by r_vals.

    fw_op_datum_r_matrix = []
    for array in fw_op_vals_z_int:
        fw_op_datum_r_matrix.append(array[:,1])
    fw_op_datum_r_matrix = np.array(fw_op_datum_r_matrix)

    start = timer()
    dscr_sigmar = np.matmul(fw_op_datum_r_matrix, vec_discrete_N)
    end = timer()
    for d, s in zip(data_sigmar, dscr_sigmar):
        print( d, s)


    print("Matmul took (s): ",end - start) # Time in seconds


    x_true = vec_discrete_N
    b_sigmar = data_sigmar["sigmar"]
    A_fwd_op = fw_op_datum_r_matrix

    print("Solving inverse problem with CUQIpy.")
    # Bayesian model
    A=A_fwd_op
    y_data=b_sigmar
    model=cuqi.model.LinearModel(A)
    # x = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), 0.5)
    d = cuqi.distribution.Gamma(10, 1)
    s = cuqi.distribution.Gamma(10, 1)
    x = cuqi.distribution.LMRF(10, lambda d: 1/d, geometry=model.domain_geometry)
    # x = cuqi.distribution.InverseGamma(shape=np.ones(model.domain_dim) , location=np.zeros(model.domain_dim), scale=np.ones(model.domain_dim), geometry=model.domain_geometry)
    # x = cuqi.distribution.Uniform(low=0, high=1, geometry=model.domain_geometry)
    y = cuqi.distribution.Gaussian(model@x, lambda s: 1/s)
    # y = cuqi.distribution.Gaussian(model@x, cov=0.01)
    # Define Bayesian problem and set data
    BP = cuqi.problem.BayesianProblem(y, x, d, s).set_data(y=y_data)
    # BP = cuqi.problem.BayesianProblem(y, x).set_data(y=y_data)
    samples = BP.sample_posterior(400)

    # Compute MAP estimate
    # x_MAP = BP.MAP()
    # x_ML = BP.ML()
    # Compute samples from posterior
    # samples = BP.sample_posterior(400)
    # Plot results
    # x_samples.plot_ci(exact=x_true)
    # plt.show()

    x_mean = samples["x"].mean()
    # x_mean = samples.mean()
    x_std = samples["x"].std()
    # x_std = samples.std()
    fig, ax = plt.subplots()
    ax.plot(interpolated_r_grid[:-1], x_mean, '-')
    ax.fill_between(interpolated_r_grid[:-1], x_mean - x_std, x_mean + x_std, alpha=0.2)
    ax.plot(interpolated_r_grid[:-1], x_true, '--', color='tab:brown')
    plt.show()

    # Step 4: Analyze results
    # x_true.plot(); plt.title("True dipole (exact solution)")
    # y_data.plot(); plt.title("Blurred and noisy image (data)")
    # samples["x"].plot_mean(); plt.title("Estimated image (posterior mean)")
    # plt.show()
    # samples["x"].plot_std(); plt.title("Uncertainty (posterior standard deviation)")
    # samples["s"].plot_trace(); plt.suptitle("Noise level (posterior trace)")
    # samples["d"].plot_trace(); plt.suptitle("Regularization parameter (posterior trace)")
    # plt.show()

    # Plot difference between MAP and sample mean
    # (x_MAP - x_samples.mean()).plot()
    # plt.title("MAP estimate - sample mean")
    # plt.show()
    exit()

    # print("Solving inverse problem with tSVD_sol regularization.")
    # (x_tsvd, truncation_value) = tSVD_sol(A_fwd_op.todense(), b_sigmar, regparam = 'dp', delta = 1e-3)
    # print("Truncation parameter is %s." % truncation_value)
    
    print("Solving inverse problem with Tikhonov regularization.")
    L = np.eye(int(A_fwd_op.shape[1]))
    x_Tikh_dp, lambda_Tikh_dp = Tikhonov(A_fwd_op, b_sigmar, L, x_true, regparam = 'dp', delta=1e-3)
    print("dp lambda: ", lambda_Tikh_dp)
    # x_Tikh, lambda_Tikh = Tikhonov(A_fwd_op, b_sigmar, L, x_true, regparam=1e-2)
    # x_Tikh2, lambda_Tikh2 = Tikhonov(A_fwd_op, b_sigmar, L, x_true, regparam=5e-9)

    # showing the naive solution, along with the exact one
    plt.plot(x_true, "r-", label='x_true')
    # plt.plot(x_tsvd, label='tSVD')
    plt.plot(x_Tikh_dp, label='Tikhonov, discr princip')
    # plt.plot(x_Tikh, label='Tikh lambda 1e-2')
    # plt.plot(x_Tikh2, label='Tikh lambda 5e-9')
    # plt.plot(x_TikhL, label='general form Tikhonov') # this needs first derivative of something
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=2, mode="expand", borderaxespad=0., fontsize=20)
    plt.show()



    # torch.mv(a,b) matrix vector product
    # Note that for the future, you may also find torch.matmul() useful. torch.matmul() infers the dimensionality of your arguments and accordingly performs either dot products between vectors, matrix-vector or vector-matrix multiplication

