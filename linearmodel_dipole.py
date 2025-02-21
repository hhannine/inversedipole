# import math
import multiprocessing
import scipy.integrate as integrate
from scipy.interpolate import CubicSpline
from timeit import default_timer as timer

# import torch
import numpy as np
import matplotlib.pyplot as plt

# from cuqi.model import LinearModel
from trips.solvers.Tikhonov import Tikhonov
from trips.solvers.GKS import *
from trips.solvers.tSVD import *
from trips.solvers.tGSVD import *

from data_manage import load_dipole, get_data
from deepinelasticscattering import fwd_op_sigma_reduced

# Need to formulate and construct the linear forward operator as a matrix.
# For this, we need to discretize the integration into a sum of products.


# cuqi model stuff:
# model = LinearModel(forward,
#                     adjoint=adjoint,
#                     range_geometry=1,
#                     domain_geometry=1)


def z_inted_fw_sigmar(datum, r_grid, sigma02):
    (xbj, qsq, y, sigmar, fl, ft) = datum
    r_grid=r_grid[0]
    z_inted_points = []
    for i, r in enumerate(r_grid[:-1]):
        delta_r = r_grid[i+1]-r_grid[i]
        z_inted_fw_sigmar_val = integrate.quad(lambda z: sigma02*fwd_op_sigma_reduced(qsq, y, z, r), 0, 1, epsrel=1e-4)
        z_inted_points.append((r, z_inted_fw_sigmar_val[0]*delta_r))
    return np.array(z_inted_points)

# def main():
if __name__=="__main__":
    """Recostruct the dipole amplitude N from simulated reduced cross section data."""

    # Load data.
    # data_sigmar = get_data("./data/simulated-lo-sigmar_DIPOLE_TAKEN.txt")
    data_sigmar = get_data("./data/simulated_lo_sigmar_with_FL_FT.dat")
    qsq_vals = data_sigmar["qsq"]
    y_vals = data_sigmar["y"]
    sigma02=48.4781

    # We need a dipole initial guess?
    # data_dipole = load_dipole("./data/readable-lo-dipolescatteringamplitude_S.txt")
    data_dipole = load_dipole("./data/readable-lo_dip_S-logstep_r.dat")
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
    # x_naive = np.linalg.solve(A_fwd_op, b_sigmar) # A not square?

    print("Solving inverse problem with tSVD_sol regularization.")
    (x_tsvd, truncation_value) = tSVD_sol(A_fwd_op.todense(), b_sigmar, regparam = 'dp', delta = 1e-3)
    print("Truncation parameter is %s." % truncation_value)
    
    print("Solving inverse problem with Tikhonov regularization.")
    L = np.eye(int(A_fwd_op.shape[1]))
    x_Tikh_dp, lambda_Tikh_dp = Tikhonov(A_fwd_op, b_sigmar, L, x_true, regparam = 'dp', delta=1e-3)
    print("dp lambda: ", lambda_Tikh_dp)
    # x_Tikh, lambda_Tikh = Tikhonov(A_fwd_op, b_sigmar, L, x_true, regparam=1e-2)
    # x_Tikh2, lambda_Tikh2 = Tikhonov(A_fwd_op, b_sigmar, L, x_true, regparam=5e-9)

    # showing the naive solution, along with the exact one
    plt.plot(x_true, "r-", label='x_true')
    plt.plot(x_tsvd, label='tSVD')
    plt.plot(x_Tikh_dp, label='Tikhonov, discr princip')
    # plt.plot(x_Tikh, label='Tikh lambda 1e-2')
    # plt.plot(x_Tikh2, label='Tikh lambda 5e-9')
    # plt.plot(x_TikhL, label='general form Tikhonov') # this needs first derivative of something
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=2, mode="expand", borderaxespad=0., fontsize=20)
    plt.show()



    # torch.mv(a,b) matrix vector product
    # Note that for the future, you may also find torch.matmul() useful. torch.matmul() infers the dimensionality of your arguments and accordingly performs either dot products between vectors, matrix-vector or vector-matrix multiplication

