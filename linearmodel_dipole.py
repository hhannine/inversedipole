import math
import scipy.integrate as integrate
from scipy import stats
from scipy.interpolate import CubicSpline

import torch
import numpy as np
from cuqi.model import LinearModel

from data_manage import load_dipole, get_data
from deepinelasticscattering import fwd_op_sigma_reduced

# Need to formulate and construct the linear forward operator as a matrix.
# For this, we need to discretize the integration into a sum of products.


model = LinearModel(forward,
                    adjoint=adjoint,
                    range_geometry=1,
                    domain_geometry=1)


# Main

def main():
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

    S_interp = CubicSpline(r_vals, S_vals)            
    discrete_S_vals = []
    for r in r_vals[:-1]:
        discrete_S_vals.append(S_interp(r))

    # We need to test the forward operator acting on a dipole to get a calculation of the reduced cross section
    # 'b = Ax', i.e. sigma_r = integrate(fwd_op*N,{r,z}), where the operator needs to integrate over r and z.

    fw_op_vals_z_int = []
    for datum in data_sigmar[0:2]:
        (xbj, qsq, y, sigmar, fl, ft) = datum
        
        # Testing DISCRETIZATION:
        for i, r in enumerate(r_vals[:-1]):
            z_inted_points = []
            delta_r = r_vals[i+1]-r_vals[i]
            z_inted_fw_sigmar = integrate.dblquad(lambda z: sigma02*fwd_op_sigma_reduced(qsq, y, z, r), 0, 1, epsrel=1e-4)
            z_inted_points.append(z_inted_fw_sigmar*delta_r)

        fw_op_vals_z_int.append(z_inted_points) # each element in fw_op_vals_z_int is a list of z-integrated operator points as the function of r, domain by r_vals.
    #end for datum

    print("xbj,    qsq,       y,   sigmar,    FL_LO,    FT_LO,   sigmr_test[0],   sigmr_test3[0],   sigmr_test3[0]/sigmar")

    # test calculating sigmar with the discretized operator
    for i, datum in enumerate(data_sigmar[0:2]):
        (xbj, qsq, y, sigmar, fl, ft) = datum
        fw_op = fw_op_vals_z_int[i]


        # torch.mv(a,b) matrix vector product
        # Note that for the future, you may also find torch.matmul() useful. torch.matmul() infers the dimensionality of your arguments and accordingly performs either dot products between vectors, matrix-vector or vector-matrix multiplication


    return 0

if __name__=="__main__":
    main()
