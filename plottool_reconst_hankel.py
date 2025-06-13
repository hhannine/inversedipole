import sys
import os
import re
from pathlib import Path

import math
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy.io import loadmat
import hankel

import matplotlib as mpl
# mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
numbers = re.compile(r'(\d+)')
from matplotlib import rc, cm
rc('text', usetex=True)


G_PATH = ""
STRUCT_F_TYPE = ""
PLOT_TYPE = ""
# USE_TITLE = True
USE_TITLE = False
R_GRID = []

helpstring = "usage: python plottool_reconst.py"
 

def dipole_interp(dipole):
    global R_GRID
    N_interp = InterpolatedUnivariateSpline(R_GRID, dipole, k=1, ext=3)
    # print(dipole.shape)
    # N_interp = CubicSpline(R_GRID, dipole)
    return N_interp

def S_interp_scalar(N_interp, r, N_max=None):
    if r > R_GRID[-1]:
        return 0
    if r < R_GRID[0]:
        return 1
    if not N_max:
        N_max = N_interp(R_GRID[-1])
    return N_max-N_interp(r)

def prob_ball_line_pick(s, R):
    """https://mathworld.wolfram.com/DiskLinePicking.html"""
    if s>R:
        return 0
    P_2 = 4*s/(math.pi * R**2) * np.acos(s/(2*R)) - (2*s**2)/(math.pi*R**3) * np.sqrt(1 - s**2/(4 * R**2))
    return P_2

def prob_uniformly_decreasing_v1(s, R):
    """integrate 1-Divide[r,2R] from 0 to 2R"""
    if s>2*R:
        return 0
    return 1/R * (1 - s/(2*R))

def prob_uniformly_decreasing_v2(s, R):
    """integrate Divide[pi*Power[R,2],pi*Power[R+r,2]] dr from 0 to 2R"""
    if s>2*R:
    # if s>3*R:
        return 0
    return 3/(2*R) * (R**2 / (R+s)**2)
    # return 4/(3*R) * (R**2 / (R+s)**2)

def prob_uniformly_decreasing_v3(s, R):
    """integrate Divide[pi*Power[R,2],pi*Power[R+r,2]] dr from 0 to infinity"""
    return 1/(R) * (R**2 / (R+s)**2)

def main(plotvar="k"):
    global G_PATH, PLOT_TYPE, R_GRID
    f_path_list = []
    # PLOT_TYPE = sys.argv[1]
    PLOT_TYPE = "dipole"
    if PLOT_TYPE not in ["dipole", "sigmar", "noise"]:
        print(helpstring)
        PLOT_TYPE = "dipole"
        # exit(1)
    G_PATH = os.path.dirname(os.path.realpath("."))


    ###################################
    ### SETTINGS ######################
    ###################################

    # use_charm = False
    use_charm = True
    # use_real_data = False
    use_real_data = True
    # use_unity_sigma0 = True # ?
    use_noise = False
    # use_noise = True

    #        0        1        2        3           4
    fits = ["MV", "MVgamma", "MVe", "bayesMV4", "bayesMV5"]
    fitname = fits[3] + "_"

    ####################
    # Data filename settings
    # data_path = "./reconstructions/"
    data_path = "./reconstructions_IUSdip/"
    str_data = "sim_"
    str_fit = fitname
    name_base = 'recon_out_'
    str_flavor = "lightonly_"
    str_flavor_c = "lightpluscharm_"
    if use_real_data:
        str_data = "hera_"
        str_fit = "data_only_"
    if use_noise:
        name_base = 'recon_with_noise_out_'
    lambda_type = ""
    # lambda_type = "broad_"
    # lambda_type = "semiconstrained_"
    lambda_type = "fixed_"
    composite_fname = name_base+str_data+str_fit+str_flavor+lambda_type
    composite_fname_c = name_base+str_data+str_fit+str_flavor_c+lambda_type
    print(composite_fname)

    # Reading data files
    recon_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname in i]
    recon_files_c = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname_c in i]
    print(recon_files)
    if not recon_files:
        print("No files found!")
        print(data_path, composite_fname)
        exit(None)
    xbj_bins = sorted([float(Path(i).stem.split("xbj")[1]) for i in recon_files])
    print(xbj_bins)
    recon_files = [x for _, x in sorted(zip(xbj_bins, recon_files))]
    recon_files_c = [x for _, x in sorted(zip(xbj_bins, recon_files_c))]

    data_list = [loadmat(data_path + fle) for fle in recon_files]
    data_list_c = [loadmat(data_path + fle) for fle in recon_files_c]

    real_sigma = data_list[0]["real_sigma"][0]
    best_lambdas = [dat["best_lambda"][0] for dat in data_list]
    lambda_list_list = [dat["lambda"][0].tolist() for dat in data_list]
    mI_list = [lambda_list.index(best_lambda) for lambda_list, best_lambda in zip(lambda_list_list, best_lambdas)]
    best_lambdas_c = [dat["best_lambda"][0] for dat in data_list_c]
    lambda_list_list_c = [dat["lambda"][0].tolist() for dat in data_list_c]
    mI_list_c = [lambda_list_c.index(best_lambda_c) for lambda_list_c, best_lambda_c in zip(lambda_list_list_c, best_lambdas_c)]
    if lambda_type in ["semiconstrained_", "fixed_"]:
        uncert_i = [range(0, 5) for mI in mI_list]
    else:
        ucrt_step = 2
        uncert_i = [range(mI-2*ucrt_step, mI+1+2*ucrt_step, ucrt_step) for mI in mI_list]

    N_max_data = [dat["N_maxima"][0] for dat in data_list]
    N_bpluseps_max_data = [dat["N_bpluseps_maxima"][0] for dat in data_list]
    N_bminuseps_max_data = [dat["N_bminuseps_maxima"][0] for dat in data_list]
    Nc_max_data = [dat["N_maxima"][0] for dat in data_list_c]
    Nc_bpluseps_max_data = [dat["N_bpluseps_maxima"][0] for dat in data_list_c]
    Nc_bminuseps_max_data = [dat["N_bminuseps_maxima"][0] for dat in data_list_c]
    Nlight_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_max_data)])
    Ncharm_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, Nc_max_data)])


    R = data_list[0]["r_grid"][0]
    R_GRID = R
    XBJ = np.array(xbj_bins)
    dip_data_rec = np.array([dat["N_reconst"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec_c = np.array([dat["N_reconst"] for dat in data_list_c]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_fit = np.array([dat["N_fit"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    if lambda_type=="fixed_":
        N_max_data = [dat["N_maxima"][0] for dat in data_list]
    else:
        N_max_data = [dat["N_maxima"][0][2] for dat in data_list]


    S_interp = np.vectorize(S_interp_scalar)
    # prob_vectorized = np.vectorize(prob_ball_line_pick)
    # prob_vectorized = np.vectorize(prob_uniformly_decreasing_v1)
    prob_vectorized = np.vectorize(prob_uniformly_decreasing_v3)

    # 2D Fourier of the dipole
    # S_p(\mathbf{k}) = \int d^2 {\mathbf{r}} e^{i\mathbf{k} \cdot \mathbf{r}} [1 - N(\mathbf{r})]
    # Assuming angular non-dependence this is the Hankel transform of 1-N

    kays = np.linspace(0.1, 25, 100)
    hank = 30e-3
    ht = hankel.HankelTransform(
        nu= 0,     # The order of the bessel function
        N = int(3.2/hank),   # Number of steps in the integration
        h = hank   # Proxy for "size" of steps in integration
    )
    ## IF N_FIT : CANNOT USE N_max FROM THE RECONSTRUCTION!!
    # S_p_array = [ht.transform(lambda r: S_interp(dipole_interp(i),r), kays, ret_err=False) for i in dip_data_fit]

    # Hankel from Reconstruction
    if plotvar=="k":
        if not use_charm:
            S_p_array = [ht.transform(lambda r: S_interp(dipole_interp(i),r, N_max=N_max), kays, ret_err=False) for i, N_max in zip(dip_data_rec, Nlight_max)]
        elif use_charm:
            S_p_array = [ht.transform(lambda r: S_interp(dipole_interp(i),r, N_max=N_max), kays, ret_err=False) for i, N_max in zip(dip_data_rec_c, Ncharm_max)]
    elif plotvar=="k-probmod":
        if not use_charm:
            S_p_array = [ht.transform(lambda r: S_interp(dipole_interp(i),r, N_max=N_max) * prob_vectorized(r, np.sqrt(N_max/(math.pi*2))), kays, ret_err=False) for i, N_max in zip(dip_data_rec, Nlight_max)]
        elif use_charm:
            S_p_array = [ht.transform(lambda r: S_interp(dipole_interp(i),r, N_max=N_max) * prob_vectorized(r, np.sqrt(N_max/(math.pi*2))), kays, ret_err=False) for i, N_max in zip(dip_data_rec_c, Ncharm_max)]

    fig, ax = plt.subplots()
    plt.tight_layout()
    # plt.subplots_adjust(top = 0.95, bottom = 0.0, right = 0.95, left = 0., hspace = 0, wspace = 0)
    fig.set_size_inches(10.5, 10.5)

    plt.subplots_adjust(bottom=0.095, left=0.12)
    # plt.title(composite_fname + 'disc_area_probability_mod_R3.6')
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(XBJ))]
    for S_p, xbj in zip(S_p_array, xbj_bins):
        plt.plot(kays , S_p, label=xbj, color=colors[xbj_bins.index(xbj)])
    plt.legend(loc="best", frameon=False)
    ax = plt.gca()
    # ax.set_xscale('log')



    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    ax.tick_params(which='major',width=1,length=6)
    ax.tick_params(which='minor',width=0.7,length=4)
    ax.tick_params(axis='both', pad=7)
    ax.tick_params(axis='both', which='both', direction="in")

    ax.set_xlabel(r'$|k| ~ \left( \mathrm{GeV}\right)$', fontsize=22)
    ax.set_ylabel(r'$S_p(k)$', fontsize=22)

    # LOG AXIS
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-1, 25])
    # ax.set_ylim([1e-4, 50])
    ax.set_ylim([1e-4, 1000])

    if plotvar=="k":
        n_plot = "plot20-k-hankel_unintglue-"
    elif plotvar=="k-probmod":
        n_plot = "plot20-k_probmod-hankel_unintglue-"
    if not n_plot:
        print("Plot number?")
        exit()

    if use_charm:
        n_plot+="lightpluscharm_"
    else:
        n_plot+="lightonly_"

    # write2file = False
    write2file = True
    if write2file:
        mpl.use('agg') # if writing to PDF
        plt.title(n_plot + name_base+str_data+str_fit+lambda_type)
        plt.draw()
        outfilename = n_plot + name_base+str_data+str_fit+lambda_type + "{}".format(PLOT_TYPE) + '.pdf'
        # outfilename = n_plot + composite_fname + "{}".format(PLOT_TYPE) + '.png'
        plotpath = G_PATH+"/inversedipole/plots/"
        print(os.path.join(plotpath, outfilename))
        plt.savefig(os.path.join(plotpath, outfilename))
    else:
        plt.margins(0,0)
        plt.title(n_plot + name_base+str_data+str_fit+lambda_type)
        plt.show()
    plt.close()
    return 0



main("k")
main("k-probmod")
