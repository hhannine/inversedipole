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
    N_interp = InterpolatedUnivariateSpline(R_GRID, dipole, ext=3)
    # print(dipole.shape)
    # N_interp = CubicSpline(R_GRID, dipole)
    return N_interp

def S_interp_scalar(N_interp, r, N_max=None):
    # if r > R_GRID[-1]:
    #     return 0
    # if r < R_GRID[0]:
    #     return 1
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

def main():
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

    use_charm = False
    # use_charm = True
    use_real_data = False
    # use_real_data = True
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
    str_flavor = "lightonly_"
    name_base = 'recon_out_'
    if use_charm:
        str_flavor = "lightpluscharm_"
    if use_real_data:
        str_data = "hera_"
        str_fit = "data_only_"
    if use_noise:
        name_base = 'recon_with_noise_out_'
    lambda_type = ""
    lambda_type = "broad_"
    # lambda_type = "semiconstrained_"
    # lambda_type = "fixed_"
    composite_fname = name_base+str_data+str_fit+str_flavor+lambda_type
    print(composite_fname)

    # Reading data files
    recon_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
                   composite_fname in i]
    print(recon_files)
    if not recon_files:
        print("No files found!")
        print(data_path, composite_fname)
        exit(None)
    xbj_bins = sorted([float(Path(i).stem.split("xbj")[1]) for i in recon_files])
    print(xbj_bins)
    recon_files = [x for _, x in sorted(zip(xbj_bins, recon_files))]

    data_list = []
    for fle in recon_files:
        data_list.append(loadmat(data_path + fle))
    
    if True:
        # plt.style.use('_mpl-gallery')
        # plt.style.use('_mpl-gallery-nogrid')
        # plot_2d()
        R = data_list[0]["r_grid"][0]
        R_GRID = R
        # print(R)
        XBJ = np.array(xbj_bins)
        rr, xx = np.meshgrid(R,XBJ)
        dip_data = np.array([dat["N_reconst"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
        # dip_data = np.array([dat["N_fit"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
        if lambda_type=="fixed_":
            N_max_data = [dat["N_maxima"][0] for dat in data_list]
        else:
            N_max_data = [dat["N_maxima"][0][2] for dat in data_list]

        print("SIZES", R.shape, XBJ.shape, dip_data[0].shape)
        reshape_dip = dip_data.reshape((len(XBJ), len(R)))
        print(reshape_dip.shape)

        S_interp = np.vectorize(S_interp_scalar)
        # prob_vectorized = np.vectorize(prob_ball_line_pick)
        # prob_vectorized = np.vectorize(prob_uniformly_decreasing_v1)
        prob_vectorized = np.vectorize(prob_uniformly_decreasing_v2)

        # 2D Fourier of the dipole
        # S_p(\mathbf{k}) = \int d^2 {\mathbf{r}} e^{i\mathbf{k} \cdot \mathbf{r}} [1 - N(\mathbf{r})]
        # Assuming angular non-dependence this is the Hankel transform of 1-N

        kays = np.linspace(0.1, 10, 200)
        ht = hankel.HankelTransform(
            nu= 0,     # The order of the bessel function
            N = 750,   # Number of steps in the integration
            h = 0.00005   # Proxy for "size" of steps in integration
        )
        ## IF N_FIT : CANNOT USE N_max FROM THE RECONSTRUCTION!!
        # S_p_array = [ht.transform(lambda r: S_interp(dipole_interp(i),r), kays, ret_err=False) for i in dip_data]

        # Hankel from Reconstruction
        # S_p_array = [ht.transform(lambda r: S_interp(dipole_interp(i),r, N_max=N_max), kays, ret_err=False) for i, N_max in zip(dip_data, N_max_data)]
        S_p_array = [ht.transform(lambda r: S_interp(dipole_interp(i),r, N_max=N_max) * prob_vectorized(r, np.sqrt(N_max/(math.pi*2))), kays, ret_err=False) for i, N_max in zip(dip_data, N_max_data)]
        fig, ax = plt.subplots()
        fig.set_size_inches(10.5, 10.5)
        plt.subplots_adjust(bottom=0.025, left=0.035)
        plt.title(composite_fname)
        # plt.title(composite_fname + 'disc_area_probability_mod_R3.6')
        cmap = plt.get_cmap('RdBu_r')
        colors = [cmap(i) for i in np.linspace(0, 1, len(XBJ))]
        for S_p, xbj in zip(S_p_array, xbj_bins):
            plt.plot(kays , S_p, label=xbj, color=colors[xbj_bins.index(xbj)])
        plt.legend(loc="best", frameon=False)
        ax = plt.gca()
        # ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_ylim([1e-4, 10])
        plt.show()
        exit()
        

        # plot
        fig, ax = plt.subplots()
        fig.set_size_inches(10.5, 10.5)
        plt.subplots_adjust(bottom=0.025, left=0.035)
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.pcolormesh(rr, xx, reshape_dip, shading='auto') 
        c = ax.pcolormesh(rr, xx, reshape_dip, vmin=0, vmax=30.0, cmap = plt.colormaps['magma']) 
        plt.colorbar(c)
        # ax.plot_surface(xx, rr, reshape_dip, cmap=cm.Blues) 
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_xlim([min(R), max(R)])
        ax.set_xlim([1e-1, max(R)])
        ax.set_ylim([min(XBJ), max(XBJ)])
        plt.show()
        exit()

    ####
    # old plotting (needed for below?)
    plt.figure()
    ax = plt.gca()
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    ax.tick_params(which='major',width=1,length=6)
    ax.tick_params(which='minor',width=0.7,length=4)
    ax.tick_params(axis='both', pad=7)
    ax.tick_params(axis='both', which='both', direction="in")

    # if USE_TITLE:
    #     plt.title(title)
    if PLOT_TYPE == "dipole":
        plt.xlabel(r'$r ~ \left(\mathrm{GeV}^{-1} \right)$', fontsize=22)
        plt.ylabel(r'$N(r)$', fontsize=22)
        xvar = data_list[0]["r_grid"]
    elif PLOT_TYPE == "sigmar":
        plt.xlabel(r'$Q^2 ~ \left(\mathrm{GeV}^{2} \right)$', fontsize=22)
        plt.ylabel(r'$\sigma_r ~ \left(\mathrm{GeV}^{-2} \right)$', fontsize=22)
        xvar = data_list[0]["q2vals"]

    # LOG AXIS
    ax.set_xscale('log')
    # ax.set_yscale('log')
    
    fit_color_set = ["orange", "red", "blue", "green", "magenta", "cyan"]
    fit_line_style = ['-', '--', '-.', ':']

    # make labels, line styles and colors
    labels = []
    colors = []
    line_styles = []
    scalings = []
    for fname in f_path_list:
            fname = os.path.splitext(os.path.basename(fname))[0]
            if "urp" in fname:
                label = r'$\mathrm{Fit ~ 1}$'
                col = "red"
            elif "ukp" in fname:
                label = r'$\mathrm{Fit ~ 2}$'
                col = "orange"
            elif "utp" in fname:
                label = r'$\mathrm{TBK ~ p.d.}$'
                col = "black"
            elif "urs" in fname:
                label = r'$\mathrm{ResumBK ~ bs.d.}$'
                col = "magenta"
            elif "uks" in fname:
                label = r'$\mathrm{KCBK ~ bs.d.}$'
                col = "brown"
            elif "uts" in fname:
                label = r'$\mathrm{Fit ~ 3}$'
                col = "blue"
            if "cc" in fname:
                # label += r"$~ \mathrm{charm}$"
                style = "--"
                scale = 2
            elif "bb" in fname:
                # label += r"$~ \mathrm{bottom}$"
                style = ":"
                scale = 30
            elif "lpcb" in fname:
                # label += r"$~ \mathrm{incl.}$"
                style = "-"
                scale = 1
            labels.append(label)
            colors.append(col)
            line_styles.append(style)
            scalings.append(scale)

    # colors = ["red", "orange", "blue"]

    # print(labels)
    # line1 = Line2D([0,1],[0,1],linestyle='-', color='r')
    # line2 = Line2D([0,1],[0,1],linestyle='-', color='black')
    # line3 = Line2D([0,1],[0,1],linestyle='-', color='blue')
    line1 = Patch(facecolor=colors[0])
    line2 = Patch(facecolor=colors[1])
    line3 = Patch(facecolor=colors[2])
    line_incl = Line2D([0,1],[0,1],linestyle='-', color='grey')
    line_char = Line2D([0,1],[0,1],linestyle='--', color='grey')
    line_bott = Line2D([0,1],[0,1],linestyle=':', color='grey')
    manual_handles = [line1, line2, line3, line_incl, line_char, line_bott]
    manual_labels = [
        # r'$\mathrm{Fit ~ 1}$',
        # r'$\mathrm{Fit ~ 2}$',
        # r'$\mathrm{Fit ~ 3}$',
        labels[0],
        labels[1],
        labels[2],
        r'$\mathrm{inclusive}$',
        r'$\mathrm{charm}\times 2$',
        r'$\mathrm{bottom}\times 30$'
    ]

    for i, f_nlo in enumerate(f_list_nlo):
        plt.plot(xvar , scalings[i]*f_nlo,
                label=labels[i],
                linestyle=line_styles[i],
                linewidth=1.2,
                color=colors[i % 3]
                )

    # plt.text(0.95, 0.146, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black')
    # plt.text(0.95, 0.14, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black') # scaled log log
    plt.text(1.16, 0.225, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black') # scaled linear log

    # if len(f_path_list) < 6:
    #     plt.legend(loc="best", frameon=False)
    # else:
    #     order=[1,0,2,4,3,5,7,6,8]
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], frameon=False, fontsize=14) 
    plt.legend(manual_handles, manual_labels, frameon=False, fontsize=14, ncol=2) 
    # plt.legend(handlelength=1, handleheight=1)

    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.xlim(1, 100)
    plt.ylim(bottom=0, top=0.33)
    plt.draw()
    plt.tight_layout()
    outfilename = 'plot-EIC-' + "{}".format(STRUCT_F_TYPE) + '.pdf'
    plt.savefig(os.path.join(G_PATH, outfilename))
    return 0




main()
