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
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, ScalarFormatter, NullFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
numbers = re.compile(r'(\d+)')
from matplotlib import rc, cm
rc('text', usetex=True)

G_PATH = ""
STRUCT_F_TYPE = ""
PLOT_TYPE = ""
# USE_TITLE = True
USE_TITLE = False
R_GRID = []
gev_to_mb = 1/2.56819

helpstring = "usage: python plottool_reconst.py"
 

def dipole_interp(dipole):
    global R_GRID
    N_interp = InterpolatedUnivariateSpline(R_GRID, dipole, ext=3)
    return N_interp

def S_interp(N_interp, r, N_max=None):
    # if r > R_GRID[-1]:
        # return 0
    if not N_max:
        N_max = N_interp(R_GRID[-1])
    return N_max-N_interp(r)


def main(plotvar="xbj"):
    global G_PATH, PLOT_TYPE, gev_to_mb
    f_path_list = []
    G_PATH = os.path.dirname(os.path.realpath("."))

    ###################################
    ### SETTINGS ######################
    ###################################

    use_charm = False
    # use_charm = True
    # use_real_data = False
    use_real_data = True
    # use_unity_sigma0 = True # ?
    use_noise = False
    # use_noise = True

    #        0        1        2        3           4
    fits = ["MV", "MVgamma", "MVe", "bayesMV4", "bayesMV5"]
    fitname = fits[3] + "_"
    # fitname = fits[4] + "_"

    ####################
    # Data filename settings
    # data_path = "./reconstructions/"
    data_path = "./reconstructions_IUSdip/"
    str_data = "sim_"
    str_fit = fitname
    str_flavor = "lightonly_"
    str_flavor_c = "lightpluscharm_"
    name_base = 'recon_out_'
    if use_real_data:
        str_data = "hera_"
        str_fit = "data_only_"
    if use_noise:
        name_base = 'recon_with_noise_out_'
    
    # lambda_type = "broad_"
    # lambda_type = "semiconstrained_"
    # lambda_type = "semicon2_"
    lambda_type = "fixed_"
    composite_fname = name_base+str_data+str_fit+str_flavor+lambda_type
    composite_fname_c = name_base+str_data+str_fit+str_flavor_c+lambda_type
    print(composite_fname, composite_fname_c)

    # Reading data files
    recon_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname in i]
    recon_files_c = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname_c in i]
    print(recon_files)
    print(recon_files_c)
    if not recon_files:
        print("No files found!")
        print(data_path, composite_fname)
        exit(None)
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in recon_files]
    recon_files = [x for _, x in sorted(zip(xbj_bins, recon_files))]
    recon_files_c = [x for _, x in sorted(zip(xbj_bins, recon_files_c))]
    xbj_bins = sorted(xbj_bins)
    print(xbj_bins)

    data_list = [loadmat(data_path + fle) for fle in recon_files]
    data_list_c = [loadmat(data_path + fle) for fle in recon_files_c]
    
    #######################
    # Reading data
    Q2vals_grid = [dat["q2vals"][0] for dat in data_list]
    q_averages = [np.median(np.sqrt(qvals)) for qvals in Q2vals_grid] # Best?

    R = data_list[0]["r_grid"][0]
    XBJ = np.array(xbj_bins)
    rr, xx = np.meshgrid(R,XBJ)

    real_sigma = data_list[0]["real_sigma"][0]
    best_lambdas = [dat["best_lambda"][0] for dat in data_list]
    lambda_list_list = [dat["lambda"][0].tolist() for dat in data_list]
    mI_list = [lambda_list.index(best_lambda) for lambda_list, best_lambda in zip(lambda_list_list, best_lambdas)]
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

    
    # PROPER PLOTS
    # 1. reconstruction from simulated data (light only)
    #       - dipole vs reconstruction
    # 2. simu sigmar data vs data from reconst
    # 3. dipole rec with charm (simulated)
    # 4. simu sigmar rec with charm
    # 5. dipole reconstruction from HERA, light only
    # 6. HERA sigmar vs sigmar(fit) vs sigmar(reconst), light only
    # 7. dipole reconstruction from HERA, light plus charm
    # 8. HERA sigmar vs sigmar(fit) vs sigmar(reconst), light plus charm
    # 9. Sigma0(xbj) plot from real data reconstruction (or in W^2?)
    #       - lightonly rec vs lightpluscharm rec vs horizontal line from fits (how about average of rec values?)
    # 10. 2D dipole comparison in (r, xbj) or (r, W^2)
    #       - fit vs real rec lightonly vs real rec lightpluscharm (3 plots/figures/images)


    ####################
    ### PLOT TYPE DIPOLE IMAGE
    ####################

    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(20, 7)
    plt.tight_layout()
    plt.subplots_adjust(top = 0.95, bottom = 0.12, right = 1.09, left = 0.06, hspace = 0, wspace = 0)

    titles = [r"$\mathrm{Fit ~ dipole}$",
              r"$\mathrm{Reconstructed ~ dipole, ~ light ~ only}$",
              r"$\mathrm{Reconstructed ~ dipole, ~ light + charm}$",
              "",
              ]
    if plotvar == "r":
        xvar = xbj_bins
        for i, ax1 in enumerate(axs):
            # ax1.xticks(fontsize=20, rotation=0)
            # ax1.yticks(fontsize=20, rotation=0)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.grid(False)
            ax1.tick_params(which='major',width=1,length=6,labelsize=20)
            ax1.tick_params(which='minor',width=0.7,length=4,labelsize=20)
            ax1.tick_params(axis='both', pad=7)
            ax1.tick_params(axis='both', which='both', direction="in")
            ax1.set_xlim([1e-1, max(R)])
            ax1.set_ylim([min(XBJ), max(XBJ)])
            # ax1.set_title(titles[i], fontsize=20, pad=10)
            ax1.set_title(titles[i], fontsize=10, pad=1)
            ax1.set_xlabel(r'$r ~ \left( \mathrm{GeV}^{-1} \right)$', fontsize=22)
            if i==0:
                ax1.set_ylabel(r'$x_{\mathrm{Bj.}}$', fontsize=22)
            elif i!=0:
                ax1.set_yticks([])



    ####################
    #################### PLOTTING
    #################### 

    
    if plotvar=="r":
        xvar = np.array(R)
    elif plotvar=="Q":
        xvar = np.array(q_averages)

    Nlight_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_max_data)])
    Nlight_bplus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_bpluseps_max_data)])
    Nlight_bminus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_bminuseps_max_data)])
    Nlight_err_upper = Nlight_bplus_max - Nlight_max
    Nlight_err_lower = Nlight_max - Nlight_bminus_max
    Nlight_errs = np.array([gev_to_mb*Nlight_err_lower, gev_to_mb*Nlight_err_upper])

    Ncharm_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, Nc_max_data)])
    Ncharm_bplus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, Nc_bpluseps_max_data)])
    Ncharm_bminus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, Nc_bminuseps_max_data)])
    Ncharm_err_upper = Ncharm_bplus_max - Ncharm_max
    Ncharm_err_lower = Ncharm_max - Ncharm_bminus_max
    Ncharm_errs = np.array([gev_to_mb*Ncharm_err_lower, gev_to_mb*Ncharm_err_upper])

    x_srted = sorted(xvar)
    x_range = np.logspace(-2, -0.5, 3)
    q_range = np.linspace(9.6, 10.6, 1)
    q_range2 = np.linspace(10.2, 10.6, 1, endpoint=False)
    target_radii = np.sqrt(Ncharm_max/(10*math.pi)) # divide millibarn/10 to get square femtometers

    dip_data_rec = np.array([dat["N_reconst"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec_c = np.array([dat["N_reconst"] for dat in data_list_c]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_fit = np.array([dat["N_fit"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    reshape_fit = dip_data_fit.reshape((len(XBJ), len(R)))
    reshape_dip = dip_data_rec.reshape((len(XBJ), len(R)))
    reshape_dip_c = dip_data_rec_c.reshape((len(XBJ), len(R)))

    # mapname = 'plasma'
    mapname = 'magma'
    cmap = plt.colormaps[mapname]
    
    norm = plt.Normalize(np.min(Ncharm_max), np.max(Ncharm_max))
    # norm = mpl.colors.LogNorm(np.min(Ncharm_max), np.max(Ncharm_max))
    # norm = mpl.colors.LogNorm(np.min(Ncharm_max), np.max(Ncharm_max))
    smap = plt.cm.ScalarMappable(cmap='plasma', norm=norm)

    print(np.min(Ncharm_max)/10, np.max(Ncharm_max))

    target_radii = np.sqrt(Nlight_max/(10*math.pi)) # divide millibarn/10 to get square femtometers
    target_radii_c = np.sqrt(Ncharm_max/(10*math.pi)) # divide millibarn/10 to get square femtometers
    fm_to_gev = 5.068
    radii_in_gev = target_radii*fm_to_gev
    radii_in_gev_c = target_radii_c*fm_to_gev
    sigma0_to_rad_fm = np.sqrt(real_sigma*gev_to_mb/(10*math.pi))*fm_to_gev

    ax = axs[0] 
    cfit = ax.pcolormesh(rr, xx, real_sigma*reshape_fit, vmin=0, vmax=max(Ncharm_max), cmap = cmap) 
    ax.plot([sigma0_to_rad_fm]*len(xbj_bins), xbj_bins, marker="s", c="white")
    # cfit = ax.pcolormesh(rr, xx, real_sigma*reshape_fit, cmap = cmap, norm = mpl.colors.LogNorm(np.min(Ncharm_max)/10, np.max(Ncharm_max))) 
    ax = axs[1] 
    c = ax.pcolormesh(rr, xx, reshape_dip, vmin=0, vmax=max(Ncharm_max), cmap = cmap) 
    ax.plot(radii_in_gev, xbj_bins, marker="s", c="white")
    # c = ax.pcolormesh(rr, xx, reshape_dip+0.6, cmap = cmap, norm = mpl.colors.LogNorm(np.min(Ncharm_max)/10, np.max(Ncharm_max)))  
    ax = axs[2] 
    cc = ax.pcolormesh(rr, xx, reshape_dip_c, vmin=0, vmax=max(Ncharm_max), cmap = cmap) 
    ax.plot(radii_in_gev_c, xbj_bins, marker="s", c="white")
    # cc = ax.pcolormesh(rr, xx, reshape_dip_c+0.9, cmap = cmap, norm = mpl.colors.LogNorm(np.min(Ncharm_max)/10, np.max(Ncharm_max))) 
    
    cbar=fig.colorbar(cc, ax=axs.ravel().tolist(), shrink=1, pad=0.02)
    cbar.set_label(r'$\frac{\sigma_0}{2} N(r) ~ \left(\mathrm{mb}\right)$', fontsize=22)
    cbar.ax.tick_params(labelsize=15) 

    data_rec_point_c = Line2D([0,1],[0,1],linestyle='-', marker="", color="white")

    manual_handles = [
                        data_rec_point_c,
                        ]
    manual_labels = [
        # r'${\frac{r}{r_g} = 1}$',
        r'${r = r_g}$',
    ]
    leg = axs[0].legend(manual_handles, manual_labels, frameon=False, fontsize=24, ncol=1, loc="upper left") 
    for txt in leg.get_texts():
        txt.set_color("white")

    if plotvar=="r":
        n_plot = "plot10-r-dipoleimage-"
    if not n_plot:
        print("Plot number?")
        exit()

    # write2file = False
    write2file = True
    if write2file:
        mpl.use('agg') # if writing to PDF
        plt.draw()
        outfilename = n_plot + composite_fname + "{}".format(PLOT_TYPE) + '.pdf'
        # outfilename = n_plot + composite_fname + "{}".format(PLOT_TYPE) + '.png'
        plotpath = G_PATH+"/inversedipole/plots/"
        print(os.path.join(plotpath, outfilename))
        plt.savefig(os.path.join(plotpath, outfilename))
    else:
        plt.margins(0,0)
        plt.show()
    fig.clear()
    plt.close()
    return 0

def mb_to_fmrad(x):
    return np.sqrt(x/(10*math.pi))

def fmrad_to_mb(x):
    return x**2*(10*math.pi)


# plotvar r
main("r")
