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
from scipy.optimize import root_scalar

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

def calc_saturation_scale(dipole_N, Nmax):
    # N(x, r^2 = 2/Q_s^2) = 1 - e^{-½}
    # i.e. we look for the zero point of the function: N(x, r^2 = 2/Q_s^2) - 1 + e^{-½} = 0
    dipole_N = dipole_interp(dipole_N)
    r_s = np.sqrt(root_scalar(lambda r: dipole_N(r)/Nmax-1+0.606530659712, bracket=[0.01, 5],).root)
    Q_s = math.sqrt(2)/r_s
    # Q_s = 1/(math.sqrt(math.sqrt(2))*r_s)
    return Q_s

def main(use_charm=False, real_data=False, fitname_i=None):
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
    # use_charm = True
    use_charm = use_charm
    # use_real_data = False
    # use_real_data = True
    use_real_data = real_data
    # use_unity_sigma0 = True # ?
    use_noise = False
    # use_noise = True

    print(use_charm, use_real_data, fitname_i)

    #        0        1        2        3           4
    fits = ["MV", "MVgamma", "MVe", "bayesMV4", "bayesMV5"]
    if not fitname_i:
        fitname = fits[3] + "_"
    else:
        fitname = fits[fitname_i] + "_"
    # fitname = fits[4] + "_"

    ####################
    # Data filename settings
    data_path = "./reconstructions_hera_uq/"
    str_data = "sim_"
    s_str = "s318.1_"
    str_fit = "dis_inclusive_" + s_str
    str_flavor = "standard_"
    # str_flavor_c = "lightpluscharm_"
    r_steps = "384"
    name_base = 'hera_recon_uq_'+r_steps+"_"
    if use_real_data:
        str_data = "rec_hera_data_"
    if use_noise:
        name_base = 'recon_with_noise_out_'
    q_cut_str = "no_Q_cut"

    lambda_type = "lambdaSRN_"
    composite_fname = name_base + str_data + str_fit + str_flavor + lambda_type + q_cut_str
    print(composite_fname)

    # Reading data files
    recon_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname in i]
    if not recon_files:
        print("No files found!")
        print(data_path, composite_fname)
        exit(None)
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in recon_files]
    recon_files = [x for _, x in sorted(zip(xbj_bins, recon_files))]
    print(recon_files)

    xbj_bins = sorted(xbj_bins)
    print(xbj_bins)

    data_list = [loadmat(data_path + fle) for fle in recon_files]
   
    # composite_fname_c = name_base+str_data+str_fit+str_flavor_c+lambda_type
    # recon_files_c = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname_c in i]
    # recon_files_c = [x for _, x in sorted(zip(xbj_bins, recon_files_c))]
    # data_list_c = [loadmat(data_path + fle) for fle in recon_files_c]
    
    # Reading data
    R_GRID = data_list[0]["r_grid"][0]
    XBJ = np.array(xbj_bins)

# data from
# dip_props_strict % Nmax, rmax, rs, Qs as [mean, dn, up, dn2, up2] -> Qs mean is 16th element
# and ref_dip_props

    # dip_data_fit = np.array([dat["N_fit"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    # dip_data_rec = np.array([dat["N_reconst"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid

    Qs_fit = []
    Qs_strict = []
    for xbj, dat in zip(xbj_bins, data_list):
        # print(dat["dip_props_strict"][0][16])
        Qs_strict.append(dat["dip_props_strict"][0][16])
        if xbj < 0.01:
            Qs_fit.append(dat["ref_dip_props"][0][3])
        else:
            a = 0
            Qs_fit.append(a)
    Qs_fit = np.array(Qs_fit)
    Qs_strict = np.array(Qs_strict)
    # print(Qs_fit[0])
    # print(Qs_strict[0])
    # exit()

    N_max_data = np.array([dat["N_max_data_strict"][0] for dat in data_list]) # this is used for the colorization scaling
    N_max_data = N_max_data[:,0]
    sig_max = max(N_max_data)


    ####################
    ### PLOT TYPE Q_sat
    ####################

    # Qs_fit = [calc_saturation_scale(dip, 1) for dip in dip_data_fit]
    # Qs_rec = [calc_saturation_scale(dip, mx) for dip, mx in zip(dip_data_rec, Nlight_max)]
    # Qs_rec_c = [calc_saturation_scale(dip, mx) for dip, mx in zip(dip_data_rec_c, Ncharm_max)]

    fig = plt.figure()
    ax = plt.gca()
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    ax.tick_params(which='major',width=1,length=6)
    ax.tick_params(which='minor',width=0.7,length=4)
    ax.tick_params(axis='both', pad=7)
    ax.tick_params(axis='both', which='both', direction="in")
    plt.xlabel(r'$x_{\mathrm{Bj.}}$', fontsize=22)
    plt.ylabel(r'$Q_s ~ \left(\mathrm{GeV}\right)$', fontsize=22)
    xvar = xbj_bins

    # LOG AXIS
    ax.set_xscale('log')
    
    ##############
    # LABELS
    colors = ["blue", "green", "brown", "orange", "magenta", "red"]
    lw=2.8
    ms=4
    mstyle = "o"
    color_alph = 1

    uncert_col0 = Patch(facecolor=colors[1], alpha=color_alph)
    uncert_col1 = Patch(facecolor=colors[3], alpha=color_alph)
    uncert_col2 = Patch(facecolor=colors[5], alpha=color_alph)
    line_fit0 = Line2D([0,1],[0,1],linestyle=':',linewidth=lw, color="black")
    
    uncert_col = Patch(facecolor="black", alpha=0.3)
    line_rec = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color="black")

    manual_handles = [line_fit0, (line_rec, uncert_col),
                    uncert_col0,
                    uncert_col1,
                    uncert_col2,]
    manual_labels = [
        r'${\mathrm{Fit ~ dipole}}$',
        r'${\mathrm{Reconstruction ~ from ~ HERA ~ data}\, \pm \, \varepsilon_\lambda}$',
    ]


    ####################
    #################### PLOTTING
    #################### 

    # Plot fit dipoles and their reconstructions
    # ax.plot(xbj_bins, [0.28]*len(xbj_bins),
    ax.plot([0.01], [0.28]*len([0.01]),
            # label=labels[i],
            label="Fit at x0",
            linestyle="",
            linewidth=lw*1,
            marker="o",
            # color=colors[2*i]
            color="black"
            )
    ax.plot(xbj_bins, Qs_fit,
            # label=labels[i],
            label="Fit",
            linestyle=":",
            linewidth=lw*1,
            # color=colors[2*i]
            color="black"
            )
    # ax.plot(xbj_bins, Qs_rec*Nlight_max/max(Nlight_max),
    ax.plot(xbj_bins, Qs_strict*N_max_data/max(N_max_data),
    # ax.plot(xbj_bins, Qs_strict,
            # label=labels[i+1],
            # label="Qs light * sigma0(x)/max(sigma0)",
            label="Qs strict",
            linestyle="-",
            linewidth=lw/2.5,
            color=colors[0],
            alpha=1
            )
    # ax.plot(xbj_bins, np.ones(len(Qs_rec_c))/Qs_rec_c,
    # ax.plot(xbj_bins, Qs_rec_c*Ncharm_max/max(Ncharm_max),
    #         # label=labels[i+1],
    #         label="Qs charm * sigma0(x)/max(sigma0)",
    #         linestyle="-",
    #         linewidth=lw/2.5,
    #         color=colors[1],
    #         alpha=1
    #         )


    # ################## SHADING        
    # # Plot reconstruction uncertainties by plotting and shading between adjacent lambdas
    # i=0
    # shade_alph_closer = 0.2
    # shade_alph_further = 0.1
    # for i, (i_rnge, adj_dips) in enumerate(zip(uncert_i, binned_dip_data_rec_adj)):
    #     mI = binned_mI_list[i]
    #     print(mI)
    #     rec_dip = adj_dips[mI]
    #     needed_adj_dips = [adj_dips[i] for i in i_rnge]
    #     if mI==0:
    #         ax.fill_between(xvar[0], gev_to_mb*scalings[i%3]*rec_dip+additives[i%3], gev_to_mb*scalings[i%3]*needed_adj_dips[1]+additives[i%3], color=colors[2*i+1], alpha=shade_alph_closer)
    #         ax.fill_between(xvar[0], gev_to_mb*scalings[i%3]*rec_dip+additives[i%3], gev_to_mb*scalings[i%3]*needed_adj_dips[2]+additives[i%3], color=colors[2*i+1], alpha=shade_alph_further)
    #     elif mI==4:
    #         ax.fill_between(xvar[0], gev_to_mb*scalings[i%3]*rec_dip+additives[i%3], gev_to_mb*scalings[i%3]*needed_adj_dips[3]+additives[i%3], color=colors[2*i-1], alpha=shade_alph_closer)
    #         ax.fill_between(xvar[0], gev_to_mb*scalings[i%3]*rec_dip+additives[i%3], gev_to_mb*scalings[i%3]*needed_adj_dips[2]+additives[i%3], color=colors[2*i+1], alpha=shade_alph_further)
    #     else:
    #         # at least one step on both sides
    #         ax.fill_between(xvar[0], gev_to_mb*scalings[i%3]*rec_dip+additives[i%3], gev_to_mb*scalings[i%3]*needed_adj_dips[3]+additives[i%3], color=colors[2*i+1], alpha=shade_alph_closer)
    #         ax.fill_between(xvar[0], gev_to_mb*scalings[i%3]*rec_dip+additives[i%3], gev_to_mb*scalings[i%3]*needed_adj_dips[1]+additives[i%3], color=colors[2*i+1], alpha=shade_alph_closer)
    #         if mI==2:
    #             # 2 steps on either side
    #             ax.fill_between(xvar[0], gev_to_mb*scalings[i%3]*rec_dip+additives[i%3], gev_to_mb*scalings[i%3]*needed_adj_dips[4]+additives[i%3], color=colors[2*i+1], alpha=shade_alph_further)
    #             ax.fill_between(xvar[0], gev_to_mb*scalings[i%3]*rec_dip+additives[i%3], gev_to_mb*scalings[i%3]*needed_adj_dips[0]+additives[i%3], color=colors[2*i+1], alpha=shade_alph_further)
    #     i+=1

    
    # plt.legend(manual_handles, manual_labels, frameon=False, fontsize=12, ncol=1, loc="upper left") 
    plt.legend(frameon=False, fontsize=12, ncol=1, loc="upper left") 
    
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    # plt.xlim(1e-3, 20)
    # plt.xlim(0.05, 25)
    # plt.ylim(bottom=0, top=40)
    fig.set_size_inches(7,7)
    
    n_plot = "plot12-satscale-"
    
    if not n_plot:
        print("Plot number?")
        exit()

    write2file = False
    # write2file = True
    plt.tight_layout()
    if write2file:
        mpl.use('agg') # if writing to PDF
        plt.draw()
        outfilename = n_plot + composite_fname + "{}".format(PLOT_TYPE) + '.pdf'
        plotpath = G_PATH+"/inversedipole/plots/"
        print(os.path.join(plotpath, outfilename))
        plt.savefig(os.path.join(plotpath, outfilename))
    else:
        plt.show()
    return 0


# Production plotting
main(use_charm=False,real_data=True,fitname_i=3)
# main(use_charm=True,real_data=False,fitname_i=3)
# main(use_charm=False,real_data=False,fitname_i=4)
# main(use_charm=True,real_data=False,fitname_i=4)
# main(use_charm=False,real_data=True,fitname_i=3)
# main(use_charm=True,real_data=True,fitname_i=3)
