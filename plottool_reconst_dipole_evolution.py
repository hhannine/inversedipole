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
    # fitname = fits[4] + "_"

    ####################
    # Data filename settings
    # data_path = "./reconstructions/"
    data_path = "./reconstructions_IUSdip/"
    str_data = "sim_"
    str_fit = fitname
    str_flavor = "lightonly_"
    name_base = 'recon_out_'
    str_flavor_c = "lightpluscharm_"
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
    if not recon_files:
        print("No files found!")
        print(data_path, composite_fname)
        exit(None)
    # xbj_bins = sorted([float(Path(i).stem.split("xbj")[1]) for i in recon_files])
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in recon_files]
    # print(xbj_bins)
    recon_files = [x for _, x in sorted(zip(xbj_bins, recon_files))]
    recon_files_c = [x for _, x in sorted(zip(xbj_bins, recon_files_c))]
    xbj_bins = sorted(xbj_bins)
    print(xbj_bins)

    data_list = [loadmat(data_path + fle) for fle in recon_files]
    data_list_c = [loadmat(data_path + fle) for fle in recon_files_c]
    
    # Reading data
    R_GRID = data_list[0]["r_grid"][0]
    XBJ = np.array(xbj_bins)
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


    dip_data_fit = np.array([dat["N_fit"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec = np.array([dat["N_reconst"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec_c = np.array([dat["N_reconst"] for dat in data_list_c]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec_adj = np.array([dat["N_rec_adjacent"] for dat in data_list]) # matrix of all the reconstructions, need to find correct lambda
    if use_real_data:
        dip_rec_from_b_plus_err = [dat["N_reconst_from_b_plus_err"] for dat in data_list]
        dip_rec_from_b_minus_err = [dat["N_reconst_from_b_minus_err"] for dat in data_list]

    if lambda_type=="fixed_":
        N_max_data = [dat["N_maxima"][0] for dat in data_list]
    else:
        N_max_data = [dat["N_maxima"][0][2] for dat in data_list]

    N_max_data = [dat["N_maxima"][0] for dat in data_list]
    Nc_max_data = [dat["N_maxima"][0] for dat in data_list_c]

    Nlight_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_max_data)])
    Ncharm_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list_c, Nc_max_data)])
    
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
    ### PLOT TYPE DIPOLE EVOLUTION IN X
    ####################
    all_bins = True
    alt_bins = all_bins

    if not use_real_data:
        plt1_xbj_bins = [xbj_bins.index(1e-2), xbj_bins.index(1e-3),xbj_bins.index(1e-5),]
    else:
        if alt_bins:
            # plt1_xbj_bins = [xbj_bins.index(8e-2), xbj_bins.index(5e-2), xbj_bins.index(8e-3)]
            plt1_xbj_bins = range(len(xbj_bins))
        else:
            plt1_xbj_bins = [xbj_bins.index(1.3e-2), xbj_bins.index(1.3e-3), xbj_bins.index(1.3e-4)]
    # print("plt1_xbj_bins", plt1_xbj_bins)
    for xbj, dat in zip(xbj_bins, data_list):
        # print(dat.keys())
        if (str(xbj) not in dat["dip_file"][0]):
            print("SORT ERROR?", str(xbj), dat["dip_file"][0])
    if not plt1_xbj_bins:
        print("No xbj bins found!", xbj_bins)
        exit()
    binned_dip_fit = [dip_data_fit[i] for i in plt1_xbj_bins]
    binned_dip_data_rec = [dip_data_rec[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_c = [dip_data_rec_c[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_adj = [dip_data_rec_adj[i].T for i in plt1_xbj_bins]
    binned_dip_rec_from_bplus = [dip_rec_from_b_plus_err[i] for i in plt1_xbj_bins]
    binned_dip_rec_from_bminus = [dip_rec_from_b_minus_err[i] for i in plt1_xbj_bins]
    binned_mI_list = [mI_list[i] for i in plt1_xbj_bins]
    
    fig = plt.figure(layout='constrained')
    ax = plt.gca()
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    ax.tick_params(which='major',width=1,length=6)
    ax.tick_params(which='minor',width=0.7,length=4)
    ax.tick_params(axis='both', pad=7)
    ax.tick_params(axis='both', which='both', direction="in")

    # if USE_TITLE:
    #     plt.title(title)
    plt.xlabel(r'$r ~ \left(\mathrm{GeV}^{-1} \right)$', fontsize=22)
    plt.ylabel(r'$N(r)$', fontsize=22)
    xvar = data_list[0]["r_grid"]

    # LOG AXIS
    ax.set_xscale('log')
    # ax.set_yscale('log')
    
    ##############
    # LABELS

    scalings = [1, 1, 1]
    if use_real_data:
        additives = [0, 10, 20]
    else:
        additives = [0, 2, 4]
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
    line_rec_bplus = Line2D([0,1],[0,1],linestyle='-.',linewidth=lw/3, color="black")
    line_rec_bminus = Line2D([0,1],[0,1],linestyle=':',linewidth=lw/3, color="black")

    if use_real_data:
        manual_handles = [line_fit0, (line_rec, uncert_col),
                        line_rec_bplus, line_rec_bminus,
                        uncert_col0,
                        uncert_col1,
                        uncert_col2,]
    else:
        manual_handles = [line_fit0, (line_rec, uncert_col),
                        uncert_col0,
                        uncert_col1,
                        uncert_col2,]

    if use_real_data:
        manual_labels = [
            r'${\mathrm{Fit ~ dipole}}$',
            r'${\mathrm{Reconstruction ~ from ~ HERA ~ data}\, \pm \, \varepsilon_\lambda}$',
            r'${\mathrm{Reconstruction ~ to} ~ \sigma_r + \mathrm{error}}$',
            r'${\mathrm{Reconstruction ~ to} ~ \sigma_r - \mathrm{error}}$',
        ]
    else:
        manual_labels = [
            r'${\mathrm{Fit ~ dipole}}$',
            r'${\mathrm{Reconstructed ~ dipole}\, \pm \, \varepsilon_\lambda}$',
        ]
    for ibin in plt1_xbj_bins:
        xbj_str = str(xbj_bins[ibin])
        if "e" in xbj_str:
            # xbj_str = "0.00001" #"10^{{-5}}"
            xbj_str = "10^{{-5}}"
        manual_labels.append('$x_{{\\mathrm{{Bj.}} }} = {xbj}$'.format(xbj = xbj_str))


    ####################
    #################### PLOTTING
    #################### 
    colors = plt.cm.managua(np.linspace(0, 1, len(xbj_bins)))

    plot_fit = False
    plot_q = "light"
    # plot_q = "charm"

    # Plot fit dipoles and their reconstructions
    for i, (dip_rec, dip_rec_c) in enumerate(zip(binned_dip_data_rec, binned_dip_data_rec_c)):
        if xbj_bins[i] < 1e-2 and plot_fit:
            ax.plot(xvar[0], binned_dip_fit[i].T,
                    # label=labels[i+1],
                    label="Fit" + str(xbj_bins[i]),
                    linestyle="-",
                    linewidth=lw/2.5,
                    color=colors[i%len(colors)],
                    alpha=1
                    )
        else:
            continue
    for i, (dip_rec, dip_rec_c) in enumerate(zip(binned_dip_data_rec, binned_dip_data_rec_c)):
        if not plot_fit and plot_q == "charm":
            if xbj_bins[i] < 1e-3:
                ax.plot(xvar[0], dip_rec_c.T[0]/Ncharm_max[i],
                    # label=labels[i+1],
                    label="rec charm" + str(xbj_bins[i]),
                    linestyle="-",
                    linewidth=lw/2.5,
                    color=colors[i%len(colors)],
                    alpha=1
                    )
        if not plot_fit and plot_q == "light":
            if xbj_bins[i] < 1e-3:
                ax.plot(xvar[0], dip_rec.T[0]/Nlight_max[i],
                    # label=labels[i+1],
                    label="rec light" + str(xbj_bins[i]),
                    linestyle="-",
                    linewidth=lw/2.5,
                    color=colors[i%len(colors)],
                    alpha=1
                    )



    ################## SHADING        
    # Plot reconstruction uncertainties by plotting and shading between adjacent lambdas
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
    
    plt.xlim(1e-1, 10)
    plt.ylim(bottom=1e-3, top=1.1)
    # plt.tight_layout()
    fig.set_size_inches(7,7)
    fig.legend(frameon=False, fontsize=12, ncol=1, loc="outside right") 

    n_plot = "plot13-dip_evol-"
    
    if alt_bins:
        n_plot += "alt_bins-"

    if not n_plot:
        print("Plot number?")
        exit()

    write2file = False
    # write2file = True
    if write2file:
        mpl.use('agg') # if writing to PDF
        plt.draw()
        outfilename = n_plot + name_base+str_data+str_fit+lambda_type + "{}".format(PLOT_TYPE) + '.pdf'
        plotpath = G_PATH+"/inversedipole/plots/"
        print(os.path.join(plotpath, outfilename))
        plt.savefig(os.path.join(plotpath, outfilename))
    else:
        plt.show()
    return 0


main()
