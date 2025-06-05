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
    PLOT_TYPE = "sigmar"
    if PLOT_TYPE not in ["dipole", "sigmar", "noise"]:
        print(helpstring)
        PLOT_TYPE = "sigmar"
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
    fitname = fits[0] + "_"

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
    # xbj_bins = sorted([float(Path(i).stem.split("xbj")[1]) for i in recon_files])
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in recon_files]
    # print(xbj_bins)
    recon_files = [x for _, x in sorted(zip(xbj_bins, recon_files))]
    xbj_bins = sorted(xbj_bins)
    print(xbj_bins)

    data_list = []
    for fle in recon_files:
        data_list.append(loadmat(data_path + fle))
    
    # Reading data
    R_GRID = data_list[0]["r_grid"][0]
    Q2vals_grid = [dat["q2vals"][0] for dat in data_list]
    XBJ = np.array(xbj_bins)
    real_sigma = data_list[0]["real_sigma"][0]
    best_lambdas = [dat["best_lambda"][0] for dat in data_list]
    lambda_list_list = [dat["lambda"][0].tolist() for dat in data_list]
    mI_list = [lambda_list.index(best_lambda) for lambda_list, best_lambda in zip(lambda_list_list, best_lambdas)]
    uncert_i = [range(mI-4,mI+5,2) for mI in mI_list]
    # print(best_lambda, mI, lambda_list[mI-2:mI+3], )
    print(uncert_i)

    # for dat, qsq in zip(data_list, Q2vals_grid):
    #     print(dat["run_file"], len(qsq))
    # exit(0)

    # Reading dipoles
    dip_data_fit = np.array([dat["N_fit"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec = np.array([dat["N_reconst"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec_adj = np.array([dat["N_rec_adjacent"] for dat in data_list]) # matrix of all the reconstructions, need to find correct lambda

    # Reading sigma_r (b)
    # for dat in data_list:
    #     print(dat["run_file"], len(dat["q2vals"][0]), len(dat["b_fit"]))
    b_fit = [dat["b_fit"] for dat in data_list]
    b_rec = [dat["b_from_reconst"] for dat in data_list]
    b_rec_adj = [dat["b_from_reconst_adjacent"].T for dat in data_list] # this is a list of arrays [b_i,] instead of b_rec, which is just an array of elements of b_rec 
    print(b_rec_adj[0])
    print(b_rec_adj[0][0])

    if lambda_type=="fixed_":
        N_max_data = [dat["N_maxima"][0] for dat in data_list]
    else:
        N_max_data = [dat["N_maxima"][0][2] for dat in data_list]
    
    # PROPER PLOTS
    # 1. reconstruction from simulated data (light only)
    #       - dipole vs reconstruction
    #       - data vs data_reconst
    # 2. --||-- with charm
    # 3.
    # 4.
    # 5. Sigma0(xbj) plot from real data reconstruction

    ####################
    ### PLOT TYPE 1B --- sigma_r comparison
    ####################
    plt1_xbj_bins = [xbj_bins.index(1e-2), xbj_bins.index(1e-3),xbj_bins.index(1e-5),]
    for xbj, dat in zip(xbj_bins, data_list):
        # print(dat.keys())
        if (str(xbj) not in dat["dip_file"][0]):
            print("SORT ERROR?", str(xbj), dat["dip_file"][0])
    if not plt1_xbj_bins:
        print("No xbj bins found!", xbj_bins)
        exit()
    binned_dip_data_fit = [dip_data_fit[i] for i in plt1_xbj_bins]
    binned_dip_data_rec = [dip_data_rec[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_adj = [dip_data_rec_adj[i].T for i in plt1_xbj_bins]
    binned_b_fit = [b_fit[i] for i in plt1_xbj_bins]
    binned_b_rec = [b_rec[i] for i in plt1_xbj_bins]
    binned_b_rec_adj = [b_rec_adj[i] for i in plt1_xbj_bins]
    binned_qsq_grids = [Q2vals_grid[i] for i in plt1_xbj_bins]
        
    fig = plt.figure()
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
        plt.ylabel(r'$\frac{\sigma_0}{2} N(r) ~ \left(\mathrm{mb}\right)$', fontsize=22)
        xvar = data_list[0]["r_grid"][0]
    elif PLOT_TYPE == "sigmar":
        plt.xlabel(r'$Q^2 ~ \left(\mathrm{GeV}^{2} \right)$', fontsize=22)
        plt.ylabel(r'$\sigma_r$', fontsize=22)
        xvar = binned_qsq_grids

    # LOG AXIS
    ax.set_xscale('log')
    # ax.set_yscale('log')
    
    fit_color_set = ["orange", "red", "blue", "green", "green", "cyan"]
    fit_line_style = ['-', '--', ':']

    # make labels, line styles and colors
    labels = []
    colors = []
    line_styles = []
    scalings = []
    for fname in recon_files:
            # print(fname)
            if "bayesMV4" in fname:
                label = r'$\mathrm{bayesMV4}$'
                col = "red"
            elif "bayesMV5" in fname:
                label = r'$\mathrm{bayesMV5}$'
                col = "blue"
            elif "MV_" in fname:
                label = r'$\mathrm{MV}$'
                col = "black"
            else:
                continue
            if "xbj0.01" in fname:
                # label += r"$~ \mathrm{charm}$"
                style = "-"
                # scale = 0.8
            elif "xbj0.001" in fname:
                # label += r"$~ \mathrm{bottom}$"
                style = "--"
                # scale = 0.9
            elif "xbj1e-05" in fname:
                # label += r"$~ \mathrm{incl.}$"
                style = ":"
                # scale = 1
            else:
                continue
            labels.append(label)
            # colors.append(col)
            line_styles.append(style)
            # scalings.append(scale)

    # scalings = [1, 1.06, 1.09]
    # scalings = [0.8, 0.9, 1]
    scalings = [1, 1, 1]
    additives = [0, 2, 4]
    colors = ["orange", "orange", "black", "red", "green", "green"]
    lw=2.8

    uncert_col0 = Patch(facecolor=colors[1], alpha=0.3)
    uncert_col1 = Patch(facecolor=colors[3], alpha=0.3)
    uncert_col2 = Patch(facecolor=colors[5], alpha=0.3)
    # line_fit0 = Line2D([0,1],[0,1],linestyle='-', color=colors[0])
    line_fit0 = Line2D([0,1],[0,1],linestyle=':',linewidth=lw, color="black")
    line_rec0 = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color=colors[1])
    # line_fit1 = Line2D([0,1],[0,1],linestyle='-', color=colors[2])
    line_fit1 = Line2D([0,1],[0,1],linestyle='--',linewidth=lw, color="black")
    line_rec1 = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color=colors[3])
    # line_fit2 = Line2D([0,1],[0,1],linestyle='-', color=colors[4])
    line_fit2 = Line2D([0,1],[0,1],linestyle=':',linewidth=lw, color="black")
    line_rec2 = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color=colors[5])
    
    uncert_col = Patch(facecolor="black", alpha=0.3)
    line_rec = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color="black")

    manual_handles = [line_fit0, (line_rec, uncert_col),
                      uncert_col0,
                      uncert_col1,
                      uncert_col2,
    ]
    manual_labels = [
        r'${\mathrm{Fit ~ } \sigma_r}$',
        r'${\mathrm{Reconstructed ~ dipole}\, \pm \, \mathrm{uncertainty}}$',
        r'${x_{\mathrm{Bj.}} = 10^{-2}}$',
        r'${x_{\mathrm{Bj.}} = 10^{-3}}$',
        r'${x_{\mathrm{Bj.}} = 10^{-5}}$',
    ]
    # manual_handles = [line_fit0, (line_rec0, uncert_col0),
    #                   line_fit1, (line_rec1, uncert_col1),
    #                   line_fit2, (line_rec2,uncert_col2),]
    # manual_labels = [
    #     # r'${\mathrm{Fit ~ dipole,} ~ x_{\mathrm{Bj.}} = 10^{-2}} ~ (\times 1)$',
    #     # r'${\mathrm{Fit ~ dipole,} ~ x_{\mathrm{Bj.}} = 10^{-2}} ~ (+ 0 \mathrm{mb})$',
    #     r'${\mathrm{Fit ~ } \sigma_r, ~ x_{\mathrm{Bj.}} = 10^{-2}}$',
    #     # r'${\mathrm{Reconstructed ~ dipole}}$',
    #     r'${\mathrm{Reconstructed ~ dipole}\, \pm \, \mathrm{uncertainty}}$',
    #     # r'$\mathrm{Reconstruction ~ uncertainty}$',
    #     # r'${\mathrm{Fit ~ dipole,} ~ x_{\mathrm{Bj.}} = 10^{-3}}~ (\times 1.06)$',
    #     # r'${\mathrm{Fit ~ dipole,} ~ x_{\mathrm{Bj.}} = 10^{-3}}~ (+ 2 \mathrm{mb})$',
    #     r'${\mathrm{Fit ~ } \sigma_r, ~ x_{\mathrm{Bj.}} = 10^{-3}}$',
    #     # r'${\mathrm{Reconstructed ~ dipole,} ~ x_{\mathrm{Bj.}} = 10^{-3}}$',
    #     # r'${\mathrm{Reconstructed ~ dipole}}$',
    #     r'${\mathrm{Reconstructed ~ dipole}\, \pm \, \mathrm{uncertainty}}$',
    #     # r'$\mathrm{Reconstruction ~ uncertainty}$',
    #     # r'${\mathrm{Fit ~ dipole,} ~ x_{\mathrm{Bj.}} = 10^{-5}}~ (\times 1.09)$',
    #     # r'${\mathrm{Fit ~ dipole,} ~ x_{\mathrm{Bj.}} = 10^{-5}}~ (+ 4 \mathrm{mb})$',
    #     r'${\mathrm{Fit ~ } \sigma_r, ~ x_{\mathrm{Bj.}} = 10^{-5}}$',
    #     # r'${\mathrm{Reconstructed ~ dipole,} ~ x_{\mathrm{Bj.}} = 10^{-5}}$',
    #     r'${\sigma_r ~ \mathrm{from ~ reconstructed ~ dipole}\, \pm \, \mathrm{uncertainty}}$',
    #     # r'$\mathrm{Reconstruction ~ uncertainty}$',
    # ]

    # Plot fit dipoles and their reconstructions
    for i, (b_fit, b_rec) in enumerate(zip(binned_b_fit, binned_b_rec)):
        x_srted, b_fit = zip(*sorted(zip(xvar[i], b_fit)))
        x_srted, b_rec = zip(*sorted(zip(xvar[i], b_rec)))
        # ax.plot(xvar[i], scalings[i%3]*b_fit+additives[i%3],
        ax.plot(x_srted, b_fit,
                # label=labels[i],
                label="Fit sigma",
                # linestyle=fit_line_style[i%3],
                linestyle=":",
                # marker="x",
                linewidth=lw*1.2,
                # color=colors[2*i]
                color="black"
                )
        # ax.plot(xvar[i], scalings[i%3]*b_rec+additives[i%3],
        ax.plot(x_srted, b_rec,
                # label=labels[i+1],
                label="Reconstuction sigma",
                # linestyle=line_styles[i],
                # linestyle="",
                linestyle="-",
                # marker="x",
                linewidth=lw/2,
                color=colors[2*i+1],
                alpha=1
                # color="blue"
                )
        
    # Plot reconstruction uncertainties by plotting and shading between adjacent lambdas
    i=0
    xvar = binned_qsq_grids
    for i_rnge, adj_sigr in zip(list(uncert_i), binned_b_rec_adj):
        needed_adj_sigr = [adj_sigr[i] for i in i_rnge]
        x_srted, adj_sig0 = zip(*sorted(zip(xvar[i], needed_adj_sigr[0])))
        x_srted, adj_sig1 = zip(*sorted(zip(xvar[i], needed_adj_sigr[1])))
        x_srted, adj_sig3 = zip(*sorted(zip(xvar[i], needed_adj_sigr[3])))
        x_srted, adj_sig4 = zip(*sorted(zip(xvar[i], needed_adj_sigr[4])))
        # ax.fill_between(x_srted, adj_sig0, adj_sig4, color=colors[2*i+1], alpha=0.59)
        ax.fill_between(x_srted, adj_sig0, adj_sig4, color=colors[2*i+1], alpha=0.2)
        ax.fill_between(x_srted, adj_sig1, adj_sig3, color=colors[2*i+1], alpha=0.4)
        # ax.fill_between(xvar[i], scalings[i%3]*needed_adj_sigr[0]+additives[i%3], scalings[i%3]*needed_adj_sigr[4]+additives[i%3], color=colors[2*i+1], alpha=0.09)
        # ax.fill_between(xvar[i], scalings[i%3]*needed_adj_sigr[1]+additives[i%3], scalings[i%3]*needed_adj_sigr[3]+additives[i%3], color=colors[2*i+1], alpha=0.16)
        i+=1


    # plt.text(0.95, 0.146, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black')
    # plt.text(0.95, 0.14, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black') # scaled log log
    # plt.text(1.16, 0.225, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black') # scaled linear log
    
    plt.legend(manual_handles, manual_labels, frameon=False, fontsize=14, ncol=1, loc="upper left") 
    

    # ax.xaxis.set_major_formatter(ScalarFormatter())
    # plt.xlim(1e-3, 20)
    plt.xlim(0.1, 200)
    # plt.ylim(bottom=0, top=6)
    fig.set_size_inches(7,7)
    
    # write2file = False
    write2file = True
    plt.tight_layout()
    if write2file:
        mpl.use('agg') # if writing to PDF
        plt.draw()
        outfilename = 'plot1-'+ composite_fname + "{}".format(PLOT_TYPE) + '.pdf'
        plotpath = G_PATH+"/inversedipole/plots/"
        print(os.path.join(plotpath, outfilename))
        plt.savefig(os.path.join(plotpath, outfilename))
    else:
        plt.show()
    return 0

main()
