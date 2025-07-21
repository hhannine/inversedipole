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

    #        0        1        2        3           4
    fits = ["MV", "MVgamma", "MVe", "bayesMV4", "bayesMV5"]
    if not fitname_i:
        fitname = fits[3] + "_"
    else:
        fitname = fits[fitname_i] + "_"
    # fitname = fits[4] + "_"

    ####################
    # Data filename settings
    # data_path = "./reconstructions/"
    data_path = "./reconstructions_gausserr/"
    str_data = "sim_"
    str_fit = fitname
    str_flavor = "lightonly_"
    name_base = 'recon_gausserr_'
    if use_charm:
        str_flavor = "lightpluscharm_"
    if use_real_data:
        str_data = "hera_"
        str_fit = "data_only_"
    if use_noise:
        name_base = 'recon_with_noise_out_'
    
    lambda_type = "broad_"
    # lambda_type = "semiconstrained_"
    # lambda_type = "semicon2_"
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
    XBJ = np.array(xbj_bins)
    real_sigma = data_list[0]["real_sigma"][0]


    dip_data_fit = np.array([dat["N_fit"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec = np.array([dat["N_reconst"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec_std_up = np.array([dat["N_rec_one_std_up"] for dat in data_list])
    dip_data_rec_std_dn = np.array([dat["N_rec_one_std_dn"] for dat in data_list])
    
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
    ### PLOT TYPE DIPOLE
    ####################
    # alt_bins = True
    alt_bins = False

    if not use_real_data:
        plt1_xbj_bins = [xbj_bins.index(1e-2), xbj_bins.index(1e-3),xbj_bins.index(1e-5),]
    else:
        if alt_bins:
            plt1_xbj_bins = [xbj_bins.index(8e-2), xbj_bins.index(5e-2), xbj_bins.index(8e-3)]
        else:
            plt1_xbj_bins = [xbj_bins.index(1.3e-2), xbj_bins.index(1.3e-3), xbj_bins.index(1.3e-4)]
    # print("plt1_xbj_bins", plt1_xbj_bins)
    for i in plt1_xbj_bins:
        xbj = xbj_bins[i]
        print(xbj, data_list[i]["run_file"], data_list[i]["dip_file"])
        print(data_list[i]["N_fit"][0][:5])
        print(data_list[i]["N_reconst"].T[0][:5])
    for xbj, dat in zip(xbj_bins, data_list):
        # print(dat.keys())
        if (str(xbj) not in dat["dip_file"][0]):
            print("SORT ERROR?", str(xbj), dat["dip_file"][0])
    if not plt1_xbj_bins:
        print("No xbj bins found!", xbj_bins)
        exit()
    binned_dip_data_fit = [dip_data_fit[i] for i in plt1_xbj_bins]
    binned_dip_data_rec = [dip_data_rec[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_std_up = [dip_data_rec_std_up[i].T for i in plt1_xbj_bins]
    binned_dip_data_rec_std_dn = [dip_data_rec_std_dn[i].T for i in plt1_xbj_bins]

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
        # xvar = data_list[0]["r_grid"][0][:-1]
        xvar = data_list[0]["r_grid"][0]
    elif PLOT_TYPE == "sigmar":
        plt.xlabel(r'$Q^2 ~ \left(\mathrm{GeV}^{2} \right)$', fontsize=22)
        plt.ylabel(r'$\sigma_r ~ \left(\mathrm{GeV}^{-2} \right)$', fontsize=22)
        xvar = data_list[0]["q2vals"]

    # LOG AXIS
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ##############
    # LABELS
    labels = []
    colors = []
    line_styles = []
    scalings = []
    for fname in recon_files:
            # print(fname)
            if "bayesMV4" in fname:
                label = r'$\mathrm{bayesMV4}$'
            elif "bayesMV5" in fname:
                label = r'$\mathrm{bayesMV5}$'
            elif "MV_" in fname:
                label = r'$\mathrm{MV}$'
            else:
                continue
            labels.append(label)

    scalings = [1, 1, 1]
    if use_real_data:
        additives = [0, 10, 20]
    else:
        # additives = [0, 2, 4]
        additives = [0, 0, 0]
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

    # Plot fit dipoles and their reconstructions
    for i, (dip_fit, dip_rec) in enumerate(zip(binned_dip_data_fit, binned_dip_data_rec)):
        ax.plot(xvar, gev_to_mb*scalings[i%3]*real_sigma*dip_fit[0]+additives[i%3],
                # label=labels[i],
                label="Fit dipole",
                # linestyle=":",
                linestyle="-",
                marker="+",
                linewidth=lw*1,
                # color=colors[2*i]
                color="black"
                )
        ax.plot(xvar, gev_to_mb*scalings[i%3]*dip_rec.T[0]+additives[i%3],
                # label=labels[i+1],
                label="Reconstuction of fit dipole",
                linestyle="-",
                linewidth=lw/2.5,
                color=colors[2*i+1],
                alpha=1
                )
        # if use_real_data:
        #     dip_from_bplus = binned_dip_rec_from_bplus[i].T[0]
        #     dip_from_bminus = binned_dip_rec_from_bminus[i].T[0]
        #     ax.plot(xvar[0], gev_to_mb*scalings[i%3]*dip_from_bplus+additives[i%3],
        #             # label=labels[i+1],
        #             label="Reconstuction of fit dipole",
        #             linestyle="-.",
        #             linewidth=lw/3.5,
        #             color=colors[2*i+1],
        #             alpha=1
        #             )
        #     ax.plot(xvar[0], gev_to_mb*scalings[i%3]*dip_from_bminus+additives[i%3],
        #             # label=labels[i+1],
        #             label="Reconstuction of fit dipole",
        #             linestyle=":",
        #             linewidth=lw/3.5,
        #             color=colors[2*i+1],
        #             alpha=1
        #             )

    ################## SHADING        
    # Plot reconstruction uncertainties by plotting and shading between adjacent lambdas
    i=0
    shade_alph_closer = 0.2
    shade_alph_further = 0.1
    for i, (rec_up, rec_dn) in enumerate(zip(binned_dip_data_rec_std_up, binned_dip_data_rec_std_dn)):
        rec_dip = binned_dip_data_rec[i][:,0]
        rec_up = rec_up[0]
        rec_dn = rec_dn[0]
        print(xvar.shape, rec_dip.shape, rec_up.shape, rec_dn.shape)
        ax.fill_between(xvar, gev_to_mb*scalings[i%3]*rec_dip+additives[i%3], gev_to_mb*scalings[i%3]*rec_up+additives[i%3], color=colors[2*i+1], alpha=shade_alph_closer)
        ax.fill_between(xvar, gev_to_mb*scalings[i%3]*rec_dip+additives[i%3], gev_to_mb*scalings[i%3]*rec_dn+additives[i%3], color=colors[2*i+1], alpha=shade_alph_closer)
        i+=1


    # plt.text(0.95, 0.146, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black')
    # plt.text(0.95, 0.14, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black') # scaled log log
    # plt.text(1.16, 0.225, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black') # scaled linear log
    
    plt.legend(manual_handles, manual_labels, frameon=False, fontsize=12, ncol=1, loc="upper left") 
    
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.xlim(1e-3, 25)
    # plt.xlim(0.05, 25)
    # plt.ylim(bottom=0, top=40)
    # plt.ylim(bottom=1e-5)
    plt.ylim(bottom=1e-16)
    fig.set_size_inches(7,7)
    
    if not use_real_data:
        if not use_charm:
            n_plot = "plotg1-"
        elif use_charm:
            n_plot = "plotg3-"
    elif use_real_data:
        if not use_charm:
            n_plot = "plotg5-"
        elif use_charm:
            n_plot = "plotg7-"
    
    if alt_bins:
        n_plot += "alt_bins-"

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
        plotpath = G_PATH+"/inversedipole/plots_gaussian/"
        print(os.path.join(plotpath, outfilename))
        plt.savefig(os.path.join(plotpath, outfilename))
    else:
        plt.show()
    return 0

# main(use_charm=False,real_data=False,fitname_i=3)
# main(use_charm=False,real_data=False,fitname_i=4)
main(use_charm=True,real_data=False,fitname_i=4)

# Production plotting
# main(use_charm=False,real_data=False,fitname_i=3)
# main(use_charm=True,real_data=False,fitname_i=3)
# main(use_charm=False,real_data=False,fitname_i=4)
# main(use_charm=True,real_data=False,fitname_i=4)
# main(use_charm=False,real_data=True,fitname_i=3)
# main(use_charm=True,real_data=True,fitname_i=3)
