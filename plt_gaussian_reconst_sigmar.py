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
    PLOT_TYPE = "sigmar"
    if PLOT_TYPE not in ["dipole", "sigmar", "noise"]:
        print(helpstring)
        PLOT_TYPE = "sigmar"
        # exit(1)
    G_PATH = os.path.dirname(os.path.realpath("."))

    ###################################
    ### SETTINGS ######################
    ###################################

    # use_charm = False
    # use_charm = True
    # use_real_data = False
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
    # name_base = 'recon_gausserr_v4-2'
    name_base = 'recon_gausserr_v4-4r256_'
    if use_charm:
        str_flavor = "lightpluscharm_"
    if use_real_data:
        str_data = "hera_"
        str_fit = "data_only_"
    if use_noise:
        name_base = 'recon_with_noise_out_'
    
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

    # Reading dipoles
    dip_data_fit = np.array([dat["N_fit"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec = np.array([dat["N_reconst"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid

    # Reading sigma_r (b)
    b_cpp_sim = [np.array(dat["b_cpp_sim"]) for dat in data_list]
    # b_fit = [np.array(dat["b_fit"]) for dat in data_list]
    b_rec = [np.array(dat["b_from_reconst"]) for dat in data_list]
    sigmar_principal = [np.array(dat["sigmar_principal"]) for dat in data_list] # same as b_from_rec
    b_CI682_up = [np.array(dat["sigmar_CI682_up"]) for dat in data_list]
    b_CI682_dn = [np.array(dat["sigmar_CI682_dn"]) for dat in data_list]
    b_CI95_up = [np.array(dat["sigmar_CI95_up"]) for dat in data_list]
    b_CI95_dn = [np.array(dat["sigmar_CI95_dn"]) for dat in data_list]
    if use_real_data:
        b_hera = [dat["b_hera"] for dat in data_list]
        b_err = [dat["b_errs"] for dat in data_list]
        # dip_rec_from_b_plus_err = [dat["N_reconst_from_b_plus_err"] for dat in data_list]
        # dip_rec_from_b_minus_err = [dat["N_reconst_from_b_minus_err"] for dat in data_list]
        # b_plus_err_from_reconst = [dat["b_plus_err_from_reconst"] for dat in data_list]
        # b_minus_err_from_reconst = [dat["b_minus_err_from_reconst"] for dat in data_list]    


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
    ### PLOT TYPE sigma_r comparison
    ####################
    if not use_real_data:
        plt1_xbj_bins = [xbj_bins.index(1e-2), xbj_bins.index(1e-3),xbj_bins.index(1e-5),]
        # plt1_xbj_bins = [xbj_bins.index(1e-2), xbj_bins.index(1e-3),xbj_bins.index(1e-4),xbj_bins.index(1e-5),]
    else:
        plt1_xbj_bins = [xbj_bins.index(1.3e-2), xbj_bins.index(1.3e-3), xbj_bins.index(1.3e-4)]
    for xbj, dat in zip(xbj_bins, data_list):
        # print(dat.keys())
        if (str(xbj) not in dat["run_file"][0]):
            print("SORT ERROR?", str(xbj), dat["run_file"][0])
        print(dat["dip_file"])
    if not plt1_xbj_bins:
        print("No xbj bins found!", xbj_bins)
        exit()
    binned_dip_data_fit = [dip_data_fit[i] for i in plt1_xbj_bins]
    binned_dip_data_rec = [dip_data_rec[i] for i in plt1_xbj_bins]
    binned_b_cpp_sim = [b_cpp_sim[i] for i in plt1_xbj_bins]
    # binned_b_fit = [b_fit[i] for i in plt1_xbj_bins]
    binned_b_rec = [b_rec[i] for i in plt1_xbj_bins]
    binned_b_rec_up = [b_CI682_up[i] for i in plt1_xbj_bins]
    binned_b_rec_dn = [b_CI682_dn[i] for i in plt1_xbj_bins]
    binned_b_rec_95_up = [b_CI95_up[i] for i in plt1_xbj_bins]
    binned_b_rec_95_dn = [b_CI95_dn[i] for i in plt1_xbj_bins]
    binned_qsq_grids = [Q2vals_grid[i] for i in plt1_xbj_bins]
    if use_real_data:
        binned_b_hera = [b_hera[i].T for i in plt1_xbj_bins]
        binned_b_err = [b_err[i].T for i in plt1_xbj_bins]
        
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

    ##############
    # LABELS

    scalings = [1, 1, 1]
    additives = [0, 2, 4]
    colors = ["blue", "green", "brown", "orange", "magenta", "red"]
    lw=2.8
    ms=4
    mstyle = "o"
    color_alph = 1

    uncert_col0 = Patch(facecolor=colors[1], alpha=color_alph)
    uncert_col1 = Patch(facecolor=colors[3], alpha=color_alph)
    uncert_col2 = Patch(facecolor=colors[5], alpha=color_alph)
    uncert_col3 = Patch(facecolor=colors[2], alpha=color_alph)
    uncert_col4 = Patch(facecolor=colors[4], alpha=color_alph)
    line_fit0 = Line2D([0,1],[0,1],linestyle=':',linewidth=lw, color="black")
    
    uncert_col = Patch(facecolor="black", alpha=0.3)
    line_rec = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color="black")
    data_square = Line2D([], [], color='black', marker=mstyle, linestyle='None', markersize=ms)

    if use_real_data:
        # Now we want to plot the HERA data POINTS, not a fit curve
        manual_handles = [
                    data_square, 
                    # (line_fit0, uncert_col), # show HERA data, fit, AND reconstruction
                    (line_fit0,), # show HERA data, fit, AND reconstruction
                    (line_rec, uncert_col),
                    uncert_col0,
                    uncert_col1,
                    uncert_col2,
        ]
        if use_charm:
            manual_labels = [
                r'${\mathrm{HERA ~ data ~ } \sigma_r}$',
                r'${\sigma_r ~ \mathrm{prediction ~ for ~ f \in \{u,d,s,c\} ~ from ~ fit} ~ (\sigma_0 ~ \mathrm{refit})}$',
                r'${\sigma_r ~ \mathrm{from ~ reconstructed ~ dipole}\, \pm \, 95\% \, \mathrm{C.I.}}$',
            ]
        else:
            manual_labels = [
                r'${\mathrm{HERA ~ data ~ } \sigma_r}$',
                # r'${\sigma_r ~ \mathrm{from ~ fit}\, \pm \, 2\sigma}$',
                r'${\sigma_r ~ \mathrm{from ~ fit}}$',
                r'${\sigma_r ~ \mathrm{from ~ reconstructed ~ dipole}\, \pm \, 95\% \, \mathrm{C.I.}}$',
            ]
        sigr_marker = "s"
        sigr_linestyle = ""
    else:
        # Plotting sigma_r from fit as a curve
        manual_handles = [
                    line_fit0, 
                    (line_rec, uncert_col),
                    uncert_col0,
                    uncert_col1,
                    uncert_col2,
                    uncert_col3,
                    uncert_col4,
        ]
        manual_labels = [
            r'${\sigma_r ~ \mathrm{from ~ fit}}$',
            r'${\sigma_r ~ \mathrm{from ~ reconstructed ~ dipole}\, \pm \, 95\% \, \mathrm{C.I.}}$',
            # r'${\mathrm{Reconstructed ~ dipole}\, \pm \, \varepsilon_\lambda}$',
        ]
        sigr_marker = ""
        sigr_linestyle = ":"
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
    for i, (b_data, b_rec) in enumerate(zip(binned_b_cpp_sim, binned_b_rec)):
        x_srted, b_data = zip(*sorted(zip(xvar[i], b_data)))
        x_srted, b_rec = zip(*sorted(zip(xvar[i], b_rec)))
        # x_srted, b_rec_up = zip(*sorted(zip(xvar[i], binned_b_rec_up[i])))
        # x_srted, b_rec_dn = zip(*sorted(zip(xvar[i], binned_b_rec_dn[i])))
        ax.plot(x_srted, b_data,
                label="Fit sigma",
                linestyle=":",
                linewidth=lw*1.,
                # color=colors[2*i]
                color="black"
                )
        # ax.plot(xvar[i], scalings[i%3]*b_rec+additives[i%3],
        ax.plot(x_srted, b_rec,
                label="Reconstuction sigma",
                linestyle="-",
                linewidth=lw/3,
                color=colors[2*i+1],
                alpha=1
                )
        # ax.plot(x_srted, b_rec_up,
        #         label="b_plus_from_rec",
        #         linestyle="-.",
        #         linewidth=lw/4,
        #         # marker="",
        #         # markersize=ms,
        #         color=colors[2*i+1],
        #         alpha=0.8
        #         )
        # ax.plot(x_srted, b_rec_dn,
        #         label="b_minus_from_rec",
        #         linestyle=":",
        #         linewidth=lw/4,
        #         # marker="",
        #         # markersize=ms,
        #         color=colors[2*i+1],
        #         alpha=0.8
        #         )
        if use_real_data:
            x_srted, b_hera = zip(*sorted(zip(xvar[i], binned_b_hera[i][0])))
            x_srted, b_err = zip(*sorted(zip(xvar[i], binned_b_err[i][0])))
            # x_srted, b_plus_from_rec = zip(*sorted(zip(xvar[i], binned_b_plus_err[i][0])))
            # x_srted, b_minus_from_rec = zip(*sorted(zip(xvar[i], binned_b_minus_err[i][0])))
            ax.plot(x_srted, b_hera,
                label="HERA data",
                linestyle="",
                marker=mstyle,
                markersize=ms,
                color=colors[2*i+1],
                alpha=1
                )
            ax.errorbar(x_srted, b_hera, yerr=b_err,
                        linestyle="",
                        color=colors[2*i+1],
                )

    ################## SHADING
    # Plot reconstruction uncertainties by plotting and shading between adjacent lambdas
    i=0
    xvar = binned_qsq_grids
    for i, xbj in enumerate(plt1_xbj_bins):        
        x_srted, rec_sig = zip(*sorted(zip(xvar[i], binned_b_rec[i])))
        x_srted, b_rec_up = zip(*sorted(zip(xvar[i], binned_b_rec_up[i])))
        x_srted, b_rec_dn = zip(*sorted(zip(xvar[i], binned_b_rec_dn[i])))
        x_srted, b_rec_95_up = zip(*sorted(zip(xvar[i], binned_b_rec_95_up[i])))
        x_srted, b_rec_95_dn = zip(*sorted(zip(xvar[i], binned_b_rec_95_dn[i])))
        rec_sig = np.array(rec_sig)[:,0]
        b_rec_up = np.array(b_rec_up)[:,0]
        b_rec_dn = np.array(b_rec_dn)[:,0]
        b_rec_95_up = np.array(b_rec_95_up)[:,0]
        b_rec_95_dn = np.array(b_rec_95_dn)[:,0]
        ax.fill_between(x_srted, b_rec_dn, b_rec_up, color=colors[2*i+1], alpha=0.2)
        ax.fill_between(x_srted, b_rec_95_dn, b_rec_95_up, color=colors[2*i+1], alpha=0.1)
        # ax.fill_between(x_srted, 1.01*rec_sig, 0.99*rec_sig, color=colors[2*i], alpha=0.4)
        # i+=1


    plt.legend(manual_handles, manual_labels, frameon=False, fontsize=14, ncol=1, loc="upper left") 
    

    # ax.xaxis.set_major_formatter(ScalarFormatter())
    # plt.xlim(1e-3, 20)
    # plt.xlim(0.1, 200)
    # plt.ylim(bottom=0, top=6)
    fig.set_size_inches(7,7)
    
    if not use_real_data:
        if not use_charm:
            n_plot = "plotg2-"
        elif use_charm:
            n_plot = "plotg4-"
    elif use_real_data:
        if not use_charm:
            n_plot = "plotg6-"
        elif use_charm:
            n_plot = "plotg8-"
    
    if not n_plot:
        print("Plot number?")
        exit()

    write2file = False
    write2file = True
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

# Production plotting
main(use_charm=False,real_data=False,fitname_i=3)
main(use_charm=True,real_data=False,fitname_i=3)
main(use_charm=False,real_data=False,fitname_i=4)
main(use_charm=True,real_data=False,fitname_i=4)
# main(use_charm=False,real_data=True,fitname_i=3)
# main(use_charm=True,real_data=True,fitname_i=3)
