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
import matplotlib.offsetbox as offsetbox
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


def main(use_charm=False, real_data=False, fitname_i=None, q_mass_scheme=None, q_cut=None, use_log=True, big_bins=False, ratio=False):
    global G_PATH, PLOT_TYPE, R_GRID
    f_path_list = []
    # PLOT_TYPE = sys.argv[1]
    PLOT_TYPE = "dipole"
    if ratio:
        PLOT_TYPE = "ratio"
        use_log=False
    if PLOT_TYPE not in ["dipole", "sigmar", "ratio"]:
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
    data_path = "./reconstructions_hera_uq/"
    name_base = 'hera_recon_uq_256_'
    str_data = "heraII_"
    str_process = "dis_inclusive"
    str_sqrts = "s318.1_"
    str_data += str_process + str_sqrts
    str_fit = "data_only_"
    # str_fit = fitname # todo need to implement fit dipole importing

    if use_charm and not q_mass_scheme:
        str_flavor = "lightpluscharm_"
    else:
        str_flavor = q_mass_scheme + "_"

    if q_cut==None:
        q_cut_str = "no_Q_cut_"
    elif q_cut=="low":
        q_cut_str = "cut_low_Q_"
    elif q_cut=="high":
        q_cut_str = "cut_high_Q_"

    
    # lambda_type = "broad_"
    lambda_type = "lambdaSRN_"
    composite_fname = name_base+str_data+str_fit+str_flavor+lambda_type+q_cut_str
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
    # real_sigma = data_list[0]["real_sigma"][0] # TODO implement reference fit
    real_sigma = 1
    # "lambda", "lambda_type", "NUM_SAMPLES", "eps_neg_penalty"

    # dip_data_fit = np.array([dat["N_fit"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec = np.array([dat["N_reconst"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec_ptw_mean = np.array([dat["N_rec_ptw_mean"] for dat in data_list])
    dip_data_rec_CI682_up = np.array([dat["N_rec_CI682_up"] for dat in data_list])
    dip_data_rec_CI682_dn = np.array([dat["N_rec_CI682_dn"] for dat in data_list])
    dip_data_rec_CI95_up = np.array([dat["N_rec_CI95_up"] for dat in data_list])
    dip_data_rec_CI95_dn = np.array([dat["N_rec_CI95_dn"] for dat in data_list])

    b_hera_data = [dat["b_hera"] for dat in data_list] # sigma_r in each xbj bin


    ####################
    ### PLOT TYPE DIPOLE
    ####################
    # alt_bins = True
    alt_bins = False

    # real data bins
    # [0.00013, 0.0002, 0.00032, 0.0005, 0.0008, 0.0013, 0.002, 0.0032, 0.005, 0.008, 0.013, 0.02, 0.032, 0.05, 0.08]

    if not use_real_data:
        plt1_xbj_bins = [xbj_bins.index(1e-2), xbj_bins.index(1e-3),xbj_bins.index(1e-5),]
        if big_bins:
            plt1_xbj_bins = [xbj_bins.index(1e-2), xbj_bins.index(5e-3), xbj_bins.index(1e-3), 
                             xbj_bins.index(8e-4), xbj_bins.index(1.3e-4),xbj_bins.index(1e-5)]
    else:
        if big_bins:
            # plt1_xbj_bins = [xbj_bins.index(0.00013), xbj_bins.index(5e-3), xbj_bins.index(1e-3), 
            #                  xbj_bins.index(8e-4), xbj_bins.index(1.3e-4),xbj_bins.index(1e-5)]
            plt1_xbj_bins = range(len(xbj_bins))
        else:
            plt1_xbj_bins = [xbj_bins.index(1.3e-2), xbj_bins.index(1.3e-3), xbj_bins.index(1.3e-4)]
    # print("plt1_xbj_bins", plt1_xbj_bins)
    for i in plt1_xbj_bins:
        xbj = xbj_bins[i]
        print(xbj, data_list[i]["run_file"])
        print(data_list[i]["N_reconst"].T[0][:5])
    # for xbj, dat in zip(xbj_bins, data_list):
    #     # print(dat.keys())
    #     if (str(xbj) not in dat["dip_file"][0]):
    #         print("SORT ERROR?", str(xbj), dat["dip_file"][0])
    if not plt1_xbj_bins:
        print("No xbj bins found!", xbj_bins)
        exit()
    # binned_dip_data_fit = [dip_data_fit[i] for i in plt1_xbj_bins]
    binned_dip_data_rec = [dip_data_rec[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_ptw_mean = [dip_data_rec_ptw_mean[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_std_up = [dip_data_rec_CI682_up[i].T for i in plt1_xbj_bins]
    binned_dip_data_rec_std_dn = [dip_data_rec_CI682_dn[i].T for i in plt1_xbj_bins]
    binned_dip_data_rec_CI95_up = [dip_data_rec_CI95_up[i].T for i in plt1_xbj_bins]
    binned_dip_data_rec_CI95_dn = [dip_data_rec_CI95_dn[i].T for i in plt1_xbj_bins]


    fig = plt.figure()
    if big_bins:
        if len(plt1_xbj_bins) > 6:
            gs = fig.add_gridspec(5, 3, hspace=0, wspace=0)
        else:
            gs = fig.add_gridspec(2, 3, hspace=0, wspace=0)
    else:
        gs = fig.add_gridspec(1, 3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    # ax = plt.gca()
    fs_labels = 18
    
    for ax in axs.flatten():
        ax.tick_params(which='major',width=1,length=6,labelsize=18)
        ax.tick_params(which='minor',width=0.7,length=4,labelsize=18)
        ax.tick_params(axis='both', pad=7)
        ax.tick_params(axis='both', which='both', direction="in")
        # LOG AXIS
        ax.set_xscale('log')
        if use_log:
            ax.set_yscale('log')

    if PLOT_TYPE == "dipole":
        axs.flatten()[0].set_ylabel(r'$\frac{\sigma_0}{2} N(r) ~ \left(\mathrm{mb}\right)$', fontsize=fs_labels+2)
        if big_bins:
            axs.flatten()[3].set_ylabel(r'$\frac{\sigma_0}{2} N(r) ~ \left(\mathrm{mb}\right)$', fontsize=fs_labels+2)
            axs.flatten()[6].set_ylabel(r'$\frac{\sigma_0}{2} N(r) ~ \left(\mathrm{mb}\right)$', fontsize=fs_labels+2)
            axs.flatten()[9].set_ylabel(r'$\frac{\sigma_0}{2} N(r) ~ \left(\mathrm{mb}\right)$', fontsize=fs_labels+2)
            axs.flatten()[12].set_ylabel(r'$\frac{\sigma_0}{2} N(r) ~ \left(\mathrm{mb}\right)$', fontsize=fs_labels+2)
        for ax in axs.flatten():
            ax.set_xlabel(r'$r ~ \left(\mathrm{GeV}^{-1} \right)$', fontsize=fs_labels)
            ax.xaxis.set_major_formatter(ScalarFormatter())
        # xvar = data_list[0]["r_grid"][0][:-1]
        xvar = data_list[0]["r_grid"][0]
    elif PLOT_TYPE == "sigmar":
        plt.xlabel(r'$Q^2 ~ \left(\mathrm{GeV}^{2} \right)$', fontsize=fs_labels)
        plt.ylabel(r'$\sigma_r ~ \left(\mathrm{GeV}^{-2} \right)$', fontsize=fs_labels)
        xvar = data_list[0]["q2vals"]
    elif PLOT_TYPE == "ratio":
        axs.flatten()[0].set_ylabel(r'$\frac{N_{\mathrm{rec.}}}{N_{\mathrm{fit}}}$', fontsize=fs_labels+6)
        if big_bins:
            axs.flatten()[3].set_ylabel(r'$\frac{N_{\mathrm{rec.}}}{N_{\mathrm{fit}}}$', fontsize=fs_labels)
            axs.flatten()[6].set_ylabel(r'$\frac{N_{\mathrm{rec.}}}{N_{\mathrm{fit}}}$', fontsize=fs_labels)
            axs.flatten()[9].set_ylabel(r'$\frac{N_{\mathrm{rec.}}}{N_{\mathrm{fit}}}$', fontsize=fs_labels)
            axs.flatten()[12].set_ylabel(r'$\frac{N_{\mathrm{rec.}}}{N_{\mathrm{fit}}}$', fontsize=fs_labels)
        for ax in axs.flatten():
            ax.set_xlabel(r'$r ~ \left(\mathrm{GeV}^{-1} \right)$', fontsize=fs_labels)
            ax.xaxis.set_major_formatter(ScalarFormatter())
        xvar = data_list[0]["r_grid"][0]

    
    ##############
    # LABELS

    colors = ["blue", "green", "brown", "orange", "magenta", "red"]
    lw=2.8
    ms=4
    mstyle = "o"
    color_alph = 1
    shade_alph_closer = 0.18
    shade_alph_further = 0.08
    mpl.rcParams["font.size"] = 13
    mpl.rcParams["legend.fontsize"] = 13

    if fitname_i==3:
        col_i = 1
    elif fitname_i==4:
        col_i = 3

    uncert_col0 = Patch(facecolor=colors[col_i], alpha=shade_alph_closer+shade_alph_further)
    uncert_col0b = Patch(facecolor=colors[col_i], alpha=shade_alph_further)
    uncert_col1 = Patch(facecolor=colors[4], alpha=color_alph)
    uncert_col2 = Patch(facecolor=colors[5], alpha=color_alph)
    line_fit0 = Line2D([0,1],[0,1],linestyle=':',linewidth=lw, color="black")
    
    uncert_col = Patch(facecolor=colors[col_i], alpha=0.3)
    line_rec = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color=colors[0])
    line_rec_ptw_mean = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color=colors[col_i])
    line_rec_nless = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color=colors[5])


    if use_real_data:
        manual_handles = [
                        # line_fit0,
                        (line_rec, uncert_col),
                        ]
    else:
        if PLOT_TYPE == "ratio":
            manual_handles = [
                        line_rec,
                        (line_rec_ptw_mean, uncert_col),
                        line_rec_nless,
                        uncert_col0, uncert_col0b,
                        uncert_col1,
                        uncert_col2,]
        else:
            manual_handles = [
                            line_fit0,
                            line_rec,
                            (line_rec_ptw_mean, uncert_col),
                            line_rec_nless,
                            uncert_col0, uncert_col0b,
                            uncert_col1,
                            uncert_col2,]

    if use_real_data:
        manual_labels = [
            # r'${\mathrm{Fit ~ dipole}}$',
            r'${\mathrm{Reconstruction ~ from ~ HERA ~ data}\, \pm \, \mathrm{C.I.}}$',
        ]
    else:
        if PLOT_TYPE == "ratio":
            manual_labels = [
                r'${\frac{N_{\mathrm{rec.}}}{N_{\mathrm{fit}}}}$',
                r'${\frac{N_{\mathrm{rec.}}^\mathrm{mean}}{N_{\mathrm{fit}}}}$',
                r'${\frac{N_{\mathrm{rec.}}^\mathrm{noiseless}}{N_{\mathrm{fit}}}}$',
                r'${68 \% \, \mathrm{C.I.}}$',
                r'${95 \% \, \mathrm{C.I.}}$',
            ]
        else:
            manual_labels = [
                r'${\mathrm{Fit ~ dipole}}$',
                # r'${\mathrm{Reconstructed ~ dipole}\, \pm \, \mathrm{C.I.}}$',
                r'${\mathrm{Reconstructed ~ dipole}}$',
                r'${\mathrm{Reconstruction ~ mean}\, \pm \, \mathrm{C.I.}}$',
                r'${\mathrm{Noiseless ~ reconstruction}}$',
                r'${68 \% \, \mathrm{C.I.}}$',
                r'${95 \% \, \mathrm{C.I.}}$',
            ]

    ####################
    #################### PLOTTING
    #################### 

    # Plot fit dipoles and their reconstructions
    for i, dip_rec in enumerate(binned_dip_data_rec):
        ax = axs.flatten()[i]
        if PLOT_TYPE == "ratio":
            # todo ratio of real data rec. to fit dipole might be useful?
            ax.plot(xvar, [1]*len(xvar),
                    label="ratio",
                    linestyle="--",
                    linewidth=lw/4.5,
                    color="black",
                    alpha=1
                    )    
            dip_rec_del = dip_rec.T[0]
            dip_rec_del[dip_rec_del < 0] = np.nan
            # ax.plot(xvar, dip_rec_del/(real_sigma*dip_fit[0]),
            #         label="ratio",
            #         linestyle="-",
            #         linewidth=lw/2.5,
            #         # color=colors[col_i],
            #         color=colors[0],
            #         alpha=1
            #         )
            # dip_ptw_mean = binned_dip_data_rec_ptw_mean[i].T[0]
            # ax.plot(xvar, dip_ptw_mean/(real_sigma*dip_fit[0]),
            #         label="ratio",
            #         linestyle="-",
            #         linewidth=lw/2.5,
            #         color=colors[col_i],
            #         alpha=1
            #         )
        else:
            # ax.plot(xvar, gev_to_mb*scalings[i%3]*real_sigma*dip_fit[0]+additives[i%3],
            #         label="Fit dipole",
            #         linestyle=":",
            #         # linestyle="-",
            #         # marker="+",
            #         linewidth=lw/1.5,
            #         color="black"
            #         )
            dip_rec_del = dip_rec.T[0]
            if use_log:
                dip_rec_del[dip_rec_del < 0] = np.nan
            ax.plot(xvar, gev_to_mb*dip_rec_del,
                    label="Reconstuction of fit dipole",
                    linestyle="-",
                    linewidth=lw/2.5,
                    # color=colors[col_i],
                    color=colors[0],
                    alpha=1
                    )
            dip_ptw_mean = binned_dip_data_rec_ptw_mean[i]
            ax.plot(xvar, gev_to_mb*dip_ptw_mean,
                    label="Reconstuction of fit dipole",
                    linestyle="-",
                    linewidth=lw/2.5,
                    color=colors[col_i],
                    alpha=1
                    )

    ################## SHADING        
    # Plot reconstruction uncertainties by plotting and shading between adjacent lambdas
    if PLOT_TYPE=="ratio":
        for i, (rec_up, rec_dn) in enumerate(zip(binned_dip_data_rec_std_up, binned_dip_data_rec_std_dn)):
            rec_dip = binned_dip_data_rec[i][:,0]
            ref_fit = real_sigma*binned_dip_data_fit[i][0]
            rec_up = rec_up[0]/ref_fit
            rec_dn = rec_dn[0]/ref_fit
            rec_CI95_up = binned_dip_data_rec_CI95_up[i].T[:,0]/ref_fit
            rec_CI95_dn = binned_dip_data_rec_CI95_dn[i].T[:,0]/ref_fit
            ax = axs.flatten()[i]
            ax.fill_between(xvar, rec_dn, rec_up, color=colors[col_i], alpha=shade_alph_closer)
            ax.fill_between(xvar, rec_CI95_dn, rec_CI95_up, color=colors[col_i], alpha=shade_alph_further)
    else:
        for i, (rec_up, rec_dn) in enumerate(zip(binned_dip_data_rec_std_up, binned_dip_data_rec_std_dn)):
            rec_dip = binned_dip_data_rec[i][:,0]
            rec_up = rec_up[0]
            rec_dn = rec_dn[0]
            rec_CI95_up = binned_dip_data_rec_CI95_up[i].T[:,0]
            rec_CI95_dn = binned_dip_data_rec_CI95_dn[i].T[:,0]
            # print(xvar.shape, rec_dip.shape, rec_up.shape, rec_dn.shape, rec_CI95_up.shape, rec_CI95_dn.shape)
            ax = axs.flatten()[i]
            ax.fill_between(xvar, gev_to_mb*rec_dn, gev_to_mb*rec_up, color=colors[col_i], alpha=shade_alph_closer)
            ax.fill_between(xvar, gev_to_mb*rec_CI95_dn, gev_to_mb*rec_CI95_up, color=colors[col_i], alpha=shade_alph_further)

    if big_bins:
        leg = axs.flatten()[0].legend(manual_handles, manual_labels, frameon=False, fontsize=13, handlelength=1.2, ncol=1, loc="upper left") 
    elif PLOT_TYPE == "ratio":
        leg = axs.flatten()[0].legend(manual_handles, manual_labels, frameon=True, framealpha=0.4, fontsize=16, handlelength=1.2, ncol=2, columnspacing=1, loc="upper right") 
    else:
        leg = axs.flatten()[0].legend(manual_handles, manual_labels, frameon=False, fontsize=13, handlelength=1.2, ncol=1, loc="upper left") 
    
    h_align = 'left'
    if PLOT_TYPE == "ratio":
        h_align = 'right'
        # x_crd = 0.05
        x_crd = 0.98
        y_crd = 0.14
    elif use_log == False:
        h_align = 'left'
        # x_crd = 0.98
        x_crd = 0.05
        y_crd = 0.8
    else:
        x_crd = 0.05
        y_crd = 0.98

    for i, ibin in enumerate(plt1_xbj_bins):
        xbj_str = str(xbj_bins[ibin])
        if "e" in xbj_str:
            xbj_str = "10^{{-5}}"
        x_lbl = '$x_{{\\mathrm{{Bj.}} }} = {xbj}$'.format(xbj = xbj_str)
        axs.flatten()[i].text(x_crd, y_crd, x_lbl,
            horizontalalignment=h_align,
            verticalalignment='top',
            fontsize=13,
            transform=axs.flatten()[i].transAxes)
        n_datapoints = len(b_hera_data[ibin])
        n_lbl = '$N_{{\\mathrm{{datap.}} }}(\\sigma_r) = {num}$'.format(num = n_datapoints)

        axs.flatten()[i].text(x_crd, y_crd-0.07, n_lbl,
            horizontalalignment=h_align,
            verticalalignment='top',
            fontsize=13,
            transform=axs.flatten()[i].transAxes)
    
    if use_log:
        # plt.xlim(5e-3, 25)
        plt.xlim(1e-2, 25)
        plt.ylim(bottom=1e-3)
    else:
        if PLOT_TYPE == "ratio":
            plt.xlim(0.05, 25)
            plt.ylim(bottom=0.6, top=1.4)
        else:
            plt.xlim(0.05, 25)
            # plt.ylim(bottom=0, top=30)
    # plt.ylim(bottom=1e-5)
    if big_bins:
        fig.set_size_inches(12,12)
    else:
        fig.set_size_inches(15,5.5)

    # todo should use mass scheme, data type, XBJ to name plots.
    
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
    
    if not use_log:
        n_plot += "not_log-"

    if alt_bins:
        n_plot += "alt_bins-"
    
    if big_bins:
        n_plot += "BIG_BINS-"

    if PLOT_TYPE=="ratio":
        n_plot += "RATIO-"


    if not n_plot:
        print("Plot number?")
        exit()

    write2file = False
    # write2file = True
    plt.tight_layout()
    if write2file:
        # mpl.use('agg') # if writing to PDF
        plt.draw()
        outfilename = n_plot + composite_fname + "{}".format(PLOT_TYPE) + '.pdf'
        # plotpath = G_PATH+"/inversedipole/plots_gaussian/"
        plotpath = G_PATH+"/inversedipole/plots_hera_uq/"
        print(os.path.join(plotpath, outfilename))
        plt.savefig(os.path.join(plotpath, outfilename))
    else:
        plt.show()
    return 0

log=False
big=True
ratio=False
real=True
q_cut = "low"
# q_cut = None

run_settings=[
        "standard",
        # "pole",
        "mqMpole", # this is the more accurate alternative to 'standard'
        # "mqmq",
        "mqMcharm",
        # "mqMbottom",
        "mqMW",
        # "mass_scheme_heracc_charm_only"
    ]
qms=run_settings[2]
# qms=run_settings[0]

main(use_charm=False,real_data=real,fitname_i=3, q_mass_scheme=qms, q_cut=q_cut, use_log=True, big_bins=True, ratio=False) # fig 3 big
# main(use_charm=False,real_data=real,fitname_i=3, q_mass_scheme=qms, q_cut=q_cut, use_log=False, big_bins=True, ratio=False) # fig 3 big - non log
# main(use_charm=False,real_data=real,fitname_i=3, use_log=False, big_bins=False, ratio=False) # fig 4 log-linear
# main(use_charm=False,real_data=real,fitname_i=3, use_log=False, big_bins=False, ratio=True) # fig 5 ratio
# main(use_charm=False,real_data=real,fitname_i=4, use_log=False, big_bins=False, ratio=True) # fig 7 5param ratio
# main(use_charm=True,real_data=real,fitname_i=3, use_log=False, big_bins=False, ratio=True) # fig 9 4param charm ratio

# Production plotting
# main(use_charm=False,real_data=False,fitname_i=3)
# main(use_charm=True,real_data=False,fitname_i=3)
# main(use_charm=False,real_data=False,fitname_i=4)
# main(use_charm=True,real_data=False,fitname_i=4)
# main(use_charm=False,real_data=True,fitname_i=3)
# main(use_charm=True,real_data=True,fitname_i=3)
