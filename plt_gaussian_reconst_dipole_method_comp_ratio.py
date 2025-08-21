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


def main(use_charm=False, real_data=False, use_log=True, big_bins=False, ratio=False, wide=False):
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
    # use_unity_sigma0 = True # ?
    use_noise = False
    # use_noise = True

    #        0        1        2        3           4
    fits = ["MV", "MVgamma", "MVe", "bayesMV4", "bayesMV5"]
    fitname1 = fits[3] + "_"
    fitname2 = fits[4] + "_"

    ####################
    # Data filename settings
    # data_path = "./reconstructions/"
    data_path = "./reconstructions_gausserr/"
    str_data = "sim_"
    str_fit = fitname1
    str_fit2 = fitname2
    str_flavor = "lightonly_"
    # name_base = 'recon_gausserr_v4-2'
    # name_base = 'recon_gausserr_v4-2r500_'
    name_base = 'method_comp_recon_gausserr_v4-4r256_'
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
    composite_fname2 = name_base+str_data+str_fit2+str_flavor+lambda_type
    print(composite_fname)

    # Reading data files
    recon_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname in i]
    # recon_files2 = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname2 in i]
    print(recon_files)
    # print(recon_files2)
    if not recon_files:
        print("No files found!")
        print(data_path, composite_fname)
        exit(None)
    # xbj_bins = sorted([float(Path(i).stem.split("xbj")[1]) for i in recon_files])
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in recon_files]
    # print(xbj_bins)
    recon_files = [x for _, x in sorted(zip(xbj_bins, recon_files))]
    # recon_files2 = [x for _, x in sorted(zip(xbj_bins, recon_files2))]
    xbj_bins = sorted(xbj_bins)
    print(xbj_bins)

    data_list = [loadmat(data_path + fle) for fle in recon_files]
    # data_list2 = [loadmat(data_path + fle) for fle in recon_files2]

    
    # Reading data
    R_GRID = data_list[0]["r_grid"][0]
    XBJ = np.array(xbj_bins)
    real_sigma = data_list[0]["real_sigma"][0]
    # real_sigma2 = data_list2[0]["real_sigma"][0]
    # "lambda", "lambda_type", "NUM_SAMPLES", "eps_neg_penalty"
    print(data_list[0]["lambda"][0], data_list[0]["eps_neg_penalty"][0])

    dip_data_fit = np.array([dat["N_fit"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec = np.array([dat["N_rec_principal"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec_tikh0 = np.array([dat["rec_tikh0"] for dat in data_list])
    dip_data_rec_tikh1 = np.array([dat["rec_tikh1"] for dat in data_list])
    dip_data_rec_tikh2 = np.array([dat["rec_tikh2"] for dat in data_list])
    dip_data_rec_cimmino = np.array([dat["rec_cimmino"] for dat in data_list])
    dip_data_rec_cimmino1 = np.array([dat["rec_cimmino1"] for dat in data_list])
    dip_data_rec_cimmino2 = np.array([dat["rec_cimmino2"] for dat in data_list])
    dip_data_rec_kacz = np.array([dat["rec_kacz"] for dat in data_list])
    dip_data_rec_kacz1 = np.array([dat["rec_kacz1"] for dat in data_list])
    dip_data_rec_kacz2 = np.array([dat["rec_kacz2"] for dat in data_list])


    # dip_data_fit2 = np.array([dat["N_fit"] for dat in data_list2]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    # dip_data_rec2 = np.array([dat["N_reconst"] for dat in data_list2]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid

    # for num of data points in sigmar
    b_cpp_sim = [np.array(dat["b_cpp_sim"]) for dat in data_list]


    ####################
    ### PLOT TYPE DIPOLE reconstruction method comparison
    ####################
    # alt_bins = True
    alt_bins = False

    if not use_real_data:
        plt1_xbj_bins = [xbj_bins.index(1e-2), xbj_bins.index(1e-3),xbj_bins.index(1e-5),]
        if big_bins:
            plt1_xbj_bins = [xbj_bins.index(1e-2), xbj_bins.index(5e-3), xbj_bins.index(1e-3), 
                             xbj_bins.index(8e-4), xbj_bins.index(1.3e-4),xbj_bins.index(1e-5)]
    else:
        if alt_bins:
            plt1_xbj_bins = [xbj_bins.index(8e-2), xbj_bins.index(5e-2), xbj_bins.index(8e-3)]
        else:
            plt1_xbj_bins = [xbj_bins.index(1.3e-2), xbj_bins.index(1.3e-3), xbj_bins.index(1.3e-4)]
    # print("plt1_xbj_bins", plt1_xbj_bins)
    for i in plt1_xbj_bins:
        xbj = xbj_bins[i]
        print(xbj, data_list[i]["run_file"], data_list[i]["dip_file"])
    for xbj, dat in zip(xbj_bins, data_list):
        # print(dat.keys())
        if (str(xbj) not in dat["dip_file"][0]):
            print("SORT ERROR?", str(xbj), dat["dip_file"][0])
    if not plt1_xbj_bins:
        print("No xbj bins found!", xbj_bins)
        exit()
    binned_dip_data_fit = [dip_data_fit[i] for i in plt1_xbj_bins]
    binned_dip_data_rec = [dip_data_rec[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_tikh0 = [dip_data_rec_tikh0[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_tikh1 = [dip_data_rec_tikh1[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_tikh2 = [dip_data_rec_tikh2[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_cimmino = [dip_data_rec_cimmino[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_cimmino1 = [dip_data_rec_cimmino1[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_cimmino2 = [dip_data_rec_cimmino2[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_kacz = [dip_data_rec_kacz[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_kacz1 = [dip_data_rec_kacz1[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_kacz2 = [dip_data_rec_kacz2[i] for i in plt1_xbj_bins]
    binned_b_cpp_sim = [b_cpp_sim[i] for i in plt1_xbj_bins]


    fig = plt.figure()
    if big_bins:
        gs = fig.add_gridspec(2, 3, hspace=0, wspace=0)
    else:
        gs = fig.add_gridspec(1, 3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    # ax = plt.gca()
    fs_labels = 18
    
    # use_log = True
    # use_log = False
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
        axs.flatten()[0].set_ylabel(r'$\frac{\sigma_0}{2} N(r) ~ \left(\mathrm{mb}\right)$', fontsize=fs_labels)
        if big_bins:
            axs.flatten()[3].set_ylabel(r'$\frac{\sigma_0}{2} N(r) ~ \left(\mathrm{mb}\right)$', fontsize=fs_labels)
        for ax in axs.flatten():
            ax.set_xlabel(r'$r ~ \left(\mathrm{GeV}^{-1} \right)$', fontsize=fs_labels)
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
        for ax in axs.flatten():
            ax.set_xlabel(r'$r ~ \left(\mathrm{GeV}^{-1} \right)$', fontsize=fs_labels)
            ax.xaxis.set_major_formatter(ScalarFormatter())
        xvar = data_list[0]["r_grid"][0]

    
    ##############
    # LABELS

    scalings = [1, 1, 1]
    if use_real_data:
        additives = [0, 10, 20]
    else:
        # additives = [0, 2, 4]
        additives = [0, 0, 0]
    # method colors, [1] "reserved" for Tikhonov1
    # colors = ["xkcd:aquamarine", "yellow", "orange", "red", "pink", "magenta", "violet", "xkcd:sky blue", "xkcd:bright blue", "xkcd:dark blue"]
    # colors = ["xkcd:greenish cyan", "yellow", "xkcd:deep pink", "xkcd:dusty red", "xkcd:pinkish red", "xkcd:bright blue", "xkcd:periwinkle"]
    colors = ["xkcd:vibrant green", "yellow", "xkcd:vibrant green", "xkcd:pinkish red", "xkcd:pinkish red", "xkcd:bright blue", "xkcd:bright blue"]
    lw=1.75
    ms=4
    mstyle = "o"
    color_alph = 1
    shade_alph_closer = 0.18
    shade_alph_further = 0.08
    mpl.rcParams["font.size"] = 13
    mpl.rcParams["legend.fontsize"] = 13

    col_i = 0
    col_fit = "black"

    uncert_col0 = Patch(facecolor=colors[col_i], alpha=shade_alph_closer+shade_alph_further)
    uncert_col0b = Patch(facecolor=colors[col_i], alpha=shade_alph_further)
    uncert_col1 = Patch(facecolor=colors[4], alpha=color_alph)
    uncert_col2 = Patch(facecolor=colors[5], alpha=color_alph)
    line_fit0 = Line2D([0,1],[0,1],linestyle=':',linewidth=lw/1.5, color=col_fit)
    
    uncert_col = Patch(facecolor=colors[col_i], alpha=0.3)
    line_rec = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/1.5, color=colors[0])
    # line_mthd1 = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/1.5, color=colors[1])
    line_mthd2 = Line2D([0,1],[0,1],linestyle='--',linewidth=lw/1.5, color=colors[2])
    line_mthd3 = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/1.5, color=colors[3])
    line_mthd4 = Line2D([0,1],[0,1],linestyle='--',linewidth=lw/1.5, color=colors[4])
    line_mthd5 = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/1.5, color=colors[5])
    line_mthd6 = Line2D([0,1],[0,1],linestyle='--',linewidth=lw/1.5, color=colors[6])
    # line_mthd7 = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/1.5, color=colors[7])
    # line_mthd8 = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/1.5, color=colors[8])
    # line_mthd9 = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/1.5, color=colors[9])
    line_rec_bplus = Line2D([0,1],[0,1],linestyle='-.',linewidth=lw/3, color="black")
    line_rec_bminus = Line2D([0,1],[0,1],linestyle=':',linewidth=lw/3, color="black")


    if wide:
            manual_handles = [
                        line_fit0,
                        (line_rec, uncert_col),
                        uncert_col0, uncert_col0b,
                        uncert_col1,
                        uncert_col2,]
    else:
            manual_handles = [
                line_rec,
                # line_mthd1,
                line_mthd2,
                line_mthd3,
                line_mthd4,
                line_mthd5,
                line_mthd6,
                ]


    if wide:
        manual_labels = [
            r'${\frac{N^{\mathrm{5-param.}}_{\mathrm{fit}}}{N^{\mathrm{4-param.}}_{\mathrm{fit}}}}$',
            r'${\frac{N^{\mathrm{5-param.}}_{\mathrm{rec.}}}{N^{\mathrm{4-param.}}_{\mathrm{rec.}}}}$',
            r'${68 \% \, \mathrm{C.I.}}$',
            r'${95 \% \, \mathrm{C.I.}}$',
        ]
    else:
        manual_labels = [
            r'${\mathrm{Principal ~ method}}$',
            # r'${\mathrm{0th order Tikhonov}}$',
            # r'${\mathrm{1st order Tikhonov}}$',
            r'${\mathrm{2nd ~ order ~ Tikhonov}}$',
            # r'${\mathrm{Cimmino}}$',
            r'${\mathrm{1st ~ order ~ PCimmino}}$',
            r'${\mathrm{2nd ~ order ~ PCimmino}}$',
            # r'${\mathrm{Kaczmarz}}$',
            r'${\mathrm{1st ~ order ~ PKaczmarz}}$',
            r'${\mathrm{2nd ~ order ~ PKaczmarz}}$',
        ]

    ####################
    #################### PLOTTING
    #################### 

    # Plot fit dipoles and their reconstructions
    for i, (dip_fit, dip_rec) in enumerate(zip(binned_dip_data_fit, binned_dip_data_rec)):
        ax = axs.flatten()[i]
        dip_fit = real_sigma*dip_fit.T
        # print(dip_fit.shape, dip_rec.shape, binned_dip_data_rec_tikh0[i].shape)
        if PLOT_TYPE == "ratio":
            ax.plot(xvar, [1]*len(xvar), linestyle="--", linewidth=lw/4.5, color="black", alpha=1)    
            # ax.plot(xvar, binned_dip_data_rec_tikh0[i]/dip_fit, linestyle="-", linewidth=lw/1.5, color=colors[1])
            # ax.plot(xvar, binned_dip_data_rec_tikh1[i]/dip_fit, linestyle=":", linewidth=lw/1.5, color=colors[1])
            # ax.plot(xvar, binned_dip_data_rec_cimmino[i]/dip_fit, linestyle="-", linewidth=lw/1.5, color=colors[4])
            ax.plot(xvar, binned_dip_data_rec_cimmino1[i]/dip_fit, linestyle="-", linewidth=lw/1.5, color=colors[3])
            ax.plot(xvar, binned_dip_data_rec_cimmino2[i]/dip_fit, linestyle="--", linewidth=lw/1.5, color=colors[4])
            # ax.plot(xvar, binned_dip_data_rec_kacz[i]/dip_fit, linestyle="-", linewidth=lw/1.5, color=colors[7])
            ax.plot(xvar, binned_dip_data_rec_kacz1[i]/dip_fit, linestyle="-", linewidth=lw/1.5, color=colors[5])
            ax.plot(xvar, binned_dip_data_rec_kacz2[i]/dip_fit, linestyle="--", linewidth=lw/1.5, color=colors[6])
            ax.plot(xvar, dip_rec/dip_fit, linestyle="-", linewidth=lw/1.5, color=colors[0])
            ax.plot(xvar, binned_dip_data_rec_tikh2[i]/dip_fit, linestyle="--", linewidth=lw/1.5, color=colors[2])




    if big_bins:
        leg = axs.flatten()[5].legend(manual_handles, manual_labels, frameon=False, fontsize=12, ncol=1, loc="lower right") 
    elif PLOT_TYPE == "ratio":
        leg = axs.flatten()[0].legend(manual_handles, manual_labels, frameon=False, fontsize=12, ncol=1, loc="upper right") 
    
    h_align = 'left'
    if wide==True:
        x_crd = 0.05
        y_crd = 0.98
    elif wide==False:
        h_align = 'right'
        x_crd = 0.98
        y_crd = 0.14
    elif use_log == False:
        h_align = 'right'
        x_crd = 0.98
        y_crd = 0.97
    else:
        x_crd = 0.05
        y_crd = 0.98

    for i, ibin in enumerate(plt1_xbj_bins):
        xbj_str = str(xbj_bins[ibin])
        if "e" in xbj_str:
            # xbj_str = "0.00001" #"10^{{-5}}"
            xbj_str = "10^{{-5}}"
        # manual_labels.append('$x_{{\\mathrm{{Bj.}} }} = {xbj}$'.format(xbj = xbj_str))
        # xbj_label=offsetbox.TextArea('$x_{{\\mathrm{{Bj.}} }} = {xbj}$'.format(xbj = xbj_str),)
        x_lbl = '$x_{{\\mathrm{{Bj.}} }} = {xbj}$'.format(xbj = xbj_str)
        # print(x_lbl, i)
        axs.flatten()[i].text(x_crd, y_crd, x_lbl,
            horizontalalignment=h_align,
            verticalalignment='top',
            fontsize=15,
            transform=axs.flatten()[i].transAxes)
        n_datapoints = len(b_cpp_sim[ibin])
        n_lbl = '$N_{{\\mathrm{{datap.}} }}(\\sigma_r) = {num}$'.format(num = n_datapoints)

        axs.flatten()[i].text(x_crd, y_crd-0.06, n_lbl,
            horizontalalignment=h_align,
            verticalalignment='top',
            fontsize=15,
            transform=axs.flatten()[i].transAxes)
    
    use_wide_view = wide
    if use_wide_view == True:
        plt.xlim(0.05, 25)
        plt.ylim(bottom=0.1, top=5)
    else:
        plt.xlim(0.05, 25)
        # plt.ylim(bottom=0.985, top=1.065)
        # plt.ylim(bottom=0.1, top=5)
        plt.ylim(bottom=0.5, top=1.5)
    if big_bins:
        fig.set_size_inches(12,8)
    else:
        fig.set_size_inches(15,5.5)
    
    if not use_real_data:
        if not use_charm:
            n_plot = "plotg0-method_comp-"
        elif use_charm:
            n_plot = "plotg0c-method_comp-"
    # elif use_real_data:
    #     if not use_charm:
    #         n_plot = "plotg5-"
    #     elif use_charm:
    #         n_plot = "plotg7-"
    
    if not use_log:
        n_plot += "not_log-"

    if alt_bins:
        n_plot += "alt_bins-"
    
    if big_bins:
        n_plot += "BIG_BINS-"

    if PLOT_TYPE=="ratio":
        n_plot += "RATIO-"

    if use_wide_view:
        n_plot += "WIDE-"
    else:
        n_plot += "zoomin-"
        


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

log=False
big=False
ratio=True

main(use_charm=False,real_data=False, use_log=log, big_bins=big, ratio=ratio, wide=False)
# main(use_charm=False,real_data=False, use_log=log, big_bins=big, ratio=ratio, wide=True)
# main(use_charm=True,real_data=False, use_log=log, big_bins=big, ratio=ratio)

# Production plotting
# main(use_charm=False,real_data=False,fitname_i=3)
# main(use_charm=True,real_data=False,fitname_i=3)
# main(use_charm=False,real_data=False,fitname_i=4)
# main(use_charm=True,real_data=False,fitname_i=4)
# main(use_charm=False,real_data=True,fitname_i=3)
# main(use_charm=True,real_data=True,fitname_i=3)
