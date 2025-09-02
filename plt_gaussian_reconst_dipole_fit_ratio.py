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
    # name_base = 'recon_gausserr_v4-4r256_'
    name_base = 'recon_gausserr_chifit_v5r256_'
    if use_charm:
        str_flavor = "lightpluscharm_"
    if use_real_data:
        str_data = "hera_"
        str_fit = "data_only_"
    if use_noise:
        name_base = 'recon_with_noise_out_'
    
    lambda_type = "broad_"
    lambda_type = "chifit_"
    # lambda_type = "semiconstrained_"
    # lambda_type = "semicon2_"
    # lambda_type = "fixed_"
    composite_fname = name_base+str_data+str_fit+str_flavor+lambda_type
    composite_fname2 = name_base+str_data+str_fit2+str_flavor+lambda_type
    print(composite_fname)

    # Reading data files
    recon_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname in i]
    recon_files2 = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname2 in i]
    print(recon_files)
    print(recon_files2)
    if not recon_files or not recon_files2:
        print("No files found!")
        print(data_path, composite_fname)
        exit(None)
    # xbj_bins = sorted([float(Path(i).stem.split("xbj")[1]) for i in recon_files])
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in recon_files]
    # print(xbj_bins)
    recon_files = [x for _, x in sorted(zip(xbj_bins, recon_files))]
    recon_files2 = [x for _, x in sorted(zip(xbj_bins, recon_files2))]
    xbj_bins = sorted(xbj_bins)
    print(xbj_bins)

    data_list = [loadmat(data_path + fle) for fle in recon_files]
    data_list2 = [loadmat(data_path + fle) for fle in recon_files2]

    
    # Reading data
    R_GRID = data_list[0]["r_grid"][0]
    XBJ = np.array(xbj_bins)
    real_sigma = data_list[0]["real_sigma"][0]
    real_sigma2 = data_list2[0]["real_sigma"][0]
    # "lambda", "lambda_type", "NUM_SAMPLES", "eps_neg_penalty"
    print(data_list[0]["lambda"][0], data_list[0]["eps_neg_penalty"][0])

    dip_data_fit = np.array([dat["N_fit"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec = np.array([dat["N_reconst"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec_noisless = np.array([dat["N_rec_principal_noiseless"] for dat in data_list])
    dip_data_rec_ptw_mean = np.array([dat["N_rec_ptw_mean"] for dat in data_list])
    dip_data_rec_CI682_up = np.array([dat["N_rec_CI682_up"] for dat in data_list])
    dip_data_rec_CI682_dn = np.array([dat["N_rec_CI682_dn"] for dat in data_list])
    dip_data_rec_CI95_up = np.array([dat["N_rec_CI95_up"] for dat in data_list])
    dip_data_rec_CI95_dn = np.array([dat["N_rec_CI95_dn"] for dat in data_list])

    dip_data_fit2 = np.array([dat["N_fit"] for dat in data_list2]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec2 = np.array([dat["N_reconst"] for dat in data_list2]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    dip_data_rec_noisless2 = np.array([dat["N_rec_principal_noiseless"] for dat in data_list2])
    dip_data_rec_ptw_mean2 = np.array([dat["N_rec_ptw_mean"] for dat in data_list2])
    dip_data_rec_CI682_up2 = np.array([dat["N_rec_CI682_up"] for dat in data_list2])
    dip_data_rec_CI682_dn2 = np.array([dat["N_rec_CI682_dn"] for dat in data_list2])
    dip_data_rec_CI95_up2 = np.array([dat["N_rec_CI95_up"] for dat in data_list2])
    dip_data_rec_CI95_dn2 = np.array([dat["N_rec_CI95_dn"] for dat in data_list2])

    # for num of data points in sigmar
    b_cpp_sim = [np.array(dat["b_cpp_sim"]) for dat in data_list]


    ####################
    ### PLOT TYPE DIPOLE FIT RATIO
    ####################
    # alt_bins = True
    alt_bins = False


    if not use_real_data:
        plt1_xbj_bins = [xbj_bins.index(1e-2), xbj_bins.index(1e-3),xbj_bins.index(1e-5),]
        if big_bins:
            plt1_xbj_bins = [xbj_bins.index(1e-2), xbj_bins.index(5e-3), xbj_bins.index(1e-3), 
                             xbj_bins.index(8e-4), xbj_bins.index(1.3e-4),xbj_bins.index(1e-5)]
        # plt1_xbj_bins = [xbj_bins.index(1e-2)]
        # plt1_xbj_bins = [xbj_bins.index(8e-3)]
        # plt1_xbj_bins = [xbj_bins.index(1e-4)]
        # plt1_xbj_bins = [xbj_bins.index(1e-5)]
    else:
        if alt_bins:
            plt1_xbj_bins = [xbj_bins.index(8e-2), xbj_bins.index(5e-2), xbj_bins.index(8e-3)]
        else:
            plt1_xbj_bins = [xbj_bins.index(1.3e-2), xbj_bins.index(1.3e-3), xbj_bins.index(1.3e-4)]
    # print("plt1_xbj_bins", plt1_xbj_bins)
    for i in plt1_xbj_bins:
        xbj = xbj_bins[i]
        print(xbj, data_list[i]["run_file"], data_list[i]["dip_file"])
        print(real_sigma*data_list[i]["N_fit"][0][:5])
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
    binned_dip_data_rec_std_up = [dip_data_rec_CI682_up[i].T for i in plt1_xbj_bins]
    binned_dip_data_rec_std_dn = [dip_data_rec_CI682_dn[i].T for i in plt1_xbj_bins]
    binned_dip_data_rec_CI95_up = [dip_data_rec_CI95_up[i].T for i in plt1_xbj_bins]
    binned_dip_data_rec_CI95_dn = [dip_data_rec_CI95_dn[i].T for i in plt1_xbj_bins]
    binned_dip_data_fit2 = [dip_data_fit2[i] for i in plt1_xbj_bins]
    binned_dip_data_rec2 = [dip_data_rec2[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_std_up2 = [dip_data_rec_CI682_up2[i].T for i in plt1_xbj_bins]
    binned_dip_data_rec_std_dn2 = [dip_data_rec_CI682_dn2[i].T for i in plt1_xbj_bins]
    binned_dip_data_rec_CI95_up2 = [dip_data_rec_CI95_up2[i].T for i in plt1_xbj_bins]
    binned_dip_data_rec_CI95_dn2 = [dip_data_rec_CI95_dn2[i].T for i in plt1_xbj_bins]
    binned_b_cpp_sim = [b_cpp_sim[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_noiseless = [dip_data_rec_noisless[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_ptw_mean = [dip_data_rec_ptw_mean[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_noiseless2 = [dip_data_rec_noisless2[i] for i in plt1_xbj_bins]
    binned_dip_data_rec_ptw_mean2 = [dip_data_rec_ptw_mean2[i] for i in plt1_xbj_bins]


    fig = plt.figure()
    if big_bins:
        gs = fig.add_gridspec(2, 3, hspace=0, wspace=0)
    else:
        gs = fig.add_gridspec(1, 3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
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
        axs.flatten()[0].set_ylabel(r'$\frac{N^{\mathrm{5-param.}}}{N^{\mathrm{4-param.}}}$', fontsize=fs_labels+6)
        if big_bins:
            axs.flatten()[3].set_ylabel(r'$\frac{N^{\mathrm{5-param.}}}{N^{\mathrm{4-param.}}}$', fontsize=fs_labels)
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
    colors = ["violet", "green", "brown", "orange", "magenta", "red"]
    lw=2.5
    ms=4
    mstyle = "o"
    color_alph = 1
    shade_alph_closer = 0.13
    shade_alph_further = 0.05
    mpl.rcParams["font.size"] = 13
    mpl.rcParams["legend.fontsize"] = 13

    col_i = 0
    # col_fit = "red"
    col_fit = "black"

    uncert_col0 = Patch(facecolor=colors[col_i], alpha=shade_alph_closer+shade_alph_further)
    uncert_col0b = Patch(facecolor=colors[col_i], alpha=shade_alph_further)
    uncert_col1 = Patch(facecolor=colors[4], alpha=color_alph)
    uncert_col2 = Patch(facecolor=colors[5], alpha=color_alph)
    line_fit0 = Line2D([0,1],[0,1],linestyle=':',linewidth=lw/1.5, color=col_fit)
    
    uncert_col = Patch(facecolor=colors[col_i], alpha=0.3)
    line_rec = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/1.5, color="blue")
    line_rec_ptw_mean = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color=colors[col_i])
    line_rec_nless = Line2D([0,1],[0,1],linestyle='--',linewidth=lw/2, color=colors[5])

    if wide:
            manual_handles = [
                        line_fit0,
                        line_rec,
                        (line_rec_ptw_mean, uncert_col),
                        line_rec_nless,
                        uncert_col0, uncert_col0b,
                        uncert_col1,
                        uncert_col2,]
    else:
            manual_handles = [line_fit0, line_rec,
                        ]


    if wide:
        manual_labels = [
            r'${\frac{N^{\mathrm{5-param.}}_{\mathrm{fit}}}{N^{\mathrm{4-param.}}_{\mathrm{fit}}}}$',
            r'${\frac{N^{\mathrm{5-param.}}_{\mathrm{rec.}}}{N^{\mathrm{4-param.}}_{\mathrm{rec.}}}}$',
            r'${\frac{N^{\mathrm{5-param.}}_{\mathrm{mean}}}{N^{\mathrm{4-param.}}_{\mathrm{mean}}}}$',
            r'${\frac{N^{\mathrm{5-param.}}_{\mathrm{noiseless}}}{N^{\mathrm{4-param.}}_{\mathrm{noiseless}}}}$',
            r'${68 \% \, \mathrm{C.I.}}$',
            r'${95 \% \, \mathrm{C.I.}}$',
        ]
    else:
        manual_labels = [
            r'${\frac{N^{\mathrm{5-param.}}_{\mathrm{fit}}}{N^{\mathrm{4-param.}}_{\mathrm{fit}}}}$',
            r'${\frac{N^{\mathrm{5-param.}}_{\mathrm{rec.}}}{N^{\mathrm{4-param.}}_{\mathrm{rec.}}}}$',
        ]

    ####################
    #################### PLOTTING
    #################### 

    # Plot fit dipoles and their reconstructions
    for i, (dip_fit, dip_rec) in enumerate(zip(binned_dip_data_fit, binned_dip_data_rec)):
        ax = axs.flatten()[i]
        if PLOT_TYPE == "ratio":
            ax.plot(xvar, [1]*len(xvar),
                    label="ratio",
                    linestyle="--",
                    linewidth=lw/4.5,
                    color="black",
                    alpha=1
                    )    
            dip_rec_del = dip_rec.T[0]
            dip_rec_del[dip_rec_del < 0] = np.nan
            dip_fit2 = binned_dip_data_fit2[i]
            dip_rec2 = binned_dip_data_rec2[i]
            dip_rec_del2 = dip_rec2.T[0]
            dip_rec_del2[dip_rec_del2 < 0] = np.nan
            ax.plot(xvar, dip_rec_del2/dip_rec_del,
                    label="ratio",
                    linestyle="-",
                    linewidth=lw/1.5,
                    color="blue",
                    alpha=1
                    )
            ax.plot(xvar, (real_sigma2*dip_fit2[0])/(real_sigma*dip_fit[0]),
                    label="ratio",
                    linestyle=":",
                    linewidth=lw/1.5,
                    color=col_fit,
                    alpha=1
                    )
            rec_dip1_mean = binned_dip_data_rec_ptw_mean[i].T[0]
            rec_dip2_mean = binned_dip_data_rec_ptw_mean2[i].T[0]
            rec_dip1_noiseless = binned_dip_data_rec_noiseless[i].T[0]
            rec_dip2_noiseless = binned_dip_data_rec_noiseless2[i].T[0]
            ax.plot(xvar, rec_dip2_mean/rec_dip1_mean,
                    label="ratio",
                    linestyle="-",
                    linewidth=lw/1.5,
                    color=colors[col_i],
                    alpha=1
                    )
            ax.plot(xvar, rec_dip2_noiseless/rec_dip1_noiseless,
                    label="ratio",
                    linestyle="--",
                    linewidth=lw/1.5,
                    color="red",
                    alpha=1
                    )


    ################## SHADING        
    # Plot reconstruction uncertainties by plotting and shading between adjacent lambdas
    if wide==True:
        for i, (rec_up, rec_dn) in enumerate(zip(binned_dip_data_rec_std_up, binned_dip_data_rec_std_dn)):
            # rec_dip1 = binned_dip_data_rec[i][:,0]
            # rec_dip2 = binned_dip_data_rec2[i][:,0]
            rec_dip1 = binned_dip_data_rec_ptw_mean[i][:,0]
            rec_dip2 = binned_dip_data_rec_ptw_mean2[i][:,0]
            ratio21 = rec_dip2/rec_dip1
            ref_fit = real_sigma*binned_dip_data_fit[i][0]
            ref_fit2 = real_sigma2*binned_dip_data_fit2[i][0]
            rec_up = rec_up[0]
            rec_dn = rec_dn[0]
            rec_up2 = binned_dip_data_rec_std_up2[i][0]
            rec_dn2 = binned_dip_data_rec_std_dn2[i][0]
            rec_CI95_up = binned_dip_data_rec_CI95_up[i].T[:,0]
            rec_CI95_dn = binned_dip_data_rec_CI95_dn[i].T[:,0]
            rec_CI95_up2 = binned_dip_data_rec_CI95_up2[i].T[:,0]
            rec_CI95_dn2 = binned_dip_data_rec_CI95_dn2[i].T[:,0]
            ratio_CI_up = np.sqrt(((rec_up-rec_dip1)/rec_dip2)**2 + (rec_dip1/rec_dip2**2 *(rec_dip2-rec_dn2))**2)
            ratio_CI_dn = np.sqrt(((rec_dn-rec_dip1)/rec_dip2)**2 + (rec_dip1/rec_dip2**2 *(rec_dip2-rec_up2))**2)
            ratio_CI95_up = np.sqrt(((rec_CI95_up-rec_dip1)/rec_dip2)**2 + (rec_dip1/rec_dip2**2 *(rec_dip2-rec_CI95_dn2))**2)
            ratio_CI95_dn = np.sqrt(((rec_CI95_dn-rec_dip1)/rec_dip2)**2 + (rec_dip1/rec_dip2**2 *(rec_dip2-rec_CI95_up2))**2)
            ax = axs.flatten()[i]
            # ax.fill_between(xvar, ratio_CI_dn, ratio_CI_up, color=colors[col_i], alpha=shade_alph_closer)
            ax.fill_between(xvar, ratio21, ratio21+ratio_CI_up, color=colors[col_i], alpha=shade_alph_closer)
            ax.fill_between(xvar, ratio21-ratio_CI_dn, ratio21, color=colors[col_i], alpha=shade_alph_closer)
            ax.fill_between(xvar, ratio21, ratio21+ratio_CI95_up, color=colors[col_i], alpha=shade_alph_closer)
            ax.fill_between(xvar, ratio21-ratio_CI95_dn, ratio21, color=colors[col_i], alpha=shade_alph_closer)
            # ax.fill_between(xvar, ratio_CI95_dn, ratio_CI95_up, color=colors[col_i], alpha=shade_alph_further)


    if big_bins:
        leg = axs.flatten()[5].legend(manual_handles, manual_labels, frameon=False, fontsize=12, ncol=1, loc="lower right") 
    elif PLOT_TYPE == "ratio":
        # n_plot=0
        n_plot=2
        # leg = axs.flatten()[n_plot].legend(manual_handles, manual_labels, frameon=True, framealpha=0.4, fontsize=16, handlelength=1.2, ncol=2, columnspacing=1, loc="upper right") 
        leg = ax.legend(manual_handles, manual_labels, frameon=True, framealpha=0.4, fontsize=20, handlelength=1.2, ncol=1, columnspacing=1, loc="center left", bbox_to_anchor=(1, 0.5)) 
    
    h_align = 'left'
    if wide==True:
        h_align = 'right'
        x_crd = 0.98
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
        # plt.ylim(bottom=0.25, top=2)
        plt.ylim(bottom=0.8, top=1.2)
    else:
        plt.xlim(0.05, 25)
        plt.ylim(bottom=0.985, top=1.065)
    if big_bins:
        fig.set_size_inches(12,8)
    else:
        fig.set_size_inches(15,5.5)
    
    if not use_real_data:
        if not use_charm:
            n_plot = "plotg1-fitratio-"
        elif use_charm:
            n_plot = "plotg3-fitratio-"
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

# main(use_charm=False,real_data=False, use_log=log, big_bins=big, ratio=ratio, wide=False)
main(use_charm=False,real_data=False, use_log=log, big_bins=big, ratio=ratio, wide=True)
# main(use_charm=True,real_data=False, use_log=log, big_bins=big, ratio=ratio)

# Production plotting
# main(use_charm=False,real_data=False,fitname_i=3)
# main(use_charm=True,real_data=False,fitname_i=3)
# main(use_charm=False,real_data=False,fitname_i=4)
# main(use_charm=True,real_data=False,fitname_i=4)
# main(use_charm=False,real_data=True,fitname_i=3)
# main(use_charm=True,real_data=True,fitname_i=3)
