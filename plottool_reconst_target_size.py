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
from matplotlib.ticker import StrMethodFormatter, ScalarFormatter
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
    # q_averages = [np.average(np.sqrt(qvals)) for qvals in Q2vals_grid]
    q_averages = [np.median(np.sqrt(qvals)) for qvals in Q2vals_grid] # Best?
    # q_averages = [np.mean(np.sqrt(qvals)) for qvals in Q2vals_grid] # Better!

    W_vals = np.array([math.sqrt((1/x)-1)*q for q, x in zip(q_averages, xbj_bins)])
    print("q_averages", q_averages)

    real_sigma = data_list[0]["real_sigma"][0]
    best_lambdas = [dat["best_lambda"][0] for dat in data_list]
    lambda_list_list = [dat["lambda"][0].tolist() for dat in data_list]
    mI_list = [lambda_list.index(best_lambda) for lambda_list, best_lambda in zip(lambda_list_list, best_lambdas)]
    best_lambdas_c = [dat["best_lambda"][0] for dat in data_list_c]
    lambda_list_list_c = [dat["lambda"][0].tolist() for dat in data_list_c]
    mI_list_c = [lambda_list_c.index(best_lambda_c) for lambda_list_c, best_lambda_c in zip(lambda_list_list_c, best_lambdas_c)]
    if lambda_type in ["semiconstrained_", "fixed_"]:
        uncert_i = [range(0, 5) for mI in mI_list]
        uncert_i_c = [range(0, 5) for mI in mI_list_c]
    else:
        ucrt_step = 2
        uncert_i = [range(mI-2*ucrt_step, mI+1+2*ucrt_step, ucrt_step) for mI in mI_list]
        uncert_i_c = [range(mI-2*ucrt_step, mI+1+2*ucrt_step, ucrt_step) for mI in mI_list_c]

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
    ### PLOT TYPE TARGET SIZE / sigma0 or B_D
    ####################
    
    fig = plt.figure()
    ax = plt.gca()
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    ax.tick_params(which='major',width=1,length=6)
    ax.tick_params(which='minor',width=0.7,length=4)
    ax.tick_params(axis='both', pad=7)
    ax.tick_params(axis='both', which='both', direction="in")


    if plotvar == "xbj":
        plt.xlabel(r'$x_{\mathrm{Bj.}}$', fontsize=22)
        plt.ylabel(r'$\frac{\sigma_0}{2} ~ \left(\mathrm{mb}\right)$', fontsize=22)
        xvar = xbj_bins
    elif plotvar == "Q":
        plt.xlabel(r'$Q ~ \left(\mathrm{GeV}\right)$', fontsize=22)
        plt.ylabel(r'$\frac{\sigma_0}{2} ~ \left(\mathrm{mb}\right)$', fontsize=22)
    elif plotvar == "xQdisks":
        plt.xlabel(r'$x_{\mathrm{Bj.}}$', fontsize=22)
        plt.ylabel(r'$Q ~ \left(\mathrm{GeV}\right)$', fontsize=22)
    elif plotvar == "W":
        plt.xlabel(r'$W ~ \left(\mathrm{GeV}\right)$', fontsize=22)
        plt.ylabel(r'$B_G ~ \left(\mathrm{GeV}^{-2}\right)$', fontsize=22)

    
    ##############
    # LABELS

    colors = ["blue", "red", "brown", "orange", "magenta", "green"]
    lw=2.8
    ms=4
    mstyle = "o"
    color_alph = 1

    line_fit0 = Line2D([0,1],[0,1],linestyle=':',linewidth=lw, color="black")

    data_lit_point1 = Line2D([0,1],[0,1],linestyle='', marker="s", markersize=10, color="black")
    data_lit_point2 = Line2D([0,1],[0,1],linestyle='', marker="^", markersize=10, color="black")
    data_lit_point3 = Line2D([0,1],[0,1],linestyle='', marker="v", markersize=10, color="black")
    data_rec_point = Line2D([0,1],[0,1],linestyle='', marker=mstyle, markersize=10, color=colors[0])
    data_rec_point_c = Line2D([0,1],[0,1],linestyle='', marker="s",markersize=10, color=colors[1])
    line_h1_bd = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color=colors[4])

    if plotvar=="xbj" or plotvar=="Q":
        manual_handles = [line_fit0,
                          data_rec_point,
                          data_rec_point_c,
                          ]
        manual_labels = [
            r'${\mathrm{Fit} ~ \frac{\sigma_0}{2}}$',
            # r'${\mathrm{Reconstruction ~ (light)} ~ \frac{\sigma_0}{2} \, \pm \, \varepsilon_\lambda}$',
            # r'${\mathrm{Reconstruction ~ (light+charm)} ~ \frac{\sigma_0}{2} \, \pm \, \varepsilon_\lambda}$',
            r'${\mathrm{Reconstruction ~ (light)} ~ \frac{\sigma_0}{2} \, \pm \, \delta_d}$',
            r'${\mathrm{Reconstruction ~ (light+charm)} ~ \frac{\sigma_0}{2} \, \pm \, \delta_d}$',
        ]
    elif plotvar=="xQdisks":
        manual_handles = [
                          data_rec_point,
                          data_lit_point1,
                          data_lit_point2,
                          data_lit_point3,
                          ]
        manual_labels = [
            r'${\mathrm{Reconstruction ~ (light+charm)} ~ \frac{\sigma_0}{2} ~ \mathrm{and ~ radius} ~ r_m}$',
            r'${\mathrm{Mass ~ radius} ~ r_m = 0.55 \, \mathrm{fm} ~ \mathrm{from ~ PhysRevD.104.054015}}$',
            # r'${\mathrm{Holographic ~ QCD ~ mass ~ radius} ~ r_m = 0.755 \, \mathrm(fm) ~ \mathrm{from ~ Nature 615, 813–816 (2023)}}$',
            # r'${\mathrm{GPD ~ mass ~ radius} ~ r_m = 0.472 \, \mathrm(fm) ~ \mathrm{from ~ Nature 615, 813–816 (2023)}}$',
            r'${\mathrm{Holographic ~ QCD ~ mass ~ radius} ~ r_m = 0.755 \, \mathrm{fm}}$',
            r'${\mathrm{GPD ~ mass ~ radius} ~ r_m = 0.472 \, \mathrm{fm}}$',
        ]
    elif plotvar=="W":
        manual_handles = [line_fit0,
                          data_rec_point, 
                          data_rec_point_c,
                          line_h1_bd,]
        manual_labels = [
            r'${B_G ~ \mathrm{from ~ HERA ~ inclusive ~ DIS ~ fit} ~ \frac{\sigma_0}{2}}$',
            r'${B_G ~ \mathrm{from ~ reconstruction~ (light)} ~ \frac{\sigma_0}{2} \, \pm \, \varepsilon_\lambda}$',
            r'${B_G ~ \mathrm{from ~ reconstruction~ (light+charm)} ~ \frac{\sigma_0}{2} \, \pm \, \varepsilon_\lambda}$',
            r"${\mathrm{H1 ~ parametrization:} ~ b_0 + 4 \alpha' \log(W/90\, GeV)}$",
        ]
    # for ibin in plt1_xbj_bins:
    #     xbj_str = str(xbj_bins[ibin])
    #     if "e" in xbj_str:
    #         # xbj_str = "0.00001" #"10^{{-5}}"
    #         xbj_str = "10^{{-5}}"
    #     manual_labels.append('$x_{{\\mathrm{{Bj.}} }} = {xbj}$'.format(xbj = xbj_str))


    ####################
    #################### PLOTTING
    #################### 

    # LOG AXIS
    ax.set_xscale('log')
    # ax.set_yscale('log')
    
    if plotvar=="xbj":
        xvar = np.array(xbj_bins)
    elif plotvar=="Q":
        xvar = np.array(q_averages)
    elif plotvar=="xQdisks":
        ax.set_yscale('log')
        xvar = np.array(xbj_bins)
        yvar = np.array(q_averages)
    elif plotvar=="W":
        xvar = W_vals
        gev_to_mb = 1 # reset back to GeV
        sigma0_to_B_d = 1/(2*math.pi) # from sigma0 (in GeV) to B_D assuming a gaussian profile for the proton
        gev_to_mb = sigma0_to_B_d

    Nlight_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_max_data)])
    Nlight_bplus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_bpluseps_max_data)])
    Nlight_bminus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_bminuseps_max_data)])
    Nlight_err_upper = Nlight_bplus_max - Nlight_max
    Nlight_err_lower = Nlight_max - Nlight_bminus_max
    Nlight_errs = np.array([gev_to_mb*Nlight_err_lower, gev_to_mb*Nlight_err_upper])

    Ncharm_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list_c, Nc_max_data)])
    Ncharm_bplus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list_c, Nc_bpluseps_max_data)])
    Ncharm_bminus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list_c, Nc_bminuseps_max_data)])
    Ncharm_err_upper = Ncharm_bplus_max - Ncharm_max
    Ncharm_err_lower = Ncharm_max - Ncharm_bminus_max
    Ncharm_errs = np.array([gev_to_mb*Ncharm_err_lower, gev_to_mb*Ncharm_err_upper])

    x_srted = sorted(xvar)
    # x_range = np.logspace(-2, -0.5, 1)
    x_range = [0.06]
    x_range_line = np.logspace(-2, -0.4, 3)
    q_range = np.linspace(9.6, 10.6, 1)
    q_range2 = np.linspace(10.2, 10.6, 1, endpoint=False)
    target_radii = np.sqrt(Ncharm_max/(10*math.pi)) # divide millibarn/10 to get square femtometers

    if plotvar=="xQdisks":
        ax.scatter(xvar, yvar, s=150*target_radii, c=Ncharm_max, cmap="plasma",)
        ax.scatter(x_range, [10.7]*len(x_range), s=[150*0.55]*len(x_range), c=[10*math.pi*0.55**2]*len(x_range), cmap="plasma", marker="s", zorder=2)
        # ax.plot(x_range, [10.7]*len(x_range), c=[10*math.pi*0.55**2]*len(x_range), cmap="plasma", linestyle="--", linewidth=5)
        ax.plot(x_range_line, [10.7]*len(x_range_line), c="gray", linestyle="--", linewidth=2, zorder=1)
        ax.scatter([0.414]*len(q_range), q_range, s=[150*0.755]*len(q_range), c=[10*math.pi*0.755**2]*len(q_range), cmap="plasma", marker="^", )
        ax.scatter([0.414]*len(q_range2), q_range2, s=[150*0.472]*len(q_range2), c=[10*math.pi*0.472**2]*len(q_range2), cmap="plasma", marker="v", )
    else:
        ax.plot(x_srted, [gev_to_mb*real_sigma]*len(x_srted),
                # label=labels[i],
                label="Fit sigma0/2",
                linestyle=":",
                linewidth=lw*1,
                # color=colors[2*i]
                color="black"
                )
        ax.plot(xvar, gev_to_mb*Nlight_max,
                label="Reconstuction of target transverse area",
                linestyle="", marker="o", markersize=5,
                color=colors[0],
                alpha=1
                )
        ax.errorbar(xvar, gev_to_mb*Nlight_max, yerr=Nlight_errs,
                    linestyle="", marker="",
                    linewidth=2.0,
                    capsize=4.,
                    capthick=1.0,
                    color=colors[0],
                    )
        ax.plot(xvar, gev_to_mb*Ncharm_max,
                label="Reconstuction of target transverse area",
                linestyle="", marker="s", markersize=5,
                color=colors[1],
                alpha=0.7
                )
        ax.errorbar(xvar, gev_to_mb*Ncharm_max, yerr=Ncharm_errs,
                    linestyle="", marker="",
                    linewidth=2.0,
                    capsize=4.,
                    capthick=1.0,
                    alpha=0.7,
                    color=colors[1],
                    )
        if plotvar=="W":
            # ax.plot(xvar, 2*math.pi*gev_to_mb*(4.15+4*0.115*np.log(xvar/90)),
            ax.plot(xvar, (4.63+4*0.164*np.log(xvar/90)),
                    label="H1 log-fit",
                    linestyle="-",
                    color=colors[4],
                    alpha=1
                    )

    
    if plotvar=="xbj":
        plt.legend(manual_handles, manual_labels, frameon=False, fontsize=16, ncol=1, loc="upper right") 
        plt.xlim(1e-4, 1e-1)
        fig.set_size_inches(8,8)
    elif plotvar=="Q":
        plt.legend(manual_handles, manual_labels, frameon=False, fontsize=16, ncol=1, loc="upper right") 
        plt.xlim(1, 25)
        fig.set_size_inches(8,8)
    elif plotvar=="xQdisks":
        plt.legend(manual_handles, manual_labels, frameon=False, fontsize=14, ncol=1, loc="upper left") 
        ax.yaxis.minorticks_on()
        ax.tick_params(which='minor',labelsize=15)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter(r'${x:,.0f}$')) # No decimal places
        ax.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
        norm = plt.Normalize(np.min(Ncharm_max), np.max(Ncharm_max))
        smap = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        cbar = fig.colorbar(smap, ax=ax, fraction=0.1, shrink = 0.9, pad=0.1)
        cbar.set_label(r"$\mathrm{{Transverse ~ area}} ~ \sigma_0/2 ~ (\mathrm{mb})$", fontsize=18)
        cbar.ax.tick_params(labelsize=15) 
        cbar2 = cbar.ax.secondary_yaxis('left',functions=(mb_to_fmrad,fmrad_to_mb))
        cbar2.set_ylabel('$\mathrm{{Radius}} ~ r_m ~ \mathrm{(fm)}$', fontsize=18)
        cbar2.tick_params(labelsize=15)
        fig.set_size_inches(11,9)
    elif plotvar=="W":
        plt.legend(manual_handles, manual_labels, frameon=False, fontsize=14, ncol=1, loc="upper left") 
        # plt.xlim(100, 190)
        plt.xlim(0.98*min(W_vals), 1.02*max(W_vals))
        # plt.xlim(3, 110) # for Q^2 = 1 average
        plt.ylim(bottom=0, top=11)
        fig.set_size_inches(8,8)
    

    if plotvar=="xbj":
        n_plot = "plot9-xbj-sigma0-"
    elif plotvar=="Q":
        n_plot = "plot9-Q-sigma0-"
    elif plotvar=="xQdisks":
        n_plot = "plot9-xQdisks-sigma0-"
    elif plotvar=="W":
        n_plot = "plot9_alt-w-B_G-"
    if not n_plot:
        print("Plot number?")
        exit()

    # write2file = False
    write2file = True
    plt.tight_layout()
    if write2file:
        mpl.use('agg') # if writing to PDF
        plt.draw()
        outfilename = n_plot + name_base+str_data+str_fit+lambda_type + "{}".format(PLOT_TYPE) + '.pdf'
        plotpath = G_PATH+"/inversedipole/plots/"
        print(os.path.join(plotpath, outfilename))
        plt.savefig(os.path.join(plotpath, outfilename))
    else:
        plt.show()
    fig.clear()
    plt.close()
    return 0

def mb_to_fmrad(x):
    return np.sqrt(x/(10*math.pi))

def fmrad_to_mb(x):
    return x**2*(10*math.pi)


# plotvar xbj / W
# main("xbj")
# main("Q")
# main("xQdisks")
main("W")
