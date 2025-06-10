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
    name_base = 'recon_out_'
    if use_charm:
        str_flavor = "lightpluscharm_"
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
    print(composite_fname)

    # Reading data files
    recon_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
                   composite_fname in i]
    print(recon_files)
    if not recon_files:
        print("No files found!")
        print(data_path, composite_fname)
        exit(None)
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in recon_files]
    recon_files = [x for _, x in sorted(zip(xbj_bins, recon_files))]
    xbj_bins = sorted(xbj_bins)
    print(xbj_bins)

    data_list = []
    for fle in recon_files:
        data_list.append(loadmat(data_path + fle))
    
    #######################
    # Reading data
    Q2vals_grid = [dat["q2vals"][0] for dat in data_list]
    # q_averages = [np.average(np.sqrt(qvals)) for qvals in Q2vals_grid]
    q_averages = [np.median(np.sqrt(qvals)) for qvals in Q2vals_grid] # Best?
    # q_averages = [np.mean(np.sqrt(qvals)) for qvals in Q2vals_grid] # Better!
    # W2_vals = np.array([1/x for q2, x in zip(q2_averages, xbj_bins)])
    W_vals = np.array([math.sqrt((1/x)-1)*q for q, x in zip(q_averages, xbj_bins)])
    # W_vals = np.array([math.sqrt((1/x)-1)*2 for q, x in zip(q_averages, xbj_bins)])
    print("q_averages", q_averages)

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
    elif plotvar == "W":
        plt.xlabel(r'$W ~ \left(\mathrm{GeV}\right)$', fontsize=22)
        plt.ylabel(r'$B_G ~ \left(\mathrm{GeV}^{-2}\right)$', fontsize=22)

    
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

    data_rec_point = Line2D([0,1],[0,1],linestyle='', marker=mstyle, color="blue")
    line_h1_bd = Line2D([0,1],[0,1],linestyle='-',linewidth=lw/2, color="red")

    if plotvar=="xbj":
        manual_handles = [line_fit0,
                          data_rec_point,
                          ]
        manual_labels = [
            r'${\mathrm{Fit} ~ \frac{\sigma_0}{2}}$',
            r'${\mathrm{Reconstruction} ~ \frac{\sigma_0}{2} \, \pm \, \varepsilon_\lambda}$',
        ]
    elif plotvar=="W":
        manual_handles = [line_fit0,
                          data_rec_point, 
                          line_h1_bd,]
        manual_labels = [
            r'${B_G ~ \mathrm{from ~ HERA ~ inclusive ~ DIS ~ fit} ~ \frac{\sigma_0}{2}}$',
            r'${B_G ~ \mathrm{from ~ reconstruction} ~ \frac{\sigma_0}{2} \, \pm \, \varepsilon_\lambda}$',
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
        # xvar = xbj_bins
        xvar = np.array(xbj_bins)
    elif plotvar=="W":
        # xvar = np.sqrt(W2_vals)
        xvar = W_vals
        gev_to_mb = 1 # reset back to GeV
        sigma0_to_B_d = 1/(2*math.pi) # from sigma0 (in GeV) to B_D assuming a gaussian profile for the proton
        gev_to_mb = sigma0_to_B_d

    # Plot sigma0(xbj) for light only, light plus charm, and fit constant
    # W^2 = Q^2 / xbj
    # W2_vals = np.array([q2/(x) for q2, x in zip(q2_averages, xbj_bins)])

    Nlight_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_max_data)])
    Nlight_bplus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_bpluseps_max_data)])
    Nlight_bminus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_bminuseps_max_data)])
    Nlight_err_upper = Nlight_bplus_max - Nlight_max
    Nlight_err_lower = Nlight_max - Nlight_bminus_max
    Nlight_errs = np.array([gev_to_mb*Nlight_err_lower, gev_to_mb*Nlight_err_upper])

    Ncharm_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_max_data)])
    Ncharm_bplus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_bpluseps_max_data)])
    Ncharm_bminus_max = np.array([max_vals[mI] for mI, max_vals in zip(mI_list, N_bminuseps_max_data)])
    Ncharm_err_upper = Ncharm_bplus_max - Ncharm_max
    Ncharm_err_lower = Ncharm_max - Ncharm_bminus_max
    Ncharm_errs = np.array([gev_to_mb*Ncharm_err_lower, gev_to_mb*Ncharm_err_upper])

    x_srted = sorted(xvar)
    # x_srted, b_rec = zip(*sorted(zip(xvar, b_rec)))

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
            linestyle="", marker="o", markersize=7,
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
    if plotvar=="W":
        # ax.plot(xvar, 2*math.pi*gev_to_mb*(4.15+4*0.115*np.log(xvar/90)),
        ax.plot(xvar, (4.63+4*0.164*np.log(xvar/90)),
                label="H1 log-fit",
                linestyle="-",
                color="red",
                alpha=1
                )

    # lambda uncertainty for the maxima: (probably not useful since it's often one sided?)
    for i, (mI, max_vals) in enumerate(zip(mI_list, N_max_data)):
        Nmax = Nlight_max[i]
        if mI==0:
            err_tuple = ([0], [gev_to_mb*abs(max_vals[mI+1]-Nmax)])
        elif mI==4:
            err_tuple = ([gev_to_mb*abs(Nmax-max_vals[mI-1])], [0])
        else:
            err_tuple = [[gev_to_mb*abs(Nmax-max_vals[mI-1])], [gev_to_mb*abs(max_vals[mI+1]-Nmax)]]
        ax.errorbar(xvar[i], gev_to_mb*Nmax, yerr=err_tuple,
                    linestyle="", marker="",
                    linewidth=2.0,
                    capsize=3.0,
                    capthick=1.0,
                    color=colors[3],
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


    # plt.text(0.95, 0.146, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black')
    # plt.text(0.95, 0.14, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black') # scaled log log
    # plt.text(1.16, 0.225, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black') # scaled linear log
    
    
    # ax.xaxis.set_major_formatter(ScalarFormatter())
    
    if plotvar=="xbj":
        plt.legend(manual_handles, manual_labels, frameon=False, fontsize=16, ncol=1, loc="upper right") 
        plt.xlim(1e-4, 1e-1)
    elif plotvar=="W":
        plt.legend(manual_handles, manual_labels, frameon=False, fontsize=14, ncol=1, loc="upper left") 
        # plt.xlim(100, 190)
        plt.xlim(0.98*min(W_vals), 1.02*max(W_vals))
        # plt.xlim(3, 110) # for Q^2 = 1 average
        plt.ylim(bottom=0, top=10)
    fig.set_size_inches(7,7)
    

    if plotvar=="xbj":
        n_plot = "plot9-xbj-sigma0-"
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
        outfilename = n_plot + composite_fname + "{}".format(PLOT_TYPE) + '.pdf'
        plotpath = G_PATH+"/inversedipole/plots/"
        print(os.path.join(plotpath, outfilename))
        plt.savefig(os.path.join(plotpath, outfilename))
    else:
        plt.show()
    fig.clear()
    plt.close()
    return 0

# plotvar xbj / W
main("xbj")
main("W")
