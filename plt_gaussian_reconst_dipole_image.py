import sys
import os
import re
from pathlib import Path

import math
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy.io import loadmat, matlab

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, ScalarFormatter, NullFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
    # composite_fname_c = name_base+str_data+str_fit+str_flavor_c+lambda_type
    print(composite_fname)

    # Reading data files
    recon_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname in i]
    # recon_files_c = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and composite_fname_c in i]
    print(recon_files)
    # print(recon_files_c)
    if not recon_files:
        print("No files found!")
        print(data_path, composite_fname)
        exit(None)
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in recon_files]
    recon_files = [x for _, x in sorted(zip(xbj_bins, recon_files))]
    # recon_files_c = [x for _, x in sorted(zip(xbj_bins, recon_files_c))]
    xbj_bins = sorted(xbj_bins)
    print(xbj_bins)

    data_list = [loadmat(data_path + fle) for fle in recon_files]
    # data_list_c = [loadmat(data_path + fle) for fle in recon_files_c]
    
    #######################
    # Reading data
    Q2vals_grid = [dat["q2vals"][0] for dat in data_list]
    q_averages = [np.median(np.sqrt(qvals)) for qvals in Q2vals_grid] # Best?

    R = data_list[0]["r_grid"][0]
    # print(R)
    XBJ = np.array(xbj_bins)
    x_fit_out = XBJ[XBJ>1e-2]
    rr, xx = np.meshgrid(R,XBJ)
    rrfit, xxfitout = np.meshgrid(R,x_fit_out)

    # real_sigma = data_list[0]["real_sigma"][0] # not needed, N_fit has it
    
    N_max_data = np.array([dat["N_max_data_strict"][0] for dat in data_list]) # this is used for the colorization scaling
    N_max_data = N_max_data[:,0]
    sig_max = max(N_max_data)
    print(sig_max)
    # print(N_max_data[:,0])
    # max_N_max = max(N_max_data)



    ####################
    ### PLOT TYPE DIPOLE IMAGE
    ####################

    fig, axs = plt.subplots(nrows=1, ncols=2)
    # fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(12, 5)
    # fig.set_size_inches(20, 7)
    plt.tight_layout()
    plt.subplots_adjust(top = 0.945, bottom = 0.13, right = 1.05, left = 0.07, hspace = 0, wspace = 0.03)

    titles = [r"$\mathrm{Fit ~ parametrization ~ dipole}$",
            #   r"$\mathrm{HERA ~ reconstruction, ~ standard ~ scheme ~ u,d,s,c,b}$",
              r"$\mathrm{Reconstruction ~ from ~ HERA ~ DIS ~ data}$",
            #   r"$\mathrm{Reconstructed ~ dipole, ~ light + charm}$",
              "",
              ]
    if plotvar == "r":
        xvar = xbj_bins
        for i, ax1 in enumerate(axs):
            # ax1.xticks(fontsize=20, rotation=0)
            # ax1.yticks(fontsize=20, rotation=0)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.grid(False)
            ax1.tick_params(which='major',width=1,length=6,labelsize=14)
            ax1.tick_params(which='minor',width=0.7,length=4,labelsize=14)
            ax1.tick_params(axis='both', pad=7)
            ax1.tick_params(axis='both', which='both', direction="in")
            # ax1.set_xlim([1e-2, max(R)])
            ax1.set_xlim([min(R), max(R)])
            # ax1.set_ylim([min(XBJ), max(XBJ)])
            # ax1.set_title(titles[i], fontsize=20, pad=10)
            ax1.set_title(titles[i], fontsize=16, pad=4)
            ax1.set_xlabel(r'$r ~ \left( \mathrm{GeV}^{-1} \right)$', fontsize=18)
            if i==0:
                ax1.set_ylabel(r'$x_{\mathrm{Bj.}}$', fontsize=26)
            elif i!=0:
                ax1.set_yticks([])



    ####################
    #################### PLOTTING
    #################### 

    
    if plotvar=="r":
        xvar = np.array(R)
    elif plotvar=="Q":
        xvar = np.array(q_averages)

    dip_data_rec = np.array([dat["N_reconst"] for dat in data_list]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    # for dat in data_list:
    # dip_data_rec_c = np.array([dat["N_reconst"] for dat in data_list_c]) # data_list is indexed the same as xbj_bins, each N_rec is indexed in r_grid
    # dip_data_fit = np.array([dat["N_fit"] for dat in data_list])
    dip_data_fit = []
    for xbj, dat in zip(xbj_bins, data_list):
        if xbj < 0.01:
            dip_data_fit.append(dat["N_fit"][:,0])
        else:
            a = np.array(dat["r_steps"][0][0] * [0])
            dip_data_fit.append(a)
    dip_data_fit = np.array(dip_data_fit)

    real_sigma = max(dip_data_fit[0])
    print(real_sigma)
    reshape_fit = dip_data_fit.reshape((len(XBJ), len(R)))
    reshape_dip = dip_data_rec.reshape((len(XBJ), len(R)))
    # reshape_dip_c = dip_data_rec_c.reshape((len(XBJ), len(R)))

    # mapname = 'viridis'
    # mapname = 'magma'
    # mapname = 'copper'
    # mapname = 'gist_heat'
    # mapname = 'bone'
    mapname = 'coolwarm'
    # mapname = 'RdBu'
    cmap = plt.colormaps[mapname]
    
    norm = plt.Normalize(np.min(N_max_data), np.max(N_max_data))
    # norm = mpl.colors.LogNorm(np.min(Ncharm_max), np.max(Ncharm_max))
    # norm = mpl.colors.LogNorm(np.min(Ncharm_max), np.max(Ncharm_max))
    smap = plt.cm.ScalarMappable(cmap=mapname, norm=norm)
    log_scale_col = True
    log_scale=400
    sig_max *= gev_to_mb
    lognorm = mpl.colors.LogNorm(sig_max/log_scale, sig_max)

    # print(np.min(Ncharm_max)/10, np.max(Ncharm_max))
    plot_rad_line = False

    target_radii = np.sqrt(N_max_data/(10*math.pi)) # divide millibarn/10 to get square femtometers
    # target_radii_c = np.sqrt(Ncharm_max/(10*math.pi)) # divide millibarn/10 to get square femtometers
    fm_to_gev = 5.068
    radii_in_gev = target_radii*fm_to_gev
    # radii_in_gev_c = target_radii_c*fm_to_gev
    sigma0_to_rad_fm = np.sqrt(real_sigma*gev_to_mb/(10*math.pi))*fm_to_gev

    max_adj_mult = 1
    # clim=[0,sig_max]
    ax = axs[0]
    ax.grid(False, which="both")
    # cfit = ax.pcolormesh(rr, xx, real_sigma*reshape_fit, vmin=0, vmax=max(N_max_data)*max_adj_mult, cmap = cmap) # new export has real_sigma, no need to multiply it anymore
    # cfit = ax.pcolormesh(rr, xx, gev_to_mb*reshape_fit, vmin=0, vmax=max(N_max_data)*max_adj_mult, cmap = cmap)
    if log_scale_col:
        cfit = ax.pcolormesh(rr, xx, gev_to_mb*reshape_fit, cmap = cmap, norm = lognorm,linewidth=0,rasterized=True)
        cfit.set_edgecolor('face')
    cfitbox = ax.pcolormesh(rrfit, xxfitout, np.zeros((len(x_fit_out), len(R))), vmin=0, vmax=max(N_max_data)*max_adj_mult, cmap = "inferno", linewidth=0,rasterized=True) 
    cfitbox.set_edgecolor('face')
    # cfit.norm.autoscale(clim)
    if plot_rad_line:
        ax.plot([sigma0_to_rad_fm]*len(xbj_bins), xbj_bins, marker="o", markersize=3.5, c="white")
    ax = axs[1]
    ax.grid(False, which="both")
    if log_scale_col:
        missing_data_bg = ax.pcolormesh(rr, xx, np.zeros((len(xbj_bins), len(R))), vmin=0, vmax=max(N_max_data)*max_adj_mult, cmap = "inferno", linewidth=0,rasterized=True)
        missing_data_bg.set_edgecolor('face')
    # c = ax.pcolormesh(rr, xx, reshape_dip, vmin=0, vmax=max(N_max_data)*max_adj_mult, cmap = cmap) 
    c = ax.pcolormesh(rr, xx, gev_to_mb*reshape_dip, cmap = cmap, norm = lognorm,linewidth=0,rasterized=True)
    c.set_edgecolor('face')
    # c.norm.autoscale(clim)
    if plot_rad_line:
        ax.plot(radii_in_gev, xbj_bins, marker="o", markersize=3.5, c="white")
    # ax = axs[2] 
    # cc = ax.pcolormesh(rr, xx, reshape_dip_c, vmin=0, vmax=max(Ncharm_max)*max_adj_mult, cmap = cmap) 
    # cc.norm.autoscale(clim)
    # ax.plot(radii_in_gev_c, xbj_bins, marker="s", c="white")
    # cc = ax.pcolormesh(rr, xx, reshape_dip_c+0.9, cmap = cmap, norm = mpl.colors.LogNorm(np.min(Ncharm_max)/10, np.max(Ncharm_max))) 
    
    cbar=fig.colorbar(c, ax=axs.ravel().tolist(), shrink=1, pad=0.01)
    cbar.set_label(r'$\frac{\sigma_0}{2} N(r) ~ \left(\mathrm{mb}\right)$', fontsize=22)
    cbar.ax.tick_params(labelsize=15) 


    data_rec_point_c = Line2D([0,1],[0,1],linestyle='-', marker="", color="white")
    noise_fail = Patch(color="Black")  #Patch(facecolor=colors[col_i], alpha=0.3)

    if plot_rad_line:
        manual_handles = [
                            data_rec_point_c,
                            noise_fail
                            ]
        manual_labels = [
            r'${r = r_g}$',
            r'$\mathrm{Rec. ~ noise}$',
        ]
    else:
        manual_handles = [
                            noise_fail
                            ]
        manual_labels = [
            r'$\mathrm{Rec. ~ noise ~/~ exclusion ~ from ~ fit}$',
        ]
    leg = axs[0].legend(manual_handles, manual_labels, frameon=True, fontsize=20, ncol=1, loc="upper left") 
    for txt in leg.get_texts():
        # txt.set_color("white")
        txt.set_color("black")
    
    prelim = True
    if prelim:
        x_crd = 0.04
        y_crd = 0.97
        prelim_lbl = r"$\mathrm{Preliminary}$"
        axs.flatten()[1].text(x_crd, y_crd, prelim_lbl,
                horizontalalignment="left",
                verticalalignment='top',
                fontsize=24,
                color="red",
                transform=axs.flatten()[i].transAxes)

    if plotvar=="r":
        n_plot = "plot10-r-dipoleimage-"
    if not n_plot:
        print("Plot number?")
        exit()

    write2file = False
    write2file = True
    if write2file:
        mpl.use('agg') # if writing to PDF
        plt.draw()
        outfilename = n_plot + name_base+str_data+str_fit+lambda_type + "{}".format(PLOT_TYPE) + '.pdf'
        # outfilename = n_plot + composite_fname + "{}".format(PLOT_TYPE) + '.png'
        plotpath = G_PATH+"/inversedipole/plots_hera_uq/"
        print(os.path.join(plotpath, outfilename))
        plt.savefig(os.path.join(plotpath, outfilename),dpi=300)
    else:
        plt.margins(0,0)
        plt.show()
    fig.clear()
    plt.close()
    return 0

def mb_to_fmrad(x):
    return np.sqrt(x/(10*math.pi))

def fmrad_to_mb(x):
    return x**2*(10*math.pi)


# plotvar r
main("r")
