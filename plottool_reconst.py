import sys
import os
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
numbers = re.compile(r'(\d+)')
from matplotlib import rc
rc('text', usetex=True)


G_PATH = ""
STRUCT_F_TYPE = ""
PLOT_TYPE = ""
# USE_TITLE = True
USE_TITLE = False

helpstring = "usage: python plottool_reconst.py"
 


def main():
    global G_PATH, PLOT_TYPE
    f_path_list = []
    PLOT_TYPE = sys.argv[1]
    if PLOT_TYPE not in ["light", "charm", "noise"]:
        print(helpstring)
        exit(1)
    G_PATH = os.path.dirname(os.path.realpath(sys.argv[1]))

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

    xbj_bin = 0.01

    #        0        1        2        3           4
    fits = ["MV", "MVgamma", "MVe", "bayesMV4", "bayesMV5"]
    fitname = fits[4]

    ####################
    # Data filename settings
    data_path = "./reconstructions/"
    str_data = "sim"
    str_fit = fitname
    str_flavor = "lightonly"
    name_base = 'recon_out_'
    if use_charm:
        str_flavor = "lightpluscharm"
    if use_real_data:
        str_data = "hera"
        str_fit = "data_only"
    if use_noise:
        name_base = 'recon_with_noise_out_'
    composite_fname = name_base+str_data+str_fit+str_flavor

    # Reading data files
    recon_files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and \
                   composite_fname in i]
    print(recon_files)
    if not recon_files:
        print("No files found!")
        print(data_path, composite_fname, xbj_bin)
        exit(None )
    xbj_bins = [float(Path(i).stem.split("xbj")[1]) for i in recon_files]
    print(xbj_bins)

    data_list = []
    for fle in recon_files:
        data_list.append(loadmat(fle))

    # plotting
    plt.figure()
    ax = plt.gca()
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    ax.tick_params(which='major',width=1,length=6)
    ax.tick_params(which='minor',width=0.7,length=4)
    ax.tick_params(axis='both', pad=7)
    ax.tick_params(axis='both', which='both', direction="in")
    #
    # if USE_TITLE:
    #     plt.title(title)
    plt.xlabel(r'$Q^2 ~ \left(\mathrm{GeV}^{2} \right)$', fontsize=22)
    # plt.xlabel(r'$m_f$', fontsize=26)
    # plt.xlabel(r'$x_{\mathrm{Bj}}$', fontsize=26)

    # if plotting_variable == "xbj":
    #     plt.xlabel(r'$x_{\mathrm{Bj}}$', fontsize=26) #\text{\textup{GeV}}
    # else:
    #     plt.xlabel(r'$Q^2 ~ \left(\mathrm{GeV}^{2} \right)$', fontsize=18) #\text{\textup{GeV}}
    if STRUCT_F_TYPE == "FL":
        plt.ylabel(r'$F_{L}$', fontsize=22)
    elif STRUCT_F_TYPE == "FT":
        plt.ylabel(r'$F_{T}$', fontsize=22)
    elif STRUCT_F_TYPE == "F2":
        plt.ylabel(r'$F_{2}$', fontsize=22)

    # LOG AXIS
    ax.set_xscale('log')
    # ax.set_yscale('log')
    

    fit_color_set = ["orange", "red", "blue", "green", "magenta", "cyan"]
    fit_line_style = ['-', '--', '-.', ':']

    q_masses = [0, 0.0023, 0.0048, 0.095, 1.35, 4.18]


    xvar = data_list[0]["qsq"]
    f_list_nlo = []
    for data in data_list:
        fl = data["flic"] + data["fldip"] + data["flqg"]
        ft = data["ftic"] + data["ftdip"] + data["ftqg"]
        if STRUCT_F_TYPE == "FL":
            f_list_nlo.append(fl)
        # elif STRUCT_F_TYPE == "FT":
        #     f_list_nlo.append(ft/ft_massless_NLO)
    # exit() # debug
    # make labels, line styles and colors
    labels = []
    colors = []
    line_styles = []
    scalings = []
    for fname in f_path_list:
            fname = os.path.splitext(os.path.basename(fname))[0]
            if "urp" in fname:
                label = r'$\mathrm{Fit ~ 1}$'
                col = "red"
            elif "ukp" in fname:
                label = r'$\mathrm{Fit ~ 2}$'
                col = "orange"
            elif "utp" in fname:
                label = r'$\mathrm{TBK ~ p.d.}$'
                col = "black"
            elif "urs" in fname:
                label = r'$\mathrm{ResumBK ~ bs.d.}$'
                col = "magenta"
            elif "uks" in fname:
                label = r'$\mathrm{KCBK ~ bs.d.}$'
                col = "brown"
            elif "uts" in fname:
                label = r'$\mathrm{Fit ~ 3}$'
                col = "blue"
            if "cc" in fname:
                # label += r"$~ \mathrm{charm}$"
                style = "--"
                scale = 2
            elif "bb" in fname:
                # label += r"$~ \mathrm{bottom}$"
                style = ":"
                scale = 30
            elif "lpcb" in fname:
                # label += r"$~ \mathrm{incl.}$"
                style = "-"
                scale = 1
            labels.append(label)
            colors.append(col)
            line_styles.append(style)
            scalings.append(scale)

    # colors = ["red", "orange", "blue"]

    # print(labels)
    # line1 = Line2D([0,1],[0,1],linestyle='-', color='r')
    # line2 = Line2D([0,1],[0,1],linestyle='-', color='black')
    # line3 = Line2D([0,1],[0,1],linestyle='-', color='blue')
    line1 = Patch(facecolor=colors[0])
    line2 = Patch(facecolor=colors[1])
    line3 = Patch(facecolor=colors[2])
    line_incl = Line2D([0,1],[0,1],linestyle='-', color='grey')
    line_char = Line2D([0,1],[0,1],linestyle='--', color='grey')
    line_bott = Line2D([0,1],[0,1],linestyle=':', color='grey')
    manual_handles = [line1, line2, line3, line_incl, line_char, line_bott]
    manual_labels = [
        # r'$\mathrm{Fit ~ 1}$',
        # r'$\mathrm{Fit ~ 2}$',
        # r'$\mathrm{Fit ~ 3}$',
        labels[0],
        labels[1],
        labels[2],
        r'$\mathrm{inclusive}$',
        r'$\mathrm{charm}\times 2$',
        r'$\mathrm{bottom}\times 30$'
    ]

    for i, f_nlo in enumerate(f_list_nlo):
        plt.plot(xvar , scalings[i]*f_nlo,
                label=labels[i],
                linestyle=line_styles[i],
                linewidth=1.2,
                color=colors[i % 3]
                )

    # plt.text(0.95, 0.146, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black')
    # plt.text(0.95, 0.14, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black') # scaled log log
    plt.text(1.16, 0.225, r"$x_\mathrm{Bj} = 0.002$", fontsize = 14, color = 'black') # scaled linear log

    # if len(f_path_list) < 6:
    #     plt.legend(loc="best", frameon=False)
    # else:
    #     order=[1,0,2,4,3,5,7,6,8]
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], frameon=False, fontsize=14) 
    plt.legend(manual_handles, manual_labels, frameon=False, fontsize=14, ncol=2) 
    # plt.legend(handlelength=1, handleheight=1)

    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.xlim(1, 100)
    plt.ylim(bottom=0, top=0.33)
    plt.draw()
    plt.tight_layout()
    outfilename = 'plot-EIC-' + "{}".format(STRUCT_F_TYPE) + '.pdf'
    plt.savefig(os.path.join(G_PATH, outfilename))
    return 0



def get_data(file):
    # todo load matlab .mat
    data = 
    return data



main()
