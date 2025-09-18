from math import sqrt
import numpy as np

def get_data(file, simulated=None, charm=False):
    # use for manually cut down file
    # skip_to = 1
    # Data parsing
    #   data starts with
    headerline="=== Computing Reduced Cross sections ===".strip()
    if "lofit" in file:
        maxrows = 177
    else:
        maxrows = 187
    maxrows=0 # This is needed then the progress NNN/187 prints are in the data file.
    # skip_to = 0
    # loadtxt defaults comment lines to '#': https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
    if simulated=="flft-contrib":
        skip_to = find_headerline(file, headerline) + maxrows + 1
        dtype = [
        ('xbj', float),
        ('qsq', float),
        ('y', float),
        ('sigmar', float),
        ('FL', float),
        ('FT', float),
        ]
    elif simulated=="reference-dipole-inverted":
        skip_to = find_headerline(file, headerline) + 1
        print("skipto", skip_to)
        dtype = [
        ('xbj', float),
        ('qsq', float),
        ('y', float),
        ('sigmar', float),
        ('sigmarerr', float),
        ('theory', float),
        ]
    elif simulated=="reference-dipole":
        skip_to = find_headerline(file, headerline) + 1
        # print("skipto", skip_to)
        dtype = [
        ('qsq', float),
        ('xbj', float),
        ('y', float),
        ('sigmar', float),
        ('sigmarerr', float),
        ('theory', float),
        ]
    elif not simulated and charm:
        dtype = [
        ('qsq', float),
        ('xbj', float),
        ('y', float),
        ('sigmar', float),
        ('sigmarerr', float),
        ]
    else:
        dtype = [
        ('qsq', float),
        ('xbj', float),
        ('y', float),
        ('sigmar', float),
        ('sigmarerr', float),
        # ('theory', float),
        ('StatErrUncor', float),
        ('tot_noproc', float),
        ('%', float),
        ]
    # data = np.loadtxt(file, dtype=dtype, skiprows=skip_to)
    data = np.loadtxt(file, dtype=dtype, skiprows=skip_to)
    # data = np.genfromtxt(file, skip_header=skip_to, skip_footer=0)
    return data

def find_headerline(file, headerline):
    # https://stackoverflow.com/questions/3961265/get-line-number-of-certain-phrase-in-file-python
    fopen = open(file, "r")
    for num, line in enumerate(fopen, 1):
        if headerline in line:
            lineindex = num
            print("header at line: ", lineindex)
            print(line)
            return lineindex
    # print("headerline not found")
    return 0

def read_sigma02(file):
    file = open(file, 'r')
    for line in file.readlines():
        if line.startswith("# qs0sqr="):
            parts = line.split(",")
            for part in parts:
                words = part.split("=")
                if "sigma02" in words[0]:
                    return float(words[1])
    print("DID NOT FIND MATCHING SIGMA02. ABORT.")
    return -666

def load_dipole(file):
    skip_to = 5
    dtype = [
        ('xbj', float),
        ('r', float),
        ('S', float),
    ]
    data = np.loadtxt(file, dtype=dtype, skiprows=skip_to)
    return data

def count_bins(arr, min=0):
    counted = []
    count_exeeds_min = []
    for v in arr:
        if v not in counted:
            cnt = arr.count(v)
            if cnt>=min:
                print(v, cnt)
                count_exeeds_min.append(v)
            counted.append(v)
    return count_exeeds_min


if __name__=="__main__":
    charm_only = False
    # charm_only = True
    use_reference_dipoles = True
    if use_reference_dipoles:
        if not charm_only:
            # data_sigmar = get_data("./data/paper2/unbinned/paper2_hera-referencedipole-lo-sigmar_bayesMV4-strict_Q_cuts.dat", simulated="reference-dipole", charm=charm_only)
            # fitname="bayesMV4-strict_Q_cuts"
            # data_sigmar = get_data("./data/paper2/unbinned/paper2_hera-referencedipole-lo-sigmar_bayesMV4-wide_Q_cuts.dat", simulated="reference-dipole", charm=charm_only)
            # fitname="bayesMV4-wide_Q_cuts"
            # data_sigmar = get_data("./data/paper2/unbinned/paper2_hera-referencedipole-lo-sigmar_bayesMV5-strict_Q_cuts.dat", simulated="reference-dipole", charm=charm_only)
            # fitname="bayesMV5-strict_Q_cuts"
            data_sigmar = get_data("./data/paper2/unbinned/paper2_hera-referencedipole-lo-sigmar_bayesMV5-wide_Q_cuts.dat", simulated="reference-dipole", charm=charm_only)
            fitname="bayesMV5-wide_Q_cuts"
            dtype = [
            ('xbj', float),
            ('qsq', float),
            ('y', float),
            ('sigmar', float),
            ('sigmarerr', float),
            ('theory', float),
            ]
        elif charm_only:
            print("charm only reference dipole prediction data not generated! Exit.")
            exit()
            data_sigmar = get_data("./data/", simulated="reference-dipole", charm=charm_only)
            dtype = [
            ('xbj', float),
            ('qsq', float),
            ('y', float),
            ('sigmar', float),
            ('sigmarerr', float),
            ('theory', float),
            ]
    else:
        if not charm_only:
            data_sigmar = get_data("./data/hera_II_combined_sigmar.txt", simulated=False)
            dtype = [
            ('qsq', float),
            ('xbj', float),
            ('y', float),
            ('sigmar', float),
            ('sigmarerr', float),
            # ('theory', float),
            ('StatErrUncor', float),
            ('tot_noproc', float),
            ('%', float),
            ]
        elif charm_only:
            data_sigmar = get_data("./data/hera_II_combined_sigmar_cc.txt", simulated=False, charm=charm_only)
            dtype = [
            ('qsq', float),
            ('xbj', float),
            ('y', float),
            ('sigmar', float),
            ('sigmarerr', float),
            # ('theory', float),
            # ('StatErrUncor', float),
            # ('tot_noproc', float),
            # ('%', float),
            ]
    qsq_vals = data_sigmar["qsq"]
    y_vals = data_sigmar["y"]
    s_vals = []
    # sqrt(s)   N points
    # 318.1     644
    # 300.3     112
    # 251.5     260
    # 224.9     210
    # s_bin = 318.1
    if not charm_only:
        for datum in data_sigmar:
            qsq = datum["qsq"]
            y = datum["y"]
            xbj = datum["xbj"]
            sqrt_s = round(sqrt(qsq/(y*xbj)),1)
            s_vals.append(sqrt_s)
        s_bins = count_bins(s_vals)
    else:
        s_bins = [318] # charm data is rounded

    binned_data_for_all_sqrts = []
    for s_bin in s_bins:
        binned_data = []
        for datum in data_sigmar:
            qsq = datum["qsq"]
            y = datum["y"]
            xbj = datum["xbj"]
            sqrt_s = round(sqrt(qsq/(y*xbj)),1)
            # print(s) # checking s-bins
            if (sqrt_s == s_bin) and (xbj<=1):
                binned_data.append(datum)
        binned_data_for_all_sqrts.append((s_bin, binned_data))


    for s_bin, binned_data in binned_data_for_all_sqrts:
        binned_data = np.array(binned_data, dtype=dtype)
        x_vals = binned_data["xbj"].tolist()
        selected_bins = count_bins(x_vals, 5)
        print(selected_bins)
        # xbj    N
        # 0.002  21
        # 0.0032 24
        # 0.005  24
        # 0.008  23
        # 0.013  30
        # 0.02   35
        # 0.032  33
        # 0.05   30
        # 0.08   30
        # 0.13   31
        # 0.18   33
        # 0.25   31
        # 0.4    33

        # xbj    Npoints  -- limiting to xbj <= 1e-2
        # 0.0008 19
        # 0.0013 18
        # 0.002  21
        # 0.0032 24
        # 0.005  24
        # 0.008  23
        # 0.00013 8
        # 0.0005 15
        # 0.0002  9
        # 0.00032 11

        # no limits, all bins with >= 6 points
        # 0.0008 19
        # 0.0013 18
        # 0.002 21
        # 0.0032 24
        # 0.005 24
        # 0.008 23
        # 0.013 30
        # 0.02 35
        # 0.032 33
        # 0.05 30
        # 0.08 30
        # 0.13 31
        # 0.18 33
        # 0.25 31
        # 0.4 33
        # 0.65 23
        # 0.00013 8
        # 0.0005 15
        # 0.0002 9
        # 0.00032 11
        # [0.0008, 0.0013, 0.002, 0.0032, 0.005, 0.008, 0.013, 0.02, 0.032, 0.05, 0.08, 0.13, 0.18, 0.25, 0.4, 0.65, 0.00013, 0.0005, 0.0002, 0.00032]

        binned_data2_s_xbj_arr = []
        for x_bin in selected_bins:
            binned_data2_s_xbj = []
            for datum in binned_data:
                xbj = datum["xbj"]
                if xbj==x_bin:
                    binned_data2_s_xbj.append(datum)
            # print(binned_data2_s_xbj)
            binned_data2_s_xbj_arr.append(binned_data2_s_xbj)
        
        # print(binned_data2_s_xbj)
        for darr in binned_data2_s_xbj_arr:
            if use_reference_dipoles:
                outf = "heraII_reference_dipoles_filtered_"+fitname+"_s"+str(s_bin)+"_xbj"+str(darr[0]["xbj"])+".dat"
            else:
                if charm_only:
                    outf = "heraII_CC_filtered_s"+str(s_bin)+"_xbj"+str(darr[0]["xbj"])+".dat"
                else:
                    outf = "heraII_filtered_s"+str(s_bin)+"_xbj"+str(darr[0]["xbj"])+".dat"
            with open(outf, 'w') as f:
                print("# qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err", file=f)
                for d in darr:
                    if use_reference_dipoles:
                        (xbj, qsq, y, sigmar, sig_err, theory) = d
                        print(qsq, xbj, y, sigmar, sig_err, theory, file=f)
                    else:
                        if charm_only:
                            (qsq, xbj, y, sigmar, sig_err) = d
                            print(qsq, xbj, y, sigmar, sig_err, file=f)
                        else:
                            (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = d
                            print(qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err, file=f)
