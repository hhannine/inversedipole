from math import sqrt
import numpy as np

def get_data(file, simulated=True, charm=False):
    # use for manually cut down file
    skip_to = 1
    # Data parsing
    #   data starts with
    headerline="# === Computing Reduced Cross sections ===".strip()
    if "lofit" in file:
        maxrows = 177
    else:
        maxrows = 187
    maxrows=0 # This is needed then the progress NNN/187 prints are in the data file.
    skip_to = 0
    # loadtxt defaults comment lines to '#': https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
    if simulated:
        skip_to = find_headerline(file, headerline) + maxrows + 1
        dtype = [
        ('xbj', float),
        ('qsq', float),
        ('y', float),
        ('sigmar', float),
        ('FL', float),
        ('FT', float),
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
            # print("header at line: ", lineindex)
            # print(line)
            return lineindex
    print("headerline not found")
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
    charm_only = True
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
    s_bin = 318.1
    binned_data = []
    for datum in data_sigmar:
        if charm_only:
            (qsq, xbj, y, sigmar, sig_err) = datum
        else:    
            (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = datum
        s = round(sqrt(qsq/(y*xbj)),1)
        # print(s) # checking s-bins
        # if (s == s_bin) and (xbj<=1e-2):
        if charm_only:
            # charm data is only at sqrt(s)=318
            binned_data.append(datum)
        elif (s == s_bin) and (xbj<=1):
            binned_data.append(datum)
        # s_vals.append(round(sqrt(s),1))
        # print(datum, "sqrt(s)= ", sqrt(s))
    
    
    binned_data = np.array(binned_data, dtype=dtype)
    x_vals = binned_data["xbj"].tolist()
    selected_bins = count_bins(x_vals, 2)
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

    # x_bin = 0.013
    binned_data2_s_xbj_arr = []
    for x_bin in selected_bins:
        binned_data2_s_xbj = []
        for datum in binned_data:
            if charm_only:
                (qsq, xbj, y, sigmar, sig_err) = datum
            else:
                (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = datum
            if xbj==x_bin:
                binned_data2_s_xbj.append(datum)
        # print(binned_data2_s_xbj)
        binned_data2_s_xbj_arr.append(binned_data2_s_xbj)
    
    # print(binned_data2_s_xbj)
    for darr in binned_data2_s_xbj_arr:
        if charm_only:
            outf = "heraII_CC_filtered_s318.1_xbj"+str(darr[0]["xbj"])+".dat"
        else:
            outf = "heraII_filtered_s318.1_xbj"+str(darr[0]["xbj"])+".dat"
        with open(outf, 'w') as f:
            print("# qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err", file=f)
            for d in darr:
                if charm_only:
                    (qsq, xbj, y, sigmar, sig_err) = d
                    print(qsq, xbj, y, sigmar, sig_err, file=f)
                else:
                    (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = d
                    print(qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err, file=f)