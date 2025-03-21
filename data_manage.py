from math import sqrt
import numpy as np

def get_data(file, simulated=True):
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
    skip_to = find_headerline(file, headerline) + maxrows + 1
    # loadtxt defaults comment lines to '#': https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html
    if simulated:
        dtype = [
        ('xbj', float),
        ('qsq', float),
        ('y', float),
        ('sigmar', float),
        ('FL', float),
        ('FT', float),
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
    print("headline not found")
    return 0


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
    for v in arr:
        if v not in counted:
            cnt = arr.count(v)
            if cnt>=min:
                print(v, cnt)
            counted.append(v)
    return counted


if __name__=="__main__":
    data_sigmar = get_data("./data/hera_II_combined_sigmar.txt", simulated=False)
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
        (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = datum
        s = round(sqrt(qsq/(y*xbj)),1)
        if s == s_bin:
            binned_data.append(datum)
        # s_vals.append(round(sqrt(s),1))
        # print(datum, "sqrt(s)= ", sqrt(s))
    
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
    binned_data = np.array(binned_data, dtype=dtype)
    # x_vals = binned_data["xbj"].tolist()
    # count_bins(x_vals, 20)
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

    # x_bin = 0.013
    x_bin = 0.002
    binned_data2_s_xbj = []
    for datum in binned_data:
        (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = datum
        if xbj==x_bin:
            binned_data2_s_xbj.append(datum)
    
    # print(binned_data2_s_xbj)
    for d in binned_data2_s_xbj:
        (qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err) = d
        print(qsq, xbj, y, sigmar, sig_err, staterruncor, tot_noproc, relative_err)