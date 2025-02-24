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
