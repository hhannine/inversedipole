#!/usr/bin/env python3

import sys
from scipy.io import loadmat


def main():
    try:
        fname = sys.argv[1]
    except:
        print("No file given. Exit. Arugments are: filename data_variable[opt]")
        exit()
    if len(sys.argv) > 2:
        dat = loadmat(fname)[sys.argv[2]]
    else:
        dat = loadmat(fname)
    print(dat)

if __name__ == "__main__":
    main()
