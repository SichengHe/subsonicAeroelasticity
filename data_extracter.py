import numpy
from scipy.fftpack import fft, fftfreq, fftshift, ifft
from mpi4py import MPI
import matplotlib.pyplot as plt


def fetch_data(filename,line_begin,line_end,col):

    data = []

    f = open(filename,'r')

    i = 0
    for line in f:
        i += 1
        if (i>=line_begin and i<=line_end):
            split_line = line.split()
            data.append(float(split_line[col]))

    return data

# serial version
