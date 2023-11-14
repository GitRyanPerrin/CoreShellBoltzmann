from setuptools import setup, Extension, dist
import sys
from gc import collect
import re
import time
from itertools import repeat
from psutil import virtual_memory
from math import ceil

import numpy as np
cimport numpy as np

import lattice as lat
#import hamiltonian as ham
from spin_velocity_matrix import Vmat
import input_data as ip

from matplotlib import pyplot as plt
import scipy.linalg as la
from scipy.sparse.linalg import eigsh
from scipy import sparse as sp
from scipy.interpolate import interp1d
import concurrent.futures

def get_RAM_dumps(Nev, Nk):

    cdef int vertices = ip.vertices
    cdef int Nr = ip.Nr
    cdef int Nphi = ip.Nphi
    cdef float R = ip.R
    cdef float t = ip.t

    cdef np.ndarray pos
    cdef int Nstat

    cdef float RAM_avail = round(virtual_memory().total/1e9)

    if vertices == 0:
        pos, Nstat = lat.nanotube.circular(Nr, Nphi, R, t)
    else:
        pos, Nstat = lat.prismatic_nanotube(vertices)
    del pos
    collect()
    
    cdef float RAM_needed = round(8*Nk/1e9*Nstat**2 + 16.0)

    cdef int dumps = ceil(RAM_needed/RAM_avail)

    return dumps    

def calc_velocity_array(Nev, alphaR=ip.alphaR, betaD=ip.betaD, B=ip.B, save_file=False, Nk=2000, k=None):

    cdef int Nr = ip.Nr
    cdef int Nphi = ip.Nphi
    cdef int Nstat
    cdef double Rext = ip.Rext
    cdef double R = ip.R
    cdef double t = ip.t
    cdef int vertices = ip.vertices


    cdef np.ndarray eigvals = np.zeros(Nk, dtype = object)
    cdef np.ndarray eigvecs = np.zeros(Nk, dtype = object)

    cdef int max_workers = 20
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        V_fut = pool.map(Vmat, k)

    v = np.stack([fut for fut in V_fut])
    del V_fut
    collect()

    if save_file == True:
        np.savez_compressed(f"./velocities/spin_velocities/cv_{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.alphaR}_{ip.betaD}.npz", v)
    else:
        return v

def split_velocity_calculation(Nev, k, Nk):

    dumps = get_RAM_dumps(Nev=Nev, Nk=Nk)
    if dumps == 0: dumps = 1

    split_k = np.array_split(k, dumps)

    for d in range(dumps):
        Nk = len(split_k[d])
        v = calc_velocity_array(Nev=Nev, save_file=False, Nk=Nk, k=split_k[d])

        np.savez_compressed(f"./velocities/spin_velocities/cv_{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.alphaR}_{ip.betaD}_({d}).npz", v)
