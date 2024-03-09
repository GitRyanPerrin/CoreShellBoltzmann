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
import hamiltonian as ham
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
    
    cdef float RAM_needed = round(8*Nk/1e9*(Nstat**2 + Nstat*Nev + Nev) + 32.0)

    cdef int dumps = ceil(RAM_needed/RAM_avail)

    return dumps    

def calc_dispersion(Nev, alphaR=ip.alphaR, betaD=ip.betaD, B=ip.B, E=ip.E, save_file=False, Nk=2000, k=None, eigvals_only=False):

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
        H_fut = pool.map(ham.Hmat, k, E, repeat(B), repeat(alphaR), repeat(betaD))

    H_list = [fut for fut in H_fut]
    del H_fut
    collect()

    if eigvals_only==False:
        for i in range(Nk):
            eigvals[i], eigvecs[i] = la.eigh(H_list[i], lower=False, subset_by_index=[0,Nev-1])
    else:
        for i in range(Nk):
            eigvals[i] = la.eigh(H_list[i], lower=False, subset_by_index=[0,Nev-1], eigvals_only=True)

    if save_file==True:
        if eigvals_only==False:
            np.savez_compressed(
                f"./eigensystems/{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.alphaR}_{ip.betaD}.npz",
                k,
                eigvals,
                eigvecs
            )
        else:
            np.savez_compressed(
                f"./eigensystems/{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.alphaR}_{ip.betaD}.npz",
                k,
                eigvals,
                eigvecs
            )
    else:
        if eigvals_only==False:
            return eigvals, eigvecs
        else:
            return eigvals

def split_dispersion_calculation(Nev, k, Nk):

    dumps = get_RAM_dumps(Nev=Nev, Nk=Nk)
    if dumps == 0: dumps = 1

    split_k = np.array_split(k, dumps)

    for d in range(dumps):
        Nk = len(split_k[d])
        k, eigvals, eigvecs = calc_dispersion(Nev=Nev, save_file=False, Nk=Nk, k=split_k[d])

        np.savez_compressed(f"./eigensystems/{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.alphaR}_{ip.betaD}_({d}).npz", 
            k, 
            eigvals, 
            eigvecs
        )
