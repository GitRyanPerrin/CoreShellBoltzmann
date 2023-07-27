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

def get_RAM_dumps(Nsing, Nk):

    cdef int vertices = ip.vertices
    cdef int Nr = ip.Nr
    cdef int Nphi = ip.Nphi
    cdef float R = ip.R
    cdef float t = ip.t

    cdef np.ndarray pos
    cdef int Nstat

    cdef float RAM_avail = round(virtual_memory().total/1e9)
    #print(RAM_avail)

    if vertices == 0:
        pos, Nstat = lat.nanotube.circular(Nr, Nphi, R, t)
    else:
        pos, Nstat = lat.prismatic_nanotube(vertices)
    del pos
    collect()
    
    cdef float RAM_needed = round(8*Nk/1e9*(Nstat**2 + Nstat*Nsing + Nsing) + 32.0)
    #print(RAM_needed)
    #print(RAM_needed/RAM_avail)

    cdef int dumps = ceil(RAM_needed/RAM_avail)

    return dumps    

def calc_dispersion(Nsing, alphaR=ip.alphaR, betaD=ip.betaD, B=ip.B, save_file=False, Nk=2000, k=None):

    cdef int Nr = ip.Nr
    cdef int Nphi = ip.Nphi
    cdef int Nstat
    #cdef int Nsing
    cdef double Rext = ip.Rext
    cdef double R = ip.R
    cdef double t = ip.t
    cdef int vertices = ip.vertices

    cdef np.ndarray eigvals = np.zeros(Nk, dtype = object)
    cdef np.ndarray eigvecs = np.zeros(Nk, dtype = object)

    cdef int max_workers = 20
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        H_futures = pool.map(ham.Hmat, k, repeat(alphaR), repeat(betaD), repeat(B))

    #H_list = [sp.coo_matrix(value) for value in H_futures]
    H_list = [value for value in H_futures]
    del H_futures
    collect()

    for i in range(Nk):
        #eigvals[i], eigvecs[i], info = la.lapack.zheevd(H_list[i])
        #eigvals[i], eigvecs[i] = eigsh(H_list[i], k=Nsing, sigma=0.0j, which="LM")
        #eigvals[i], eigvecs[i] = la.eigh(H_list[i], lower=False, turbo=True, subset_by_index=[0,Nsing])
        eigvals[i], eigvecs[i] = la.eigh(H_list[i], lower=False, turbo=True, subset_by_index=[0,Nsing-1])
        #eigvals[i], eigvecs[i] = la.eigh(H_list[i], lower=False, turbo=True)
        #eigvals[i], eigvecs[i] = np.linalg.eigh(H_list[i], UPLO='U')
        #eigvals[i], eigvecs[i] = cp.linalg.eigh(cH_list[i], UPLO='U')

    #eigvals = ip.ts*eigvals

    if save_file==True:
        #np.savez_compressed(f"/scratch1/rperrin/{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.B}_{ip.alphaR}_{ip.betaD}_{ip.mag_direction}_sparse.npz", k, eigvals, eigvecs)
        np.savez_compressed(f"./eigensystems/{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.B}_{ip.alphaR}_{ip.betaD}_{ip.mag_direction}.npz", k, eigvals, eigvecs)
    else:
        return k, eigvals, eigvecs

def split_dispersion_calculation(Nsing, k, Nk):

    dumps = get_RAM_dumps(Nsing=Nsing, Nk=Nk)
    if dumps == 0: dumps = 1

    split_k = np.array_split(k, dumps)

    for d in range(dumps):
        Nk = len(split_k[d])
        k, eigvals, eigvecs = calc_dispersion(Nsing=Nsing, save_file=False, Nk=Nk, k=split_k[d])

        np.savez_compressed(f"./eigensystems/{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.B}_{ip.alphaR}_{ip.betaD}_{ip.mag_direction}_({d}).npz", k, eigvals, eigvecs)
