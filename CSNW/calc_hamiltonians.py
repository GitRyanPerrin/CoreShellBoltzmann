from concurrent.futures import ProcessPoolExecutor

import numpy as np

import input_data as ip
from hamiltonian import Hmat

Nk = 3000
k0 = 2*np.pi/ip.R
k = np.linspace(-k0, k0, Nk)

with ProcessPoolExecutor() as pool:
    H_fut = pool.map(Hmat, k)
H_arr = np.stack([fut for fut in H_fut])

np.savez_compressed(
    f"./hamiltonians/ham_{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.alphaR}_{ip.betaD}.npz",
    H_arr
)
