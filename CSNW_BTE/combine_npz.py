import glob, os

import numpy as np

import input_data as ip

suffix = f"{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.B}_{ip.alphaR}_{ip.betaD}_{ip.mag_direction}"

files = glob.glob("./eigensystems/" + suffix + "_*")
Nfiles = len(files)
print(Nfiles)

k = np.array([])
eigvals = np.array([])
eigvecs = np.array([])

for file in files:
    with np.load(file, allow_pickle=True) as f:
        k = np.append(k, f['arr_0'])
        eigvals = np.append(eigvals, f['arr_1'])
        eigvecs = np.append(eigvecs, f['arr_2'])

k = np.stack(k)
eigvals = np.stack(eigvals)
eigvecs = np.stack(eigvecs)

np.savez_compressed(f"./eigensystems/" + suffix + ".npz", k, eigvals, eigvecs)
