import glob, os

import numpy as np

import input_data as ip

suffix = f"{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.alphaR}_{ip.betaD}"

# Search for files
files = glob.glob("./eigensystems/" + suffix + "_*")
# Number of files
Nfiles = len(files)
print(Nfiles)

# Arrays of empty lists
k = np.array([])
eigvals = np.array([])
eigvecs = np.array([])

# Append each file to lists
for file in files:
    with np.load(file, allow_pickle=True) as f:
        k = np.append(k, f['arr_0'])
        eigvals = np.append(eigvals, f['arr_1'])
        eigvecs = np.append(eigvecs, f['arr_2'])

# Stack array of lists into multidimensional arrays
k = np.stack(k)
eigvals_up = np.stack(eigvals)
eigvecs_up = np.stack(eigvecs)

# Save as single file
np.savez_compressed(
    f"./eigensystems/" + suffix + ".npz",
    k,
    eigvals,
    eigvecs
)
