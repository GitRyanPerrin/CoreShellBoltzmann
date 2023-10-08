import numpy as np

import input_data as ip

def load_eigensystem():

    '''
    Loads the eigensystem corresponding to settings in 'input_data.py'
    from HDD/SSD. Requires the calculation of the E vs k dispersion
    prior to running ('calc_k_structure.py'). The load function requires
    the 'eigensytems' folder to be stored in the working directory.
    '''

    suffix = f"{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.alphaR}_{ip.betaD}"
    with np.load(f"./eigensystems/" + suffix + ".npz", allow_pickle=True) as file:
        k = file['arr_0']
        eigvals = file['arr_1']
        eigvecs = file['arr_2']

    # Ensures that the eigenvalues and eigenvectors are arrays
    # of shape [Nk, Nsing] and [Nk, Nstat, Nsing], respectively.
    eigvals = np.stack(eigvals, axis=0)
    eigvecs = np.stack(eigvecs, axis=0)

    # Extracts the total number of states and number of eigenvalues
    # extracted from the file
    Nstat = eigvecs[0].shape[0]
    Nev = eigvecs[0].shape[1]

    return k, eigvals, eigvecs, Nstat, Nev
