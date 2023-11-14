'''
This was an attempt at using v = dE/dk to determine the spin-velocity,
but since it is not a matrix, it does not seem to properly work.
'''

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

import input_data as ip
from load_file import load_eigensystem
from fermi_derivative import fermi_derivative
from spin_matrix import Sijs

def anticommutator(A, B):
    return A@B + B@A

def main():

    print(f"Loading Eigensystem: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")
    k, eigvals, eigvecs, Nstat, Nev = load_eigensystem()

    print(eigvals.shape, eigvecs.shape)

    # number of k-points
    Nk = len(k)

    print("Calculating Velocity...")
    velocity = np.gradient(eigvals, axis=0)

    print("Calculating Spin Velocity...")
    # Spin matrix in position+spin basis
    # The Sijs function can take arguments: 'x', 'y', 'z', 'r', 'p',
    # where each corresponds to a Pauli-Matrix in the ijs basis
    S = Sijs('p')
    # Transform S to ak/energy basis
    S = np.stack([eigvecs[ik].conj().T@S@eigvecs[ik] for ik in range(Nk)])
    # Calculate spin-velocity
    spin_velocity = np.stack([anticommutator(np.diag(velocity[ik]), S[ik]) for ik in range(Nk)])

    print("Calculating Integrand...")
    spin_product = np.stack([np.diag(velocity[ik])**2*spin_velocity[ik] for ik in range(Nk)])

    # temperature and chemical potential details
    T = 0.1
    # number of chem-pot points
    Nmu = 100
    # range of chem-pots
    dmu = 10.0
    band_min = np.min(eigvals)
    band_max = band_min + dmu
    # domain of conductivity calculation
    chem_potential = np.linspace(band_min, band_max, Nmu)

    # calc fermi-derivative
    dFermi = np.stack([fermi_derivative(k, eigvals[:,:Nev], mu, T, order=2) for mu in chem_potential], axis=0)
    spin_integrand = np.stack([[spin_product[ik, :Nev]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    print("Calculating Conductivity...")
    spin_cond = np.stack([np.sum(integrate.simps(spin_integrand[imu], k, axis=0)) for imu in range(Nmu)])

    plt.plot(chem_potential, spin_cond)
    plt.show()

if __name__=="__main__":
    main()

