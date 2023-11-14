'''
Attempts to calculate the spin-seebeck coefficient. Does not work.
'''

import scipy.linalg as la
from scipy.interpolate import interp1d
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import gc
from scipy.signal import savgol_filter

import input_data as ip
from load_file import load_eigensystem
from fermi_derivative import fermi_derivative
from spin_matrix import Sijs

def anticommutator(A, B):
    return A@B + B@A

def main():

    print(f"Loading Eigensystem: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")

    vertices = ip.vertices
    k, eigvals, eigvecs, Nstat, Nev = load_eigensystem()
    Nk = len(k)

    print("Calculating Velocity...")
    velocity = np.gradient(eigvals, axis=0)

    print("Calculating Spin Velocity...")
    S = Sijs('z')
    S = np.stack([eigvecs[ik].conj().T@S@eigvecs[ik] for ik in range(Nk)])
    spin_vel = np.stack([anticommutator(np.diag(velocity[ik]), S[ik]) for ik in range(Nk)])

    T = 0.5
    band_min = np.min(eigvals)
    Nmu = 100
    dmu = 5.0
    chem_potential = np.linspace(band_min, band_min+dmu, Nmu)

    print("Calculating Integrand...")
    spin_therm_prod = np.stack(np.stack([[np.diag(velocity[ik])@spin_vel[ik]@(eigvals[ik]-mu) for ik in range(Nk)] for mu in chem_potential]))

    dFermi = np.stack([fermi_derivative(k, eigvals[:,:Nev], mu, T, order=1) for mu in chem_potential], axis=0)
    spin_integrand = np.stack([[spin_therm_prod[imu, ik, :Nev]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    print("Calculating Conductivity...")
    spin_therm_cond = np.stack([np.sum(integrate.simps(spin_integrand[imu], k, axis=0)) for imu in range(Nmu)])

    fig, axs = plt.subplots()
    axs.set_xlabel('$\mu-E_0$')
    axs.set_ylabel('$S^s$')

    axs.plot(chem_potential-band_min, spin_therm_cond, 'k')
    plt.show()

if __name__=="__main__":
    main()

