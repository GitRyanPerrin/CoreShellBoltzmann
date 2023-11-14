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

def anticommutator(A, B):
    return A@B + B@A

def main():

    print(f"Loading Eigensystem: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")
    k, eigvals, eigvecs, Nstat, Nev = load_eigensystem()

    # number of k-points
    Nk = len(k)

    print("Calculating Velocity...")
    velocity = np.gradient(eigvals, axis=0)
    v_positive = np.where(velocity > 0.0, velocity, 0.0)
    v_negative = np.where(velocity < 0.0, velocity, 0.0)

    T = 0.1
    band_min = np.min(eigvals)
    Nmu = 100
    dmu = 5.0
    chem_potential = np.linspace(band_min, band_min+dmu, Nmu)

    dFermi = np.stack([fermi_derivative(k, eigvals[:,:Nev], mu, T, order=1) for mu in chem_potential], axis=0)

    # Diffusive Transport
    diff_therm_prod = np.stack([velocity*velocity*(eigvals-mu)/T for mu in chem_potential], axis=0)
    diff_therm_integrand = np.stack([[diff_therm_prod[imu, ik, :Nev]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    diff_therm_cond = np.sum(np.sum([integrate.simps(diff_therm_integrand[imu], k, axis=0) for imu in range(Nmu)], axis=1), axis=1)

    # Ballistic Transport
    ball_therm_prod = np.stack([(v_positive-v_negative)*(eigvals-mu)/T for mu in chem_potential], axis=0)
    ball_therm_integrand = np.stack([[ball_therm_prod[imu, ik, :Nev]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    ball_therm_cond = np.sum(np.sum([integrate.simps(ball_therm_integrand[imu], k, axis=0) for imu in range(Nmu)], axis=1), axis=1)

    fig, axs = plt.subplots()
    axs2 = axs.twinx()
    axs.set_xlabel('$\mu$')
    axs.set_ylabel('S (diffusive)')
    axs2.set_ylabel('S (ballistic)')

    axs.plot(chem_potential, diff_therm_cond, 'k')
    axs.plot(chem_potential, ball_therm_cond, 'r')
    plt.show()

if __name__=="__main__":
    main()
    #plot_Sijs()
    #test_velocity()

