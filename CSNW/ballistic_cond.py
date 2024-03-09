from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

import input_data as ip
from fermi_derivative import fermi_derivative
from load_file import load_eigensystem

def main():

    print(f"Loading Eigensystem: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")
    k, eigvals, eigvecs, Nstat, Nev = load_eigensystem()

    # number of k-points
    Nk = len(k)

    print("Calculating Velocity...")
    velocity = np.gradient(eigvals, axis=0)
    # 'Right' moving electrons
    v_positive = np.where(velocity > 0.0, velocity, 0.0)
    # 'Left' moving electrons
    v_negative = np.where(velocity < 0.0, velocity, 0.0)

    # Setting temperature and chemical potential
    T = 0.5
    # number of chemical potential values
    Nmu = 100
    # range of chemical potentials
    dmu = 5.0
    band_minimum = np.min(eigvals)
    band_maximum = band_minimum + dmu
    # creates the domain for conductivity as a function of mu
    chem_potential = np.linspace(band_minimum, band_maximum, Nmu)

    ###
    print("Calculating First-Order Fermi-Derivative..")
    dFermi = np.stack([fermi_derivative(k, eigvals[:,:Nev], mu, T, order=1) for mu in chem_potential], axis=0)

    ###
    print("Calculating Diffusive Conductivity..")
    diffusive_prod = v_positive*v_positive
    diffusive_integrand = np.stack([[diffusive_prod[ik, :Nev]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    diffusive_cond = -np.sum(np.sum([integrate.simps(diffusive_integrand[imu], k, axis=0) for imu in range(Nmu)], axis=1), axis=1)

    ###
    print("Calculating Ballistic Conductivity..")
    ballistic_prod = v_positive-v_negative
    ballistic_integrand = np.stack([[ballistic_prod[ik, :Nev]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    ballistic_cond = -np.sum(np.sum([integrate.simps(ballistic_integrand[imu], k, axis=0) for imu in range(Nmu)], axis=1), axis=1)

    # Creating Plot
    fig, axs = plt.subplots()
    # Creates second y-axis w/ new scaling
    ax2 = axs.twinx()
    # Labels
    axs.set_xlabel('$\mu$')
    axs.set_ylabel('$\sigma^e$')
    ax2.set_ylabel('$G^e$')

    # Plot Lines | Diffusive is black ('k') and Ballistic is red ('r')
    axs.plot(chem_potential, diffusive_cond, 'k')
    ax2.plot(chem_potential, ballistic_cond, 'r')

    plt.show()
    suffix = f"{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.alphaR}_{ip.betaD}"
    plt.savefig("./plots/e_cond_" + suffix + ".png")

if __name__=="__main__":
    main()
