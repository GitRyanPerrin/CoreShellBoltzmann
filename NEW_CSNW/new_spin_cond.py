'''
This code is intended to use the explicit definitions of the charge-velocity and
spin-velocity matrices. Computing the charge transport differs from the use of
v = dE/dk, which is clearly problematic.
'''

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

import input_data as ip
from hamiltonian import quantum_numbers
from fermi_derivative import fermi_derivative
from load_file import load_eigensystem, load_velocity, load_spin_velocity
from spin_matrix import Sijs

def check_spin_z(eigvals, eigvecs):

    Nk = eigvals.shape[0]
    Nev = eigvals.shape[1]
    Nstat = eigvecs.shape[1]

    evup = eigvecs[:, :int(Nstat/2), :]
    evdn = eigvecs[:, int(Nstat/2):, :]

    #sz = Sijs('z')

    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    #sz = np.kron(sz, np.eye(500))
    sz = np.kron(np.eye(500), sz)

    szu = np.zeros([Nk, Nev])
    #szd = np.zeros([Nk, Nev])
    for ik in range(Nk):
        for a in range(Nev):
            szu[ik, a] = round(np.real(evup[ik,:,a].conj().T@sz[:500, :500]@evup[ik,:,a]))
            #szd[ik, a] = round(np.real(evdn[ik,:,a].conj().T@sz[500:, 500:]@evdn[ik,:,a]))
    #return szu, szd
    return szu

def main():

    print(f"Loading Eigensystem: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")
    k, eigvals, eigvecs, Nstat, Nev = load_eigensystem()
    print(f"Loading Velocity: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")
    v = load_velocity()
    print(f"Loading Spin Velocity: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")
    sv = load_spin_velocity()

    # number of k-points
    Nk = len(k)


    # Setting temperature and chemical potential
    T = 1.0
    # number of chemical potential values
    Nmu = 100
    # range of chemical potentials
    dmu = 10.0
    band_minimum = np.min(eigvals)
    band_maximum = band_minimum + dmu
    # creates the domain for conductivity as a function of mu
    chem_potential = np.linspace(band_minimum, band_maximum, Nmu)

    ###
    print("Calculating First-Order Fermi-Derivative..")
    dFermi = np.stack([fermi_derivative(k, eigvals[:,:Nev], mu, T, order=2) for mu in chem_potential], axis=0)
    v = np.stack([eigvecs[ik].conj().T@v[ik]@eigvecs[ik] for ik in range(Nk)])
    sv = np.stack([eigvecs[ik].conj().T@sv[ik]@eigvecs[ik] for ik in range(Nk)])
    #sv_diag = np.stack([np.diag(sv[ik]) for ik in range(Nk)])
    #svp = np.where(sv_diag > 0.0, sv_diag, 0.0)
    #svd = np.where(sv_diag < 0.0, sv_diag, 0.0)
    v_diag = np.stack([np.diag(v[ik]) for ik in range(Nk)])
    vp = np.where(v_diag > 0.0, v_diag, 0.0)
    vd = np.where(v_diag < 0.0, v_diag, 0.0)

    ###
    print("Calculating Diffusive Conductivity..")
    #diffusive_prod = np.stack([eigvecs[ik].conj().T@v[ik]@v[ik]@eigvecs[ik] for ik in range(Nk)])
    #diffusive_prod = np.stack([eigvecs[ik].cmmonj().T@v[ik]@eigvecs[ik]@eigvecs[ik].conj().T@v[ik]@eigvecs[ik] for ik in range(Nk)])
    diffusive_prod = np.stack([sv[ik]@v[ik]@v[ik] for ik in range(Nk)])
    diffusive_integrand = np.stack([[diffusive_prod[ik, :Nev]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    diffusive_cond = -np.stack([integrate.simps(diffusive_integrand[imu], k, axis=0) for imu in range(Nmu)])
    diffusive_cond = np.trace(diffusive_cond, axis1=1, axis2=2)

    ###
    print("Calculating Ballistic Conductivity..")
    #ballistic_prod = np.stack([(vp[ik]-vd[ik])@sv[ik] for ik in range(Nk)])
    #ballistic_integrand = np.stack([[ballistic_prod[ik, :Nev]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    #ballistic_cond = -np.stack([integrate.simps(ballistic_integrand[imu], k, axis=0) for imu in range(Nmu)])
    #ballistic_cond = np.trace(ballistic_cond, axis1=1, axis2=2)

    fig, axs = plt.subplots()
    # Creating Plot
    # Creates second y-axis w/ new scaling
    ax2 = axs.twinx()
    # Labels
    axs.set_xlabel('$\mu$')
    axs.set_ylabel('$\sigma^e$')
    ax2.set_ylabel('$G^e$')

    # Plot Lines | Diffusive is black ('k') and Ballistic is red ('r')
    #ax2.plot(chem_potential, ballistic_cond, 'b')
    axs.plot(chem_potential, diffusive_cond, 'k')

    plt.show()
    suffix = f"{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.alphaR}_{ip.betaD}"
    plt.savefig("./plots/e_cond_" + suffix + ".png")

if __name__=="__main__":
    main()
