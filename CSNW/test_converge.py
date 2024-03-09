from itertools import repeat

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

import input_data as ip
from hamiltonian import quantum_numbers
from fermi_derivative import fermi_derivative
from load_file import load_eigensystem, load_velocity, load_spin_velocity
from k_structure import calc_dispersion
from spin_matrix import Sijs

def main():

    #print(f"Loading Eigensystem: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")
    #k, eigvals, eigvecs, Nstat, Nev = load_eigensystem()
    #print(f"Loading Velocity: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")

    #runs = []

    #diffusive_cond = np.zeros(len(runs))

    #for i,j in enumerate(runs):
    Nk = 300
    #Nk = 700
    Nev = 8
    k0 = np.pi/ip.R
    k = np.linspace(-k0,k0,Nk)

    eigvals = calc_dispersion(Nev=Nev, Nk=Nk, k=k, E=repeat(0*ip.E), save_file=False, eigvals_only=True)
    eigvals2 = calc_dispersion(Nev=Nev, Nk=Nk, k=k, E=repeat(ip.E), save_file=False, eigvals_only=True)
    #eigvals, eigvecs = calc_dispersion(Nev=Nev, Nk=Nk, k=k, save_file=False, eigvals_only=False)
    eigvals = np.stack(eigvals, axis=0)
    eigvals2 = np.stack(eigvals2, axis=0)
    #eigvecs = np.stack(eigvecs, axis=0)
    v = np.gradient(eigvals, axis=0)
    v2 = np.gradient(eigvals2, axis=0)

    #v_pos = np.where(v > 0.0, v, 0.0)
    #v_neg = np.where(v < 0.0, v, 0.0)
    #v = load_velocity()
    #sv = load_spin_velocity()
    #v = np.stack([eigvecs[ik].conj().T@v[ik]@eigvecs[ik] for ik in range(Nk)])
    #sv = np.stack([eigvecs[ik].conj().T@sv[ik]@eigvecs[ik] for ik in range(Nk)])

    #S = Sijs('p')
    #S = np.kron(np.eye(ip.Nr*ip.Nphi), S)
    #s = np.stack([eigvecs[ik].conj().T@S@eigvecs[ik] for ik in range(Nk)])
    #sv = np.stack([s[ik]*np.diag(v[ik]) + np.diag(v[ik])*s[ik] for ik in range(Nk)])

    #plt.plot(k, np.sum(np.diagonal(sv, axis1=1,axis2=2),axis=1))
    #plt.show()

    T = 0.5
    dmu = 5
    band_minimum = np.min(eigvals2)
    band_maximum = band_minimum + dmu
    Nmu = 80
    chem_potential = np.linspace(band_minimum-0.25, band_maximum, Nmu)

    ###
    print("Calculating First-Order Fermi-Derivative..")
    dFermi = np.stack([fermi_derivative(k, eigvals[:,:Nev], mu, T, order=2) for mu in chem_potential])
    dFermi1 = np.stack([fermi_derivative(k, eigvals[:,:Nev], mu, T, order=1) for mu in chem_potential])

    print("Calculating Diffusive Conductivity..")
    #diffusive_prod = v*v*np.diagonal(sv,axis1=1,axis2=2)
    #diffusive_prod = ip.alphaR**3*np.ones(Nev)
    #diffusive_prod = np.stack([v[ik]*v[ik]*np.real(np.diag(sv[ik])) for ik in range(Nk)])
    #diffusive_prod = np.stack([np.diag(v[ik])*np.diag(v[ik])*np.diag(sv[ik]) for ik in range(Nk)])
    #diffusive_prod = np.stack([np.diag(v[ik])*np.diag(v[ik]) for ik in range(Nk)])
    #diffusive_prod = ip.alphaR*v*v
    diffusive_prod = v*v
    diffusive_prod2 = v2*v2
    #ballistic_prod = v_pos - v_neg
    #diffusive_prod = v*v*v - 2*ip.alphaR**2*v
    #diffusive_prod = ip.alphaR*v*v + ip.alphaR**3*np.ones(Nev)
    #diffusive_prod = 3*ip.alphaR*v*v - 2*ip.alphaR**3*np.ones(Nev)
    #ballistic_integrand = np.stack([[ballistic_prod[ik, :Nev]*np.diag(dFermi1[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    diffusive_integrand = np.stack([[diffusive_prod[ik, :Nev]*np.diag(dFermi1[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    diffusive_integrand2 = np.stack([[diffusive_prod2[ik, :Nev]*np.diag(dFermi1[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    #diffusive_integrand = np.stack([[diffusive_prod[ik, :Nev]*np.diag(dFermi1[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    #diffusive_integrand = np.stack([[np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    #diffusive_integrand = np.stack([[np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    diffusive_cond = -np.stack([integrate.simps(diffusive_integrand[imu], k, axis=0) for imu in range(Nmu)])
    diffusive_cond2 = -np.stack([integrate.simps(diffusive_integrand2[imu], k, axis=0) for imu in range(Nmu)])
    #ballistic_cond = -np.stack([integrate.simps(ballistic_integrand[imu], k, axis=0) for imu in range(Nmu)])
    #e_cond = -np.stack([integrate.simps(e_integrand[imu], k, axis=0) for imu in range(Nmu)])
    diffusive_cond = np.real(np.trace(diffusive_cond, axis1=1, axis2=2))
    diffusive_cond2 = np.real(np.trace(diffusive_cond2, axis1=1, axis2=2))
    #ballistic_cond = np.real(np.trace(ballistic_cond, axis1=1, axis2=2))
    #e_cond = np.trace(e_cond, axis1=1, axis2=2)

    fig, axs = plt.subplots()
    #axs2 = axs.twinx()

    #axs.set_ylabel('Spin Cond')
    #axs2.set_ylabel('Ballistic Cond')
    axs.set_ylabel('Diffusive Cond')
    axs.set_xlabel('Chem. Pot.')

    axs.plot(chem_potential, diffusive_cond, 'k')
    axs.plot(chem_potential, diffusive_cond2, 'r')
    #axs2.plot(chem_potential, ballistic_cond/ballistic_cond[20], 'r')
    #axs2.plot(chem_potential, e_cond, 'r')
    #axs.plot(chem_potential, e_cond, 'r')
    plt.show()

if __name__=="__main__":
    main()

