import scipy.linalg as la
from scipy.interpolate import interp1d
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import gc
from scipy.signal import savgol_filter

import input_data as ip
from hamiltonian import Hmat, quantum_numbers
from fermi_energy import dFD, ddFD
from k_structure import calc_dispersion
from spin_matrix import Sijs, unitary_transform, calc_unitary_transform

def anticommutator(A, B):
    return A@B + B@A

def load_eigensystem():

    suffix = f"{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.B}_{ip.alphaR}_{ip.betaD}_{ip.mag_direction}"
    #with np.load(f"/scratch/rperrin/rperrin/" + suffix + "_sparse.npz", allow_pickle=True) as file:
    with np.load(f"./eigensystems/" + suffix + ".npz", allow_pickle=True) as file:
        k = file['arr_0']
        Nk = len(k)
        eigvals = file['arr_1']
        eigvals = np.stack(eigvals, axis=0)
        Nstat = len(eigvals[0])
        eigvecs = file['arr_2']
        eigvecs = np.stack(eigvecs, axis=0)
        Nsing = np.shape(eigvecs[0])[1]

        return k, eigvals, eigvecs, Nstat, Nsing

def fermi_derivative(k, eigvals, mu, T, order=1):

    Nk = len(k)
    Nsing = np.shape(eigvals)[1]
    #fermi = np.zeros([Nsing, Nk])
    fermi = np.zeros([Nk, Nsing])
    for a in range(Nsing):
        for ik in range(Nk):

            if order==1:
                fermi[ik,a] = dFD(eigvals[ik,a], mu, T)
            if order==2:
                fermi[ik,a] = ddFD(eigvals[ik,a], mu, T)

    return fermi

def interpolate_energy(k, eigvals):

    Nsing = np.shape(eigvals)[0]
    model = [interp1d(k, eigvals[a, :], kind='cubic') for a in range(Nsing)]
    Nk = 10000
    K = np.linspace(k[0], k[-1], Nk)
    energy = np.stack([model[a](K) for a in range(Nsing)], axis=0)

    return K, energy

def plot_Sijs():

    k, eigvals, eigvecs, Nstat, Nsing = load_eigensystem()
    Nsing = eigvecs.shape[-1]
    #print(eigvecs[0].conj().T.shape, eigvecs[0].shape)
    s1 = eigvecs[0].conj().T@Sijs('p')@eigvecs[0]
    #s1 = eigvecs[0]@Sijs('p')[:Nsing, :Nsing]@eigvecs[0].conj().T
    s1 = s1.astype(np.single)
    print(np.trace(s1[:Nsing, :Nsing]))
    plt.imshow(np.real(s1))
    #plt.imshow(np.imag(s1))
    plt.colorbar()
    plt.savefig('./plots/spin_matrix_plot.png')


def main():

    print(f"Loading Eigensystem: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")

    vertices = ip.vertices
    k, eigvals, eigvecs, Nstat, Nsing = load_eigensystem()
    Nk = len(k)
    Nsing = eigvals.shape[-1]
    if Nsing > 50: Nsing = 100
    print(f"Eigvecs: {eigvecs.shape}")

    print("Calculating Velocity...")
    velocity = np.gradient(eigvals, axis=0)

    vr = np.where(velocity > 0.0, velocity, 0.0)
    vl = np.where(velocity < 0.0, velocity, 0.0)

    print("Calculating Spin Velocity...")
    S = Sijs('r')
    #S = np.stack([eigvecs[ik,:Nsing,:Nsing]@S[:Nsing,:Nsing]@eigvecs[ik,:Nsing,:Nsing].conj().T for ik in range(Nk)])
    S = np.stack([eigvecs[ik].conj().T@S@eigvecs[ik] for ik in range(Nk)])

    #s_vel = np.stack([anticommutator(np.diag(velocity[ik]), S[ik]) for ik in range(Nk)])
    s_vr = np.stack([anticommutator(np.diag(vr[ik]), S[ik]) for ik in range(Nk)])
    s_vl = np.stack([anticommutator(np.diag(vl[ik]), S[ik]) for ik in range(Nk)])

    T = 0.1
    band_min = np.min(eigvals)
    Nmu = 100

    dmu = 5.0
    chem_potential = np.linspace(band_min, band_min+dmu, Nmu)

    print("Calculating Integrand...")
    #prod = np.stack([np.diag(velocity[ik])*np.diag(velocity[ik])*s_vel[ik] for ik in range(Nk)])
    #e_prod = np.stack([np.diag(velocity[ik])**2 for ik in range(Nk)])
    #s_prod = np.stack([np.diag(velocity[ik])@s_vel[ik] for ik in range(Nk)])

    e_prod = vr - vl
    s_prod = s_vr - s_vl

    dFermi = np.stack([fermi_derivative(k, eigvals[:,:Nsing], mu, T, order=1) for mu in chem_potential], axis=0)
    #dFermi = np.stack([fermi_derivative(k, eigvals[:,:Nsing], mu, T, order=2) for mu in chem_potential], axis=0)
    #e_integrand = -np.stack([[e_prod[ik, :Nsing]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    #s_integrand = np.stack([[s_prod[ik, :Nsing]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    e_integrand = -ip.units_cond*np.stack([[np.diag(e_prod[ik, :Nsing])*dFermi[imu, ik] for ik in range(Nk)] for imu in range(Nmu)])
    s_integrand = ip.units_spin*np.stack([[np.diag(s_prod[ik, :Nsing])*dFermi[imu, ik] for ik in range(Nk)] for imu in range(Nmu)])
    #cond = np.sum(np.sum([integrate.simps(integrand[imu], k, axis=0) for imu in range(Nmu)], axis=1), axis=1)
    print("Calculating Conductivity...")
    #cond = np.stack([np.trace(integrate.simps(integrand[imu], k, axis=0)) for imu in range(Nmu)])
    e_cond = np.stack([np.sum(integrate.simps(e_integrand[imu], k, axis=0)) for imu in range(Nmu)])
    s_cond = np.stack([np.sum(integrate.simps(s_integrand[imu], k, axis=0)) for imu in range(Nmu)])

    fig, axs = plt.subplots()
    axs2=axs.twinx()
    axs.set_xlabel('$\mu-E_0$')
    axs.set_ylabel('$\sigma^e$')
    axs2.set_ylabel('$\sigma^s_\phi$')

    axs.tick_params(axis='y', colors='red')
    axs.spines['right'].set_color('red')
    axs.yaxis.label.set_color('red')
    axs.title.set_color('red')

    axs2.tick_params(axis='y', colors='blue')
    axs2.spines['right'].set_color('blue')
    axs2.yaxis.label.set_color('blue')
    axs2.title.set_color('blue')

    axs.plot(chem_potential-band_min, e_cond, 'r')
    axs2.plot(chem_potential-band_min, s_cond, 'b')

    plt.show()

if __name__=="__main__":
    main()

