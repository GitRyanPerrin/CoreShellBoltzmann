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

    #Nk = 2000
    #k, eigvals, eigvecs = calc_dispersion(num_pts_k=Nk)
    k, eigvals, eigvecs, Nstat, Nsing = load_eigensystem()
    Nk = len(k)
    Nsing = eigvals.shape[-1]
    if Nsing > 50: Nsing = 100
    #Nsing = 30
    print(f"Eigvecs: {eigvecs.shape}")

    #K, energy = interpolate_energy(k, eigvals)

    print("Calculating Velocity...")
    #print(f"Eigvals shape: {eigvals.shape}")
    velocity = np.gradient(eigvals, axis=0)
    print(f"Velocity vector shape: {velocity.shape}")
    #velocity = np.apply_along_axis(np.diag, 1, velocity)
    print(f"Velocity matrix shape: {velocity.shape}")
    #velocity = np.gradient(energy, axis=1)
    #print("velocity shape: ", velocity.shape)
    vr = np.where(velocity > 0.0, velocity, 0.0)
    vrave = np.stack([np.average(vr[ik]) for ik in range(Nk)])
    vrup = np.stack([np.where(vr[ik] > vrave[ik], vr[ik], 0.0) for ik in range(Nk)])
    vrdown = np.stack([np.where(vr[ik] < vrave[ik], vr[ik], 0.0) for ik in range(Nk)])
    vl = np.where(velocity < 0.0, velocity, 0.0)
    vlave = np.stack([np.average(vl[ik]) for ik in range(Nk)])
    vlup = np.stack([np.where(vl[ik] < vlave[ik], vl[ik], 0.0) for ik in range(Nk)])
    vldown = np.stack([np.where(vl[ik] > vlave[ik], vl[ik], 0.0) for ik in range(Nk)])
    #print(vr - vl)

    T = 0.1
    band_min = np.min(eigvals)
    Nmu = 100
    dmu = 40.0
    chem_potential = np.linspace(band_min, band_min+dmu, Nmu)

    # Diffusive Transport
    #elec_prod = (vrup - vlup)**2
    #elec_prod = vrup*vrup
    #elec_prod = velocity*velocity
    #therm_prod = T*np.stack([velocity*velocity*(eigvals-mu) for mu in chem_potential], axis=0)

    # Ballistic Transport
    #elec_prod = vr-vl
    #therm_prod = np.stack([(vr-vl)*(eigvals-mu)/T for mu in chem_potential], axis=0)

    # Spin Polarized Ballistic Transport
    #elec_prod = vrup-vlup
    elec_prod = vrup**2 - vrdown**2

    dFermi = np.stack([fermi_derivative(k, eigvals[:,:Nsing], mu, T, order=1) for mu in chem_potential], axis=0)
    elec_integrand = np.stack([[elec_prod[ik, :Nsing]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    #therm_integrand = np.stack([[therm_prod[imu, ik, :Nsing]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    up_cond = -ip.units_cond*np.sum(np.sum([integrate.simps(elec_integrand[imu], k, axis=0) for imu in range(Nmu)], axis=1), axis=1)
    #cond = -ip.units_cond*np.sum(np.sum([integrate.simps(therm_integrand[imu], k, axis=0) for imu in range(Nmu)], axis=1), axis=1)

    #fig, axs = plt.subplots(2,1)
    fig, axs = plt.subplots()
    axs.set_xlabel('$\mu$')
    axs.set_ylabel('$\sigma^e$')

    axs.plot(chem_potential, up_cond, 'r--')

    '''
    # Spin Polarized Ballistic Transport
    elec_prod = vrdown-vldown
    #elec_prod = vrdown*vrdown
    #elec_prod = (vrdown - vldown)**2

    dFermi = np.stack([fermi_derivative(k, eigvals[:,:Nsing], mu, T, order=1) for mu in chem_potential], axis=0)
    elec_integrand = np.stack([[elec_prod[ik, :Nsing]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    #therm_integrand = np.stack([[therm_prod[imu, ik, :Nsing]*np.diag(dFermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
    down_cond = -ip.units_cond*np.sum(np.sum([integrate.simps(elec_integrand[imu], k, axis=0) for imu in range(Nmu)], axis=1), axis=1)
    #cond = -ip.units_cond*np.sum(np.sum([integrate.simps(therm_integrand[imu], k, axis=0) for imu in range(Nmu)], axis=1), axis=1)

    axs.plot(chem_potential, down_cond, 'b--')

    cond = up_cond + down_cond
    axs.plot(chem_potential, cond, 'k--')

    #axs.plot(chem_potential, -(up_cond-down_cond)/cond, 'k')

    #fig.savefig(f'./plots/ball_test.png')
    '''
    plt.show()

if __name__=="__main__":
    main()
    #plot_Sijs()
    #test_velocity()

