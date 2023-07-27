from concurrent.futures import ProcessPoolExecutor
from gc import collect

import numpy as np
from numpy import pi, sqrt, cos, sin, exp
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import simps

import input_data as ip
import lattice as lat
import hamiltonian as ham
import k_structure
from fermi_energy import dFD, ddFD

def calc_ham(k):

    Nk = len(k)

    with ProcessPoolExecutor() as pool:
        H_futures = pool.map(ham.Hmat, k)

    Hk = np.stack([future for future in H_futures])
    del H_futures
    collect()

    Hk = np.stack([Hk[ik] + Hk[ik].conj().T - np.diag(np.diag(Hk[ik])) for ik in range(Nk)])

    return Hk

def calc_velocity(Hk):

    return np.gradient(Hk, axis=0)

def anticommutator(A, B):

    Nk = A.shape[0]
    return np.stack([A[ik]@B + B@A[ik] for ik in range(Nk)])

def Sigma(type):

    Nr = ip.Nr
    Nphi = ip.Nphi
    Nsites = Nr*Nphi

    sigma_z = np.array([[1,0],[0,-1]])
    sigma_x = np.array([[0, 1],[1,0]])
    sigma_y = np.array([[0,-1j],[1j,0]])

    cos_eye = np.zeros(Nphi)
    sin_eye = np.zeros(Nphi)
    phi = np.linspace(0, 2*pi, Nphi)
    if type == 'r' or type == 'p':
        for j in range(Nphi):
            cos_eye[j] = cos(phi[j])
            sin_eye[j] = sin(phi[j])

        cos_eye = np.kron(np.eye(Nr), np.diag(cos_eye))
        sin_eye = np.kron(np.eye(Nr), np.diag(sin_eye))

    if type == 'x':
        return np.kron(sigma_x, np.eye(Nsites))
    if type == 'y':
        return np.kron(sigma_y, np.eye(Nsites))
    if type == 'z':
        return np.kron(sigma_z, np.eye(Nsites))
    if type == 'r':
        return np.kron(sigma_x, cos_eye) + np.kron(sigma_y, sin_eye)
    if type == 'p':
        return -np.kron(sigma_x, sin_eye) + np.kron(sigma_y, cos_eye)

def fermi_matrix(k, eigvals, mu, T, order=1):

    Nk = len(k)
    Nsing = np.shape(eigvals)[1]
    fermi = np.zeros([Nk, Nsing])
    for a in range(Nsing):
        for ik in range(Nk):
            if order==1:
                fermi[ik,a] = dFD(eigvals[ik,a], mu, T)
            if order==2:
                fermi[ik,a] = ddFD(eigvals[ik,a], mu, T)
    return fermi

def main():

    Nk = 1000
    k = np.linspace(-50, 50, Nk)
    Hk = calc_ham(k)
    vel = calc_velocity(Hk)
    #s_vel = anticommutator(vel, Sigma(type='z'))

    eigvals = np.zeros(Nk, dtype=object)
    eigvecs = np.zeros(Nk, dtype=object)
    for ik in range(Nk):
        eigvals[ik], eigvecs[ik] = eigh(Hk[ik])
    eigvals = np.stack(eigvals)
    eigvecs = np.stack(eigvecs)

    vel = np.stack([eigvecs[ik].conj().T@vel[ik]@eigvecs[ik] for ik in range(Nk)])
    vr = np.where(vel > 0.0, vel, 0.0)
    vl = np.where(vel < 0.0, vel, 0.0)
    #s_vel = np.stack([eigvecs[ik].conj().T@s_vel[ik]@eigvecs[ik] for ik in range(Nk)])

    factor = vr-vl
    #factor = np.stack([vel[ik]@vel[ik]@s_vel[ik] for ik in range(Nk)])
    #factor = np.stack([vel[ik]@vel[ik] for ik in range(Nk)])

    Nmu = 100
    chem_potential = np.linspace(np.min(eigvals), np.min(eigvals)+10, Nmu)
    #fermi = np.stack([fermi_matrix(k, eigvals, mu, 0.1, order=2) for mu in chem_potential], axis=0)
    fermi = np.stack([fermi_matrix(k, eigvals, mu, 0.1, order=1) for mu in chem_potential], axis=0)
    integrand = np.stack([[factor[ik]@fermi[imu,ik] for ik in range(Nk)] for imu in range(Nmu)])

    cond = np.sum(np.stack([simps(integrand[imu], k, axis=0) for imu in range(Nmu)]), axis=1)

    print(cond.shape)

    plt.plot(chem_potential, cond)
    plt.show()

if __name__=="__main__":
    main()






