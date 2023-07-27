import gc
import time
from itertools import repeat

import concurrent.futures
import numpy as np
from numpy import pi, sin, cos, exp, sqrt, heaviside
import scipy.linalg as la
from scipy.special import expit
from scipy.optimize import fsolve, newton_krylov, anderson
from scipy import integrate
import matplotlib.pyplot as plt

import input_data as ip
from hamiltonian import Hmat, SigMat, quantum_numbers
from lattice import prismatic_nanotube, nanotube

def FD(E, mu, T):
    kBT = ip.kB*T
    x = (E-mu)/kBT
    if x > 0:
        return exp(-x)/(1.0+exp(-x))
    else:
        return 1.0/(1.0+exp(x))

def dFD(E, mu, T):
    #dE = 10**(-9)
    #return (FD(E+dE, mu, T) - FD(E-dE, mu, T))/2/dE
    return -FD(E,mu,T)*(1.0-FD(E,mu,T))

def ddFD(E, mu, T):
    kBT = ip.kB*T
    return -(1-2*FD(E,mu,T))*dFD(E,mu,T)/kBT

def particle_density(k, eigvals, mu, T):
    Nr, Nphi, R, vertices, circum, area = ip.Nr, ip.Nphi, ip.R, ip.vertices, ip.circum, ip.area

    if vertices > 0:
        Nsing = 4*ip.vertices
    else:
        Nsing = 24

    ndens = 0
    Nstat = len(eigvals[0])
    Nk = len(eigvals)

    integrand = np.zeros(Nk)
    for ik in range(Nk):
        for a1 in range(Nsing):
            integrand[ik] += np.float128(FD(eigvals[ik][a1], mu, T))

    if Nr == 1:
        ndens = integrate.simps(integrand, k)/2/pi/circum*10**3
    else:
        ndens = integrate.simps(integrand, k)/2/pi/area*10**4

    return ndens

def calculate_fermi_energy_naive(T=1.0, B=ip.B, L=ip.L):
    Nl = ip.Nl
    vertices = ip.vertices
    Nr, Nphi = ip.Nr, ip.Nphi

    if vertices > 0:
        Nsing = 4*vertices
    else:
        Nsing = 16

    if L == 'inf':
        k, eigvals, eigvecs = calc_dispersion(B)
        Nk = len(k)
        E_min, E_max = eigvals[int(Nk/2)][0], eigvals[int(Nk/2)][Nsing]
    else:
        with concurrent.futures.ProcessPoolExecutor() as pool:
            h_futures = pool.map(Hmat, range(Nl))
        h_list = [future for future in h_futures]

        eigvals, eigvecs = np.zeros(Nl, dtype=object), np.zeros(Nl, dtype=object)
        for l in range(Nl):
            eigvals[l], eigvecs[l], info = la.lapack.zheevd(h_list[l])
        E_min, E_max = eigvals[0][0], round(eigvals[0][Nsing])
    print(E_min, E_max)
    E_div = int(5*(E_max-E_min))

    fermi_energy = 0
    loop_iter = 0
    while fermi_energy == 0:
        print(loop_iter)
        E_array = np.linspace(E_min, E_max, E_div)
        if L == 'inf':
            part_density = [particle_density(k, eigvals, eigvecs, mu, T) for mu in E_array]
        else:
            part_density = [finite_wire_particle_density(eigvals, mu, T) for mu in E_array]

        if Nr == 1:
            n_measure = ip.n_InAs_2D
        else:
            n_measure = ip.n_InAs_3D

        for value in part_density:
            if round(value, 2) == round(n_measure, 2):
                fermi_energy = E_array[part_density.index(value)]
        E_div += int(2*(E_max-E_min))
        loop_iter += 1
    print(f"The Fermi energy is: {fermi_energy}..")

    return fermi_energy

def calculate_fermi_energy(T=1.0, B=ip.B, L=ip.L):
    Nl = ip.Nl
    vertices = ip.vertices
    Nr, Nphi = ip.Nr, ip.Nphi

    if vertices > 0:
        Nsing = 4*vertices
    else:
        Nsing = 24

    if Nr == 1:
        n_measure = ip.n_InAs_2D
    else:
        n_measure = ip.n_InAs_3D

    '''
    if L == 'inf':
        k, eigvals = calc_dispersion(B)
        Nk = len(k)
        #E_min, E_max = eigvals[int(Nk/2)][0], eigvals[int(Nk/2)][Nsing]
        E_min, E_max = np.min(np.stack(eigvals).flatten()), eigvals[int(Nk/2)][Nsing]
    else:
        with concurrent.futures.ProcessPoolExecutor() as pool:
            h_futures = pool.map(Hmat, range(Nl))
        h_list = [future for future in h_futures]

        eigvals, eigvecs = np.zeros(Nl, dtype=object), np.zeros(Nl, dtype=object)
        for l in range(Nl):
            eigvals[l], eigvecs[l], info = la.lapack.zheevd(h_list[l])
        E_min, E_max = np.min(np.stack(eigvals).flatten()), round(eigvals[0][Nsing])
    #print(E_min, E_max)
    '''

    suffix = f"{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.B}_{ip.alphaR}_{ip.betaD}_{ip.mag_direction}"

    with np.load(f"/scratch1/rperrin/" + suffix + "_sparse.npz", allow_pickle=True) as file:
        k = file['arr_0']
        Nk = len(k)
        eigvals = file['arr_1']
        Nstat = len(eigvals[0])

    E_min, E_max = np.min(np.stack(eigvals).flatten()), eigvals[int(Nk/2)][Nsing]
    E_div = int(5*(E_max-E_min))

    def func(mu):
        if L == 'inf':
            return particle_density(k, eigvals, mu, T) - n_measure
        else:
            return finite_wire_particle_density(eigvals, mu, T) - n_measure

    #solution = fsolve(func, (E_max+E_min)/2, maxfev=1500)
    solution = fsolve(func, 300, maxfev=1500)
    #solution = newton_krylov(func, (E_max-E_min)/2)
    #solution = anderson(func, (E_max-E_min)/2)

    return solution[0]

def create_fermi_energy_file(T=1.0, B=ip.B, L=ip.L):
    line_number = None
    input1 = None
    filename = "fermi_energies"
    current_calc_name = f"{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.alphaR}_{ip.betaD}"

    fermi_energy = calculate_fermi_energy(T, B, L)

    #with open(f"./output_files/fermi_energy/FE_{current_calc_name}.txt", 'w') as file:
    #    file.write(str(fermi_energy))

    with open(f"/scratch/rperrin/rperrin/fermi_energy/FE_{current_calc_name}.txt", 'w') as file:
        file.write(str(fermi_energy))

def main():
    shape, Nr, Nphi, R, vertices, mag_direction, alphaR = ip.shape, ip.Nr, ip.Nphi, ip.R, ip.vertices, ip.mag_direction, ip.alphaR

    start = time.time()

    T = 1.0

    create_fermi_energy_file(T=T, B=0.0)

    #fermi_energy = calculate_fermi_energy(T=T,B=0.0)
    #fermi_energy = 11.95
    #print(fermi_energy)
    #chem_potential = np.linspace(0, 350, 350)

    #k, eigvals = calc_dispersion(B=0.0)
    #ndens = [particle_density(k, eigvals, mu, 1.0) for mu in chem_potential]
    #plt.plot(chem_potential, ndens)
    #plt.axhline(y=ip.n_InAs_3D)
    #plt.savefig(f"./3D_fermi_test.png")

    #Nk = len(k)
    #Nstat = len(eigvals[0])

    #print(np.trace(abs(eigvecs[0])**2))

    #d_matrix = density_matrix(k, eigvals, eigvecs, fermi_energy, T)
    #d_matrix = d_matrix@d_matrix
    #[print(np.trace(d_matrix[ik])) for ik in range(Nk)]
    #for ik in range(Nk):
    #    print(np.trace(d_matrix[ik]))

    print(time.time()-start)

if __name__ == "__main__":
    main()






