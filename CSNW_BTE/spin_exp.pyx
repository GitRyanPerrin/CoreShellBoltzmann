import gc

import numpy as np
cimport numpy as np
from numpy import pi, exp, cos, sin, sqrt
import concurrent.futures
import matplotlib.pyplot as plt
from scipy import linalg as la
from scipy import integrate

import input_data as ip
from lattice import prismatic_nanotube, nanotube
from hamiltonian import Hmat, SigMat, quantum_numbers
from fermi_energy import FD

def spin_matrix(type, factor=1.0):

    Nr, Nphi, Rext, t, vertices = ip.Nr, ip.Nphi, ip.Rext, ip.t, ip.vertices
    qn = quantum_numbers(Nr, Nphi)
    if vertices == 0:
        pos, Nstat = nanotube.circular(Nr, Nphi, Rext, t)
    else:
        pos, Nstat = prismatic_nanotube(vertices, t)

    smat = np.zeros([Nstat, Nstat], dtype=complex)
    for a1 in range(Nstat):
        q1 = int(qn[a1,0])
        s1 = int(qn[a1,1])
        phi1 = pos[q1-1, 1]

        for a2 in range(Nstat):
            q2 = int(qn[a2,0])
            s2 = int(qn[a2,1])
            phi2 = pos[q1-2, 1]

            smat[a1, a2] = SigMat(type, s1, s2, factor*phi1)

        smat[a2, a1] = np.conjugate(smat[a1,a2])

    return smat

def spin_exp(type, a, eigvecs):
    Nr, Nphi, Rext, t, vertices = ip.Nr, ip.Nphi, ip.Rext, ip.t, ip.vertices
    qn = quantum_numbers(Nr, Nphi)
    if vertices == 0:
        pos, Nstat = nanotube.circular(Nr, Nphi, Rext, t)
    else:
        pos, Nstat = prismatic_nanotube(vertices, t)
    k = int(qn[a-1,0])
    s = qn[a-1, 1]
    phi = qn[k-1,1]
    expectation_value = abs(eigvecs[a,a])**2*SigMat(type, s, s, phi)
    return expectation_value

'''
def spin_texture(type, eigvecs):
    Nr, Nphi, Rext, t, vertices = ip.Nr, ip.Nphi, ip.Rext, ip.t, ip.vertices
    Nsites = Nr*Nphi
    qn = quantum_numbers(Nr, Nphi)
    if vertices == 0:
        pos, Nstat = lattice.nanotube.circular(Nr, Nphi, Rext, t)
    else:
        pos, Nstat = lattice.prismatic_nanotube(vertices, t)

    expectation_value = np.zeros([Nstat, Nsites])
    for a1 in range(Nstat):
        p1 = int(qn[a1-1,0])
        s1 = qn[a1-1, 1]
        phi1 = qn[p1-1,1]

        for a2 in range(Nstat):
            p2 = int(qn[a2-1,0])
            s2 = qn[a2-1, 1]

            expectation_value[a1, p2-1] += abs(eigvecs[a2,a1])**2*SigMat(type, s1, s2, phi1)

    return expectation_value
'''

def spin_density(type, k, eigvals, eigvecs, mu, T):
    Nr, Nphi, Rext, t, vertices = ip.Nr, ip.Nphi, ip.Rext, ip.t, ip.vertices
    Nsites = Nr*Nphi
    Nk = len(k)
    Nsing = 16
    qn = quantum_numbers(Nr, Nphi)
    if vertices == 0:
        pos, Nstat = nanotube.circular(Nr, Nphi, Rext, t)
    else:
        pos, Nstat = prismatic_nanotube(vertices, t)

    #expectation_value = np.zeros([Nstat, Nsites])
    #integrand = np.zeros(Nk)
    integrand = np.zeros([Nk, Nsites])
    for ik in range(Nk):
        for a1 in range(Nsing):
            p1 = int(qn[a1-1,0])
            s1 = qn[a1-1, 1]
            phi1 = qn[p1-1,1]

            for a2 in range(Nsing):
                p2 = int(qn[a2-1,0])
                s2 = qn[a2-1, 1]

                integrand[ik, p2-1] += FD(eigvals[ik][a1], mu, T)*abs(eigvecs[ik][a2,a1])**2*SigMat(type, s1, s2, phi1)

    sdens = integrate.simps(integrand, k, axis=0)/2/pi/ip.R

    return sdens

def spin_texture(eigvecs, pos, polarization):

    cdef int Nr = ip.Nr
    cdef int Nphi = ip.Nphi
    cdef int Nstat = np.shape(eigvecs[0])[1]
    cdef int Nsite = int(np.shape(eigvecs[0])[0]/2)
    cdef np.ndarray qn = quantum_numbers(Nr, Nphi)
    cdef np.ndarray sexp = np.zeros([Nstat, Nsite], dtype=complex)

    cdef double s1
    cdef int q2
    cdef double s2
    cdef double phi2

    for a1 in range(Nstat):
        s1 = qn[a1,1]
        for a2 in range(200):
            q2 = int(qn[a2,0])
            s2 = qn[a2,1]
            phi2 = pos[q2-1,1]

            # I think I acutally need eigvecs**2 times the spin orientation. But for now I will leave it here..
            sexp[a1, q2-1] += SigMat(polarization, s1, s2, phi2)*abs(eigvecs[0][a2,a1])**2

    return sexp

def energy_state_spin_exp(eigvecs, polarization, factor=1):

    cdef int Nr = ip.Nr
    cdef int Nphi = ip.Nphi
    cdef double Rext = ip.Rext
    cdef double t = ip.t
    cdef int vertices = ip.vertices
    cdef int Nk = np.shape(eigvecs)[0]

    cdef np.ndarray pos
    cdef int Nstat

    if vertices == 0:
        pos, Nstat = nanotube.circular(Nr, Nphi, Rext, t)
    else:
        pos, Nstat = prismatic_nanotube(vertices)

    # # single particle states may be lower than the total # states
    cdef int Nsing = np.shape(eigvecs[0])[1]
    cdef np.ndarray qn = quantum_numbers(Nr, Nphi)
    cdef np.ndarray sexp = np.zeros([Nk, Nsing], dtype=complex)
    #cdef np.ndarray sexp = np.zeros([Nk, Nstat], dtype=complex)

    for ik in range(Nk):
        for a1 in range(Nsing):
            q1 = int(qn[a1,0])
            s1 = qn[a1,1]
            r1 = pos[q1-1,0]
            phi1 = pos[q1-1,1]
            for a2 in range(Nsing):
                q2 = int(qn[a2,0])
                s2 = qn[a2,1]
                r2 = pos[q2-1,1]
                phi2 = pos[q2-1,1]

                if phi1 == phi2 and r1 == r2:
                    if polarization == 'p':
                        sexp[ik, a1] += (np.real(SigMat('x', s1, s2, factor*phi2))*np.cos(phi2) + np.imag(SigMat('y', s1, s2, factor*phi2)*np.sin(phi2)))*abs(eigvecs[ik][a2,a1])**2
                    else:
                        sexp[ik, a1] += SigMat(polarization, s1, s2, factor*phi2)*abs(eigvecs[ik][a2,a1])**2
                else:
                    continue

    #if polarization == 'x' or polarization == 'z':
    #    return np.real(sexp)
    #if polarization == 'y':
    #    return np.imag(sexp)
    #else:
    return sexp

def energy_state_trig_exp(eigvecs):

    cdef int Nr = ip.Nr
    cdef int Nphi = ip.Nphi
    cdef double Rext = ip.Rext
    cdef double t = ip.t
    cdef int vertices = ip.vertices
    cdef int Nk = np.shape(eigvecs)[0]

    cdef np.ndarray pos
    cdef int Nstat

    if vertices == 0:
        pos, Nstat = nanotube.circular(Nr, Nphi, Rext, t)
    else:
        pos, Nstat = prismatic_nanotube(vertices)

    # # single particle states may be lower than the total # states
    cdef int Nsing = np.shape(eigvecs[0])[1]
    cdef np.ndarray qn = quantum_numbers(Nr, Nphi)
    cdef np.ndarray sin_exp = np.zeros([Nk, Nsing], dtype=complex)
    cdef np.ndarray cos_exp = np.zeros([Nk, Nsing], dtype=complex)
    #cdef np.ndarray sexp = np.zeros([Nk, Nstat], dtype=complex)

    for ik in range(Nk):
        for a1 in range(Nsing):
            q1 = int(qn[a1,0])
            s1 = qn[a1,1]
            r1 = pos[q1-1,0]
            phi1 = pos[q1-1,1]
            for a2 in range(Nstat):
                q2 = int(qn[a2,0])
                s2 = qn[a2,1]
                r2 = pos[q2-1,1]
                phi2 = pos[q2-1,1]

                sin_exp[ik, a1] += np.sin(3*phi2)*abs(eigvecs[ik][a2,a1])**2
                cos_exp[ik, a1] += np.cos(3*phi2)*abs(eigvecs[ik][a2,a1])**2

                #if r1 == r2:
                #    sin_exp[ik, a1] += np.sin(3*phi2)*abs(eigvecs[ik][a2,a1])**2
                #    cos_exp[ik, a1] += np.cos(3*phi2)*abs(eigvecs[ik][a2,a1])**2

    return sin_exp, cos_exp

def no_k_trig_exp():

    cdef int Nr = ip.Nr
    cdef int Nphi = ip.Nphi
    cdef double Rext = ip.Rext
    cdef double t = ip.t
    cdef int vertices = ip.vertices

    eigvals, eigvecs = la.eigh(Hmat())

    cdef np.ndarray pos
    cdef int Nstat

    if vertices == 0:
        pos, Nstat = nanotube.circular(Nr, Nphi, Rext, t)
    else:
        pos, Nstat = prismatic_nanotube(vertices)

    # # single particle states may be lower than the total # states
    cdef np.ndarray qn = quantum_numbers(Nr, Nphi)
    cdef np.ndarray sin_exp = np.zeros(Nstat, dtype=complex)
    cdef np.ndarray cos_exp = np.zeros(Nstat, dtype=complex)
    #cdef np.ndarray sexp = np.zeros([Nk, Nstat], dtype=complex)

    for a1 in range(Nstat):
        q1 = int(qn[a1,0])
        for a2 in range(Nstat):
            q2 = int(qn[a2,0])
            phi2 = pos[q2-1,1]

            sin_exp[a1] += np.sin(3*phi2)*abs(eigvecs[a2,a1])**2/Nstat
            cos_exp[a1] += np.cos(3*phi2)*abs(eigvecs[a2,a1])**2/Nstat

    return sin_exp, cos_exp

def spin_matrix_exp(eigvecs, type, factor=1.0):

    Nk = len(eigvecs)
    eigvecs = np.stack(eigvecs, axis=0)
    Nsing = eigvecs.shape[-1]

    smat_ijs = spin_matrix(type, factor)
    smat_ak = np.stack([eigvecs[ik, :Nsing, :Nsing]@smat_ijs[:Nsing, :Nsing]@eigvecs[ik, :Nsing, :Nsing].conj().T for ik in range(Nk)], axis=2)

    sexp = np.sum(smat_ak, axis=0)

    return sexp
