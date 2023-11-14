'''
This code defines the Pauli spin matrices on the grid. The Hilbert space is expanded
from just a 2x2 space because the Pauli matrices in cylindrical coordinates depend
on the azimuthal position. They are unseparable in this case.

This was intended to be used in the anticommutation relation for the spin-velocity.
'''

import numpy as np
from numpy import sqrt, pi, cos, sin, exp
import matplotlib.pyplot as plt

import input_data as ip
from hamiltonian import quantum_numbers, SigMat
from lattice import nanotube, prismatic_nanotube

def Sijs(polarization, factor=1):

    # Get geometry of polar grid
    Nr, Nphi = ip.Nr, ip.Nphi
    Rext, t = ip.Rext, ip.t
    vertices = ip.vertices

    if ip.vertices == 0:
        pos, Nstat = nanotube.circular(Nr, Nphi, Rext, t)
        d_r = t/(Nr-1)
    else:
        pos, Nstat = prismatic_nanotube(vertices)
        d_r = (Rext*(1-cos(pi/vertices))+t)/(Nr-1)
    d_phi = 2*pi/Nphi

    # Get quantum numbers
    qn = quantum_numbers(Nr, Nphi)

    # output array
    sigma_ijs = np.zeros([Nstat, Nstat], dtype=complex)

    for a1 in range(Nstat):
        q1 = int(qn[a1,0])
        s1 = qn[a1,1]
        r1 = pos[q1-1, 0]
        phi1 = pos[q1-1, 1]

        i1 = int((Rext-r1)/d_r+1)
        j1 = int(phi1/d_phi)+1

        for a2 in range(Nstat):
            q2 = int(qn[a2,0])
            s2 = qn[a2,1]
            r2 = pos[q2-1, 0]
            phi2 = pos[q2-1, 1]

            i2 = int((Rext-r2)/d_r+1)
            j2 = int(phi2/d_phi)+1

            ii = abs(i1-i2)
            jj = abs(j1-j2)

            # This condition ensures continuity along azimuthal direction
            if ii == 0 and (jj == 0 or jj == (Nphi-1)):
                sigma_ijs[a1, a2] = SigMat(polarization, s1, s2, factor*phi1)

            sigma_ijs[a2,a1] = sigma_ijs[a1,a2].conj()

    return sigma_ijs
