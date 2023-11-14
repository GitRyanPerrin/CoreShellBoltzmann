############################################
# Standard Library Modules
import time
from itertools import repeat
import gc
import sys
import os

# Third Party Modules
import numpy as np
np.get_include()
#np.set_printoptions(threshold=sys.maxsize)
from numpy import pi, exp, cos, sin, sqrt, tan
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
from scipy import linalg as la
from scipy.special import expit
cimport numpy

# Local Modules
import input_data as ip
import lattice
from hamiltonian import quantum_numbers, SigMat

# Using vertex Numbers
def Vmat(k=ip.k, B=ip.B, n=0, E=ip.E, vertices=ip.vertices, t=ip.t, mag_dir = ip.mag_direction, alphaR = ip.alphaR, betaD = ip.betaD):
    Nr, Nphi, R, L, Rext, ts, gamma = ip.Nr, ip.Nphi, ip.R, ip.L, ip.Rext, ip.ts, ip.gamma
    rc = ip.rashba_coeff

    # differential radius, changes per geometry
    d_r = 0

    # Create quantum numbers, only length changes with geometry, but doesn't affect outcome
    qn = quantum_numbers(Nr, Nphi)

    # Create empty H array and other parameters based on geometry
    if vertices == 0:
        #Nr, Nphi = 10, 100
        pos, Nstat = lattice.nanotube.circular(Nr, Nphi, Rext, t)
        # If ring with thiccness, set differential radius for circular geometry
        if (Nr>1): d_r = t/(Nr-1)
        v = np.zeros([Nstat, Nstat], dtype=np.csingle)
    else:
        if Nr>1:
            pos, Nstat = lattice.prismatic_nanotube(vertices, t)
            # If ring with thiccness, set differential radius for circular geometry
            d_r = (Rext*(1-cos(pi/vertices)) + t)/(Nr-1)
        else:
            pos, Nstat = lattice.prismatic_nanotube_1D(vertices)
        v = np.zeros([Nstat,Nstat], dtype=np.csingle)

    # Differntial aziumthal angle does not change based on geometry
    d_phi = 2*pi/Nphi
    tB = 2*R**2/ip.hbar/10**(15)
    #tk = (k*R)**2
    tk = ts*(k*R)**2

    # Loop through all states and separate QNs (site index and spin polarization) and positions of each site index
    for a1 in np.arange(1, Nstat+1, 1, dtype=int):
        k1 = int(qn[a1-1,0])
        spin1 = int(qn[a1-1,1])
        r1 = pos[k1-1,0]
        r_scaled = r1/Rext
        phi1 = pos[k1-1,1]
        # For ring w/o thiccness
        # n1 is the integer of the radial site
        # if left with more precision, rounding can occur
        # and may throw off the comparisons later
        if d_r == 0:
            n1 = 1
        else:
            n1 = int(round((Rext-r1)/d_r+1))

        # integer of the azimuthal site
        # rounded for same reason as n1
        j1 = int(round(phi1/d_phi)+1)

        # scaled energy unit
        #tphi = (Rext/r1/d_phi)**2
        tphi = ts*(Rext/r1/d_phi)**2

        v[a1-1,a1-1] = v[a1-1,a1-1] + ts*k*R**2

        for a2 in np.arange(a1, Nstat+1, 1, dtype=int):
            k2 = int(qn[a2-1,0])
            spin2 = int(qn[a2-1,1])
            r2 = pos[k2-1,0]
            phi2 = pos[k2-1,1]
            
            if (d_r == 0):
                n2=1
            else:
                n2 = int(round((Rext-r2)/d_r+1))
            j2 = int(round(phi2/d_phi)+1)
            #if (d_r == 0): n2=1

            # Difference between integer position values
            # Will be used to check if on same site or one away
            nn = abs(n1-n2)
            jj = abs(j1-j2)

            ################### Rashba #################
            term = 0
            if (nn == 0 and jj == 0):
                term = alphaR*SigMat('p', spin1, spin2, phi1)
                term = ts*term*sqrt(ts)*R
                v[a1-1,a2-1] = v[a1-1,a2-1]+term

            ###################### Kokurin DSOI [111] ################################################
            if ip.SOI_definition == 'kokurin' or ip.SOI_definition == '111':
                term1, term2, term3, term4, term5, term6, term7 = 0, 0, 0, 0, 0, 0, 0
                # Dresselhaus SOI / begin
                # Angular DSOI  / begin
                if nn == 0 and jj == 0:
                    # Site Potential DSOI / begin
                    #term3 = -3.0j*ts*betaD*sqrt(3/2)*Rext/r1*SigMat('z',spin1,spin2,phi1)*cos(3*phi1)/2
                    #term4 = 1.0j*ts*betaD/4/sqrt(3)*Rext/r1*SigMat('p',spin1,spin2,phi1)
                    # Longitudinal DSOI
                    #term5 = ts*betaD/sqrt(6)*sqrt(tk)*SigMat('r',spin1,spin2,-2*phi1)
                    # Magnetic DSOI
                    v[a1-1,a2-1] = v[a1-1,a2-1] + term5

                # Dresslehaus SOI / end

            ########################## DSOI [001] ##########################
            if ip.SOI_definition == '001':
                term1, term2 = 0, 0
                # Dresselhaus SOI / begin
                # Angular DSOI  / begin
                if n1 == n2 and j1 == j2:
                    # Site Potenial DSOI / begin
                    # Longitudinal DSOI
                    term1 = 0
                    #term1 = betaD*sqrt(tk)*SigMat('z',spin1,spin2,phi1)*cos(2*phi1)
                    v[a1-1,a2-1] = v[a1-1,a2-1] + ts*term1

            # Not necessary if using 'U' triang
            v[a2-1,a1-1] = np.conjugate(v[a1-1,a2-1])
    return v
