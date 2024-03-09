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
###########################################
# Define the Pauli Matrices in a Kronecker Delta sort of way
def SigMat(type, s1, s2, phi):                    # Pauli Matrices
    cdef double complex SigMat
    # Sigma x
    if (type == 'x'):
        if s1 == s2:
            SigMat = 0.0 + 0.0j                   # s1=s2=1 or s1=s2=-1
        else:
            SigMat = 1.0 + 0.0j                   # s1=-s2=1 or -s1=s2=1

    # Sigma y
    if (type == 'y'):
        if s1 == s2:
            SigMat = 0.0 + 0.0j                   # s1=s2=1 or s1=s2=-1
        else:
            SigMat = 0.0 + s2*1.0j                # s1=-s2=1 or -s1=s2=1

    # Sigma z
    if (type == 'z'):
        if s1 == s2:
            SigMat = s2*(1.0 + 0.0j)              # s1=s2=1 or s1=s2=-1
        else:
            SigMat = s2*(0.0 + 0.0j)              # s1=-s2=1 or -s1=s2=1

    # Sigma r = Sigma x Cos(phi) + Sigma y Sin(phi)
    if (type == 'r'):
        if s1 == s2:
            SigMat = 0.0 + 0.0j                   # s1=s2=1 or s1=s2=-1
        else:
            SigMat = exp(s2*phi*1.0j)          # s1=-s2=1 or -s1=s2=1

    # Sigma p = - Sigma x Sin(phi) + Sigma y Cos(phi)
    if (type == 'p'):
        if s1 == s2:
            SigMat = 0.0 + 0.0j                   # s1=s2=1 or s1=s2=-1
        else:
            SigMat = s2*1.0j*exp(s2*phi*1.0j)  # s1=-s2=1 or -s1=s2=1

    # Identity matrix
    if (type == '1'):
        if s1 == s2:
            SigMat = 1.0 + 0.0j                   # s1=s2=1 or s1=s2=-1
        else:
            SigMat = 0.0 + 0.0j                   # s1=-s2=1 or -s1=s2=1

    return SigMat

# Defining the quantum numbers, ie lattice basis as single QN (as opposed to 2)
# and spin polarization
def quantum_numbers(Nr, Nphi):
    cdef int Nsites
    cdef int Nstat
    cdef numpy.ndarray qn
    # State index a = {k, s} = {ij, s}
    cdef int a
    # Lattice Site index, k = {ij}
    cdef int k
    # Spin polarization
    cdef int spin

    Nsites = Nr*Nphi
    Nstat = 2*Nsites

    # Array for storing the QNs
    qn = np.zeros([Nstat, 2])
    # State QN
    a = 0

    # Loop through all sites and spin polarization for each site
    for k in np.arange(1, Nsites+1, 1, dtype=int):
        for spin in np.arange(1,-2,-2, dtype=int):
            a+=1
            # Store lattice site index first element
            qn[a-1,0] = k
            # Store spin polarization in second element
            qn[a-1,1] = spin

    return qn

# Using vertex Numbers
def Hmat(k=ip.k, E=ip.E, B=ip.B, alphaR=ip.alphaR, betaD=ip.betaD, n=0, vertices=ip.vertices, t=ip.t, mag_dir=ip.mag_direction):
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
        H = np.zeros([Nstat, Nstat], dtype=np.csingle)
    else:
        if Nr>1:
            pos, Nstat = lattice.prismatic_nanotube(vertices, t)
            # If ring with thiccness, set differential radius for circular geometry
            d_r = (Rext*(1-cos(pi/vertices)) + t)/(Nr-1)
        else:
            pos, Nstat = lattice.prismatic_nanotube_1D(vertices)
        H = np.zeros([Nstat,Nstat], dtype=np.csingle)

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

        # Diagonal term with quadratic k term, important for k-slices and k-structure
        if L == "inf":
            #H[a1-1,a1-1] = H[a1-1,a1-1] + ts*tk + 2*ts*tphi + E*r1*cos(phi1)/Rext
            #H[a1-1,a1-1] = H[a1-1,a1-1] + tk + 2*tphi + E*r1/Rext
            H[a1-1,a1-1] = H[a1-1,a1-1] + tk + 2*tphi + E*r1/Rext*sin(phi1)
            #H[a1-1,a1-1] = H[a1-1,a1-1] + tk + 2*tphi + E*r1/Rext/sqrt(4)*(sin(phi1-pi/4)+sin(phi1+pi/4)+sin(phi1+3*pi/4)+sin(phi1-3*pi/4))
            #H[a1-1,a1-1] = H[a1-1,a1-1] + tk + 2*tphi + E*r1/Rext/sqrt(6)*(sin(phi1)+sin(phi1+pi/6)+sin(phi1+pi/3)+sin(phi1+pi/2)+sin(phi1+2*pi/3)+sin(phi1+5*pi/6))
            #H[a1-1,a1-1] = H[a1-1,a1-1] + tk + 2*tphi - alphaR/rc*r1/Rext
            #H[a1-1,a1-1] = H[a1-1,a1-1] + tk + 2*tphi
        else:
            H[a1-1,a1-1] = H[a1-1,a1-1] + 2*ts*tphi + E*r1*cos(phi1)/Rext + ts*(n*pi*R/L)**2
        if mag_dir == "z":
            H[a1-1,a1-1] = H[a1-1,a1-1] + ts*(tB*B*r1/4/Rext)**2 + gamma*ts*tB*B*spin1/2
            #H[a1-1,a1-1] = H[a1-1,a1-1] + (r1*R**2/2/ip.lB**2)**2 - gamma*ts*R**2/ip.lB**2*spin1/2
        if mag_dir == "x":
            if L == "inf":
                H[a1-1,a1-1] = H[a1-1,a1-1] + ts*(tB*B*r1/2/Rext*sin(phi1))**2 - tB*ts*sqrt(tk)*B*sin(phi1)*r_scaled
            else:
                H[a1-1,a1-1] = H[a1-1,a1-1] + ts*(tB*B*r1/2/Rext*sin(phi1))**2 - tB*ts*n*pi*R/L*B*sin(phi1)*r_scaled
        if mag_dir == "xy":
            if L == "inf":
                H[a1-1,a1-1] = H[a1-1,a1-1] + ts*(tB*B*r1/2/Rext*(sin(pi/vertices)*sin(phi1) - cos(pi/vertices)*cos(phi1)))**2 - tB*ts*sqrt(tk)*B*(sin(pi/vertices)*sin(phi1) - sin(pi/vertices)*cos(phi1))*r_scaled/sqrt(2)
            else:
                H[a1-1,a1-1] = H[a1-1,a1-1] + ts*(tB*B*r1/2/Rext*(sin(pi/vertices)*sin(phi1) - cos(pi/vertices)*cos(phi1)))**2 - tB*ts*n*pi*R/L*B*(sin(pi/vertices)*sin(phi1) - sin(pi/vertices)*cos(phi1))*r_scaled/sqrt(2)
            #H[a1-1,a1-1] = H[a1-1,a1-1] + ts*(tB*B*r1/2/Rext*sin(phi1))**2 + tB*ts*sqrt(tk)*B*sin(phi1)*r_scaled
        #if ip.mag_direction == "arbitrary":
        #    H[a1-1,a1-1] = H[a1-1,a1-1] + tk + 2*ts*tphi + ts*(tB*B*r1/4/Rext)**2 + ts*(tB*B*r1/2/Rext*sin(phi1))**2 + gamma*ts*tB*B*spin1/2 + E*r1*cos(phi1)/Rext

        # Loop through all states, for creating a full matrix (off-diagonal elements)
        # Assign all QNs and positions of each site
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

            # Perpendicular Zeeman
            #if ip.mag_direction == "perpendicular" or ip.mag_direction == "arbitrary":
            if mag_dir == "x":
                if (n1 == n2 and j1 == j2):
                    H[a1-1,a2-1] = H[a1-1,a2-1] - gamma*ts*tB*B*SigMat('x',spin1,spin2,phi1)/2

            if mag_dir == "xy":
                if (n1 == n2 and j1 == j2):
                    H[a1-1,a2-1] = H[a1-1,a2-1] - gamma*ts*tB*B*(cos(pi/vertices)*SigMat('x',spin1,spin2,phi1) + sin(pi/vertices)*SigMat('y',spin1,spin2,phi1))/2

            # Kinetic Radial and Azimuthal hopping happens between same spin states
            if (spin1==spin2):
                # Aziumthal hopping
                if (n1==n2):
                    #if (jj==1 or jj==(Nphi-1)): H[a1-1,a2-1]=H[a1-1,a2-1]-ts*tphi
                    if (jj==1 or jj==(Nphi-1)): H[a1-1,a2-1]=H[a1-1,a2-1]-tphi
                if (Nr>1):
                    # Scaled energy unit
                    #tr = (Rext/d_r)**2
                    tr = ts*(Rext/d_r)**2
                    # Radial Site potential term
                    #if (jj==0 and nn==0): H[a1-1,a2-1]=H[a1-1,a2-1]+2*ts*tr
                    if (jj==0 and nn==0): H[a1-1,a2-1]=H[a1-1,a2-1]+2*tr
                    # Radial Hopping term
                    #if (jj==0 and nn==1): H[a1-1,a2-1]=H[a1-1,a2-1]-ts*tr
                    if (jj==0 and nn==1): H[a1-1,a2-1]=H[a1-1,a2-1]-tr

            ################### Rashba #################
            term = 0
            if (n1 == n2 and (jj == 1 or jj == (Nphi-1))):
                # This term should not be under hopping condition
                #term = ts*alphaR*sqrt(tk)*SigMat('p', spin1, spin2, phi1)
                #H[a1-1, a2-1] = H[a1-1, a2-1] + term

                semn = -1
                if ((jj == 1 and j1 > j2) or (j1 == 1 and j2 == Nphi)): semn = 1
                #term = (alphaR + rc*E)*SigMat('z', spin1, spin2, phi1)
                #term = (alphaR + rc*E/sqrt(4)*(sin(phi1-pi/4)+sin(phi1+pi/4)+sin(phi1+3*pi/4)+sin(phi1-3*pi/4)))*SigMat('z', spin1, spin2, phi1)
                #term = (alphaR + rc*E/sqrt(6)*(sin(phi1)+sin(phi1+pi/6)+sin(phi1+pi/3)+sin(phi1+pi/2)+sin(phi1+2*pi/3)+sin(phi1+5*pi/6)))*SigMat('z', spin1, spin2, phi1)
                term = (alphaR + rc*E*sin(phi1))*SigMat('z', spin1, spin2, phi1)
                #term = alphaR*SigMat('z', spin1, spin2, phi1)
                term = ts*term*1.0j*sqrt(tphi)/2
                H[a1-1, a2-1] = H[a1-1,a2-1] + term*semn

                # Magnetic Potential Term, kinetic energy **Not part of RSOI**, but simplifies condition nesting
                #if ip.mag_direction == "axial" or ip.mag_direction == "arbitrary":
                if mag_dir == "z":
                    #if spin1 == spin2: H[a1-1,a2-1]=H[a1-1,a2-1]-0.25*1.0j*ts*tB*B/d_phi*semn
                    if spin1 == spin2: H[a1-1,a2-1]=H[a1-1,a2-1]+0.25*1.0j*ts*tB*B/d_phi*semn

                term = 0
                #if (n1 == n2 and j1 == j2):

            if (nn == 1 and jj == 0):
                term = -rc*E*cos(phi1)*SigMat('z', spin1, spin2, phi1)
                #term = -rc*E/sqrt(4)*(cos(phi1-pi/4)+cos(phi1+pi/4)+cos(phi1+3*pi/4)+cos(phi1-3*pi/4))*SigMat('z', spin1, spin2, phi1)
                #term = -rc*E/sqrt(6)*(cos(phi1)+cos(phi1+pi/6)+cos(phi1+pi/3)+cos(phi1+pi/2)+cos(phi1+2*pi/3)+cos(phi1+5*pi/6))*SigMat('z', spin1, spin2, phi1)
                term = ts*term/r1/2*1.0j
                H[a1-1,a2-1] = H[a1-1,a2-1] + term
            
            if (nn == 0 and jj == 0):
                #term = (alphaR + rc*E)*SigMat('p', spin1, spin2, phi1)
                term = (alphaR + rc*E*sin(phi1))*SigMat('p', spin1, spin2, phi1)
                #term = (alphaR + rc*E/sqrt(6)*(sin(phi1)+sin(phi1+pi/6)+sin(phi1+pi/3)+sin(phi1+pi/2)+sin(phi1+2*pi/3)+sin(phi1+5*pi/6)))*SigMat('p', spin1, spin2, phi1)
                #term = (alphaR + rc*E/sqrt(4)*(sin(phi1-pi/4)+sin(phi1+pi/4)+sin(phi1+3*pi/4)+sin(phi1-3*pi/4)))*SigMat('p', spin1, spin2, phi1)
                term += rc*E*cos(phi1)*SigMat('r',spin1,spin2,phi1)
                #term += rc*E/sqrt(4)*(cos(phi1-pi/4)+cos(phi1+pi/4)+cos(phi1+3*pi/4)+cos(phi1-3*pi/4))*SigMat('r',spin1,spin2,phi1)
                #term += rc*E/sqrt(6)*(cos(phi1)+cos(phi1+pi/6)+cos(phi1+pi/3)+cos(phi1+pi/2)+cos(phi1+2*pi/3)+cos(phi1+5*pi/6))*SigMat('r',spin1,spin2,phi1)
                #term = (alphaR + rc*E*cos(phi1))*SigMat('p', spin1, spin2, phi1)
                #term = alphaR*SigMat('p', spin1, spin2, phi1)
                term = ts*term*sqrt(tk)
                H[a1-1,a2-1] = H[a1-1,a2-1]+term

                if mag_dir == "z":
                    term = 0
                    term += alphaR*SigMat('z', spin1, spin2, phi1)
                    term = term*tB*ts*B*r_scaled/4
                    H[a1-1, a2-1] = H[a1-1, a2-1] + term
                # x and xy will need additional finite z term..
                if mag_dir == "x":
                    term = 0
                    term -= alphaR*SigMat('p',spin1,spin2,phi1)
                    term = term*tB*ts*B*r_scaled/2*sin(phi1)
                    H[a1-1,a2-1] = H[a1-1,a2-1] + term
                if mag_dir == "xy":
                    term = 0
                    term -= alphaR*SigMat('p',spin1,spin2,phi1)
                    term = term*tB*ts*B*r_scaled/2*(sin(pi/vertices)*sin(phi1) - cos(pi/vertices)*cos(phi1))/sqrt(2)
                    H[a1-1,a2-1] = H[a1-1,a2-1] + term

            ################### Manolescu DSOI #####################
            if ip.SOI_definition == 'manolescu':
                term = 0
                # Azimuthal Hopping Term
                if (n1 == n2 and (jj == 1 or jj == (Nphi-1))):
                    semn = -1
                    if ((jj == 1 and j1 > j2) or (j1 == 1 and j2 == Nphi)): semn = 1
                    term = term - betaD*(SigMat('p',spin1,spin2,phi1) + SigMat('p',spin1,spin2,phi2))/4
                    term = term*1.0j*ts*sqrt(tphi)/2
                    H[a1-1,a2-1] = H[a1-1,a2-1] + term*semn

                term = 0
                if (n1 == n2 and j1 == j2):
                    # Longitudinal
                    term -= betaD*SigMat('z',spin1,spin2,phi1)
                    if L == "inf":
                        term = term*sqrt(tk)*ts # Check scaling, seems off but is likely from alphaR scaling 1/ts..
                    else:
                        term = term*n*pi*R/L*ts
                    H[a1-1,a2-1] = H[a1-1,a2-1] + term

                    # Site potential (?) from DSOI anti-commutator
                    term = 0
                    term = SigMat('r',spin1,spin2,phi1)
                    term = term*1.0j*betaD*ts*Rext/2/r1
                    H[a1-1,a2-1] = H[a1-1,a2-1] + term

                    # Magnetic Potential term
                    #if ip.mag_direction == "axial" or ip.mag_direction == "arbitrary":
                    if mag_dir == "z":
                        term = 0
                        term = -betaD*SigMat('p',spin1,spin2,phi1)
                        term = term*tB*ts*B*r_scaled/4
                        H[a1-1,a2-1] = H[a1-1,a2-1] + term
                    # x and xy will need additional finite z term..
                    if mag_dir == "x":
                        term = 0
                        term = betaD*SigMat('z',spin1,spin2,phi1)
                        term = term*tB*ts*B*r_scaled/2*sin(phi1)
                        H[a1-1,a2-1] = H[a1-1,a2-1] + term
                    if mag_dir == "xy":
                        term = 0
                        term = betaD*SigMat('z',spin1,spin2,phi1)
                        term = term*tB*ts*B*r_scaled/2*(sin(pi/vertices)*sin(phi1) - cos(pi/vertices)*cos(phi1))/sqrt(2)
                        H[a1-1,a2-1] = H[a1-1,a2-1] + term

            ###################### Kokurin DSOI [111] ################################################
            if ip.SOI_definition == 'kokurin' or ip.SOI_definition == '111':
                term1, term2, term3, term4, term5, term6, term7 = 0, 0, 0, 0, 0, 0, 0
                # Dresselhaus SOI / begin
                # Angular DSOI  / begin
                if nn == 0:
                    if (jj == 1 or jj == (Nphi - 1)):
                        semn = -1
                        if ((jj == 1 and j1 > j2) or (j1 == 1 and j2 == Nphi)): semn = 1
                        term1 = (SigMat('r', spin1, spin2, phi1) + SigMat('r', spin1, spin2, phi2))/2
                        term1 = ts*1.0j*term1*betaD/8/sqrt(3)*sqrt(tphi)
                        # Term 2 should be uncommented. Not sure why it is not.
                        term2 = -(sin(3*phi1) + sin(3*phi2))/2
                        term2 = ts*1.0j*betaD*sqrt(tphi)*term2*sqrt(3/2)/2*SigMat('z', spin1, spin2, phi1)
                        H[a1-1, a2-1] = H[a1-1,a2-1] + semn*(term1 + term2)
                    # Site Potential DSOI / begin
                    if jj == 0:
                        #term3 = -3.0j*ts*betaD*sqrt(3/2)*Rext/r1*SigMat('z',spin1,spin2,phi1)*cos(3*phi1)/2
                        #term4 = 1.0j*ts*betaD/4/sqrt(3)*Rext/r1*SigMat('p',spin1,spin2,phi1)
                        # Longitudinal DSOI
                        term5 = ts*betaD/sqrt(6)*sqrt(tk)*SigMat('r',spin1,spin2,-2*phi1)
                        # Magnetic DSOI
                        if mag_dir == "z":
                            #term6 = -ts*betaD*tB*B/8/Rext/sqrt(3)*r1*SigMat('r',spin1,spin2,phi1)
                            #term7 = ts*betaD*tB*B*sqrt(3/2)*sin(3*phi1)*SigMat('z',spin1,spin2,phi1)*r1/4/Rext
                            term6 = ts*betaD*tB*B/8/Rext/sqrt(3)*r1*SigMat('r',spin1,spin2,phi1)
                            term7 = -ts*betaD*tB*B*sqrt(3/2)*sin(3*phi1)*SigMat('z',spin1,spin2,phi1)*r1/4/Rext
                            H[a1-1,a2-1] = H[a1-1,a2-1] + term3 + term4 + term5 + term6 + term7
                        if mag_dir == "x":
                            #term6 = ts*betaD/sqrt(6)*tB*B*r1/Rext*sin(phi1)*SigMat('r',spin1,spin2,-2*phi1)
                            term6 = -ts*betaD/sqrt(6)*tB*B*r1/Rext*sin(phi1)*SigMat('r',spin1,spin2,-2*phi1)
                            H[a1-1,a2-1] = H[a1-1,a2-1] + term3 + term4 + term5 + term6 + term7
                        if mag_dir == "xy":
                            #term6 = ts*betaD/sqrt(6)*tB*B*r1/Rext*sin(phi1)*SigMat('r',spin1,spin2,-2*phi1)
                            term6 = -ts*betaD/sqrt(12)*tB*B*r1/Rext*(sin(pi/vertices)*sin(phi1) - cos(pi/vertices)*cos(phi1))*SigMat('r',spin1,spin2,-2*phi1)
                            H[a1-1,a2-1] = H[a1-1,a2-1] + term3 + term4 + term5 + term6 + term7
                # Radial DSOI / begin
                if jj == 0 and nn == 1:
                    # Using the hopping energy is too strong for term4
                    # Using <kr> = i/(2r) works, however
                    #term3 = -3.0j*ts*betaD*sqrt(3/2)*sqrt(tr)*SigMat('z',spin1,spin2,phi1)*cos(3*phi1)/2
                    term3 = -3.0j*ts*betaD*sqrt(3/2)/2/r1*SigMat('z',spin1,spin2,phi1)*cos(3*phi1)/2
                    #term4 = -1.0j*ts*betaD/4/sqrt(3)*sqrt(tr)*SigMat('p',spin1,spin2,phi1)
                    term4 = 1.0j*ts*betaD/4/sqrt(3)/r1*SigMat('p',spin1,spin2,phi1)
                    #H[a1-1,a2-1] = H[a1-1,a2-1] + term3 + term4
                    H[a1-1,a2-1] = H[a1-1,a2-1] + term3

                # Dresslehaus SOI / end

            ########################## DSOI [001] ##########################
            if ip.SOI_definition == '001':
                term1, term2 = 0, 0
                # Dresselhaus SOI / begin
                # Angular DSOI  / begin
                if n1 == n2:
                    if (jj == 1 or jj == (Nphi - 1)):
                        term1, term2 = 0, 0
                        semn = -1
                        if ((jj == 1 and j1 > j2) or (j1 == 1 and j2 == Nphi)): semn = 1
                        ####### Radial Spin ########
                        term1 = SigMat('r', spin1, spin2, phi1) + SigMat('r', spin1, spin2, phi2)
                        term1 = term1*(sin(2*phi1) + sin(2*phi2))
                        term1 = -1.0j*term1*betaD*sqrt(tphi)/16
                        ###### Azimuthal Spin ######
                        term2 = SigMat('p', spin1, spin2, phi1) + SigMat('p', spin1, spin2, phi2)
                        term2 = term2*(cos(2*phi1) + cos(2*phi2))
                        term2 = -1.0j*betaD*sqrt(tphi)*term2/8

                        H[a1-1, a2-1] = H[a1-1,a2-1] + ts*semn*(term1 - term2)

                    # Site Potenial DSOI / begin
                    if j1 == j2:
                        # Longitudinal DSOI
                        term1 = 0
                        term1 = betaD*sqrt(tk)*SigMat('z',spin1,spin2,phi1)*cos(2*phi1)
                        H[a1-1,a2-1] = H[a1-1,a2-1] + ts*term1

                if jj == 0 and nn == 1:
                    term1 = 0
                    term1 = 2*SigMat('r', spin1, spin2, phi1)*cos(2*phi1) + 5/2*SigMat('p', spin1, spin2, phi1)*sin(2*phi1)
                    term1 = sqrt(tr)*betaD*1.0j*term1/2
                    H[a1-1, a2-1] = H[a1-1,a2-1] + ts*term1
                # Dresslehaus SOI / end

            # Not necessary if using 'U' triang
            H[a2-1,a1-1] = np.conjugate(H[a1-1,a2-1])
    return H
