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
import input_data as input
###########################################

class nanotube:

    def circular(Nr, Nphi, Rext, t):
        
        cdef int Nsites
        cdef int Nstat
        cdef numpy.ndarray pos
        cdef int k
        cdef double r
        cdef double d_r
        cdef double d_phi
        cdef double x
        cdef double y
        

        Rint = Rext - t

        Nsites = Nr*Nphi
        Nstat = 2*Nsites
        pos = np.zeros([Nsites, 4])

        k = 0
        r = Rext
        d_r = 0

        if (Nr>1): d_r = t/(Nr-1)
        d_phi = 2*pi/Nphi
        for ir in np.arange(1, Nr+1, 1, dtype=int):
            if (Nr>1): r = Rext-d_r*(ir-1)
            for iphi in np.arange(1, Nphi+1, 1, dtype=int):
                phi = d_phi*(iphi-1)
                k+=1
                pos[k-1,0] = r
                pos[k-1,1] = phi
                x = r*cos(phi)
                y = r*sin(phi)
                pos[k-1,2] = x
                pos[k-1,3] = y
                if (r==0):
                    break

        #print(f'Circular Nanotube has {Nsites} sites...')

        return pos, Nstat

def prismatic_nanotube(N, t=input.t):
    # N is the number of vertices/sides of the polygonal cross-section
    cdef int Nsites
    cdef int Nsites_geo
    cdef int Nstat
    cdef int Nstat_geo
    cdef numpy.ndarray pos
    cdef numpy.ndarray pos_geo
    cdef int k
    cdef double r
    cdef double d_r
    cdef double d_phi
    cdef double x
    cdef double y

    cdef double Rverts
    cdef double Rint

    Nr, Nphi, Rext = input.Nr, input.Nphi, input.Rext
    Xout = [Rext * cos(2*pi*n/N) for n in range(N+1)]
    Yout = [Rext * sin(2*pi*n/N) for n in range(N+1)]

    # For Odd N. Has issues with infinite slopes/intercepts
    #if N % 2 > 0:
    #    Xout = [Rext * sin(2*pi*n/N) for n in range(N+1)]
    #    Yout = [-Rext * cos(2*pi*n/N) for n in range(N+1)]

    Rverts = Rext - t/cos(pi/N)
    Rint = Rverts*cos(pi/N)

    Xin = [Rverts * cos(2*pi*n/N) for n in range(N+1)]
    Yin = [Rverts * sin(2*pi*n/N) for n in range(N+1)]

    slope_in = [(Yin[n+1]-Yin[n])/(Xin[n+1]-Xin[n]) for n in range(N)]
    intercept_in = [(Xin[n]*Yin[n+1] - Xin[n+1]*Yin[n])/(Xin[n]-Xin[n+1]) for n in range(N)]

    def inner_bound(x):
        for n in range(N):
            y = slope_in[n]*x+intercept_in[n]
        return sqrt(x**2 + y**2)

    def line_in(x, n):
        return slope_in[n]*x+intercept_in[n]

    slope_out = [(Yout[n+1]-Yout[n])/(Xout[n+1]-Xout[n]) for n in range(N)]
    intercept_out = [(Xout[n]*Yout[n+1] - Xout[n+1]*Yout[n])/(Xout[n]-Xout[n+1]) for n in range(N)]

    def outer_bound(x):
        for n in range(N):
            y = slope_out[n]*x+intercept_out[n]
        return sqrt(x**2 + y**2)

    def line_out(x, n):
        return slope_out[n]*x+intercept_out[n]


    x_lin = np.linspace(-1,1,500)

    Nsites = Nr*Nphi
    Nstat = 2*Nsites
    pos = np.zeros([Nsites, 6])
    Nsites_geo = 0

    k = 0
    r = Rext
    d_r = 0

    if (Nr>1): d_r = (Rext-Rint)/(Nr-1)
    d_phi = 2*pi/Nphi
    for ir in np.arange(0, Nr, 1, dtype=int):
        if (Nr>1): r = Rext-d_r*(ir-1)
        for iphi in np.arange(0, Nphi, 1, dtype=int):
            phi = d_phi*(iphi-1)
            x = r*cos(phi)
            y = r*sin(phi)
            k+=1
            conditions_out = []
            conditions_in = []
            for n in range(N):
                bound_out = sqrt(x**2 + line_out(x,n)**2)
                bound_in = sqrt(x**2 + line_in(x,n)**2)
                if round(cos(2*n*pi/N), 3) != round(cos(2*(n+1)*pi/N),3):
                    conditions_out.append(r <= bound_out)
                    conditions_in.append(r >= bound_in)
                else:
                    conditions_out.append(x >= -Rint - t)
                    conditions_in.append(x <= -Rint)
            if (Nr > 1 and (all(conditions_out) and any(conditions_in))) or (Nr > 1 and (abs(r) > Rverts and all(conditions_out))):
                Nsites_geo += 1
                pos[k-1,0] = r
                pos[k-1,1] = phi
                pos[k-1,2] = x
                pos[k-1,3] = y
                pos[k-1,4] = ir
                pos[k-1,5] = iphi
                if (r==0):
                    break

        Nstat_geo = 2*Nsites_geo
        pos_geo = np.zeros([Nsites_geo, 6])
        j = 0
        for i in range(Nsites):
            if pos[i,0] != 0:
                j+=1
                pos_geo[j-1,:] = pos[i,:]
            else:
                pass

    return pos_geo, Nstat_geo
