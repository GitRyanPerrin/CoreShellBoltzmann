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

    def triangular(Nr, Nphi, Rext, t):
        
        cdef int Nsites
        cdef int Nstat
        cdef numpy.ndarray pos
        cdef int k
        cdef double r
        cdef double d_r
        cdef double d_phi
        cdef double x
        cdef double y
        cdef double Rint
        cdef double Tint
        

        Nsites = Nr*Nphi
        pos = np.zeros([Nsites, 4])
        Nsites_tri = 0
        k = 0
        r = Rext
        d_r = 0
        #Rint = Rext/2-t
        Tint = Rext-t/cos(pi/3)
        Rint = Tint*cos(pi/3)

        if (Nr>1): d_r = (Rext-Rint)/(Nr-1)
        d_phi = 2*pi/Nphi
        for ir in np.arange(1, Nr+1, 1, dtype=int):
            if (Nr>1): r = Rext-d_r*(ir-1)
            for iphi in np.arange(1, Nphi+1, 1, dtype=int):
                phi = d_phi*(iphi-1)
                k+=1

                x = r*cos(phi)
                y = r*sin(phi)
                if ( ( Tint <= x <= Rext and (Rext-x)/sqrt(3) >= y >= (x-Rext)/sqrt(3) )
                    or ( ( -0.5*Rext + t <= x <= Tint ) and ( (Tint-x)/sqrt(3) <= y <= (Rext-x)/sqrt(3)
                    or (x-Rext)/sqrt(3) <= y <= (x-Tint)/sqrt(3) ) )
                    or (-0.5*Rext <= x <= -0.5*Rext + t and (x-Rext)/sqrt(3) <= y <= (Rext-x)/sqrt(3) )
                    or x == -0.5*Rext and ( x**2 + y**2 == 1 )
                    ):
                    Nsites_tri += 1
                    pos[k-1,0] = r
                    pos[k-1,1] = phi
                    pos[k-1,2] = x
                    pos[k-1,3] = y
                    if r==0:
                        break
        Nstat_tri = 2*Nsites_tri
        pos_tri = np.zeros([Nsites_tri, 4])
        j = 0
        for i in range(Nsites):
            if pos[i,0] != 0:
                j+=1
                pos_tri[j-1,:] = pos[i,:]
            else:
                pass

        return pos_tri, Nstat_tri

def prismatic_nanotube_1D(N):
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

    #Nr, Nphi, Rext, t = input.Nr, input.Nphi, input.Rext, input.t
    Nr, Nphi, Rext = input.Nr, input.Nphi, input.Rext
    X = [Rext * cos(2*pi*n/N) for n in range(N+1)]
    Y = [Rext * sin(2*pi*n/N) for n in range(N+1)]

    slope = [(Y[n+1]-Y[n])/(X[n+1]-X[n]) for n in range(N)]
    intercept = [(X[n]*Y[n+1] - X[n+1]*Y[n])/(X[n]-X[n+1]) for n in range(N)]

    def bound(x):
        for n in range(N):
            y = slope[n]*x+intercept[n]
        return sqrt(x**2 + y**2)

    def line(x, n):
        return slope[n]*x+intercept[n]

    def delta_y(n):
        return Y[n+1]-Y[n]

    def delta_x(n):
        return X[n+1]-X[n]


    k = 0
    r = Rext
    d_r = 0
    d_phi = 2*pi/Nphi
    Nc = int(Nphi/N)
    pos = np.zeros([int(N*Nc), 4])

    for n in range(N):
        d_c = sqrt(delta_y(n)**2 + delta_x(n)**2)/Nc
        theta = np.arctan(delta_y(n)/delta_x(n))
        x0 = Rext*cos(2*pi*n/N)
        for ic in range(0, Nc):
            k+=1
            #if n < N/2 and x0 != Rext*cos(2*pi/3):
            if n < N/2 and not (N==3 and x0 == Rext*cos(2*pi/3)):
                x = x0 - d_c*ic*cos(theta)
                y = line(x,n)
                #elif n >= N/2 and (N!=3 and x0 != Rext*cos(2*pi/3)):
            elif n >= N/2 and not (N==3 and x0 == Rext*cos(2*pi/3)):
                x = x0 + d_c*ic*cos(theta)
                y = line(x,n)
            elif N==3 and x0 == Rext*cos(2*pi/3):
                x = x0
                y = Rext*sin(2*pi*n/N) - d_c*ic
            pos[k-1,2] = x
            pos[k-1,3] = y
            pos[k-1,0] = sqrt(x**2 + y**2)
            #pos[k-1,1] = 2*pi*n/(ic+1)/N
            pos[k-1,1] = np.arctan(y/x)

    return pos, 2*Nc*N

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

def prismatic_nanowire(N):
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

    slope_out = [(Yout[n+1]-Yout[n])/(Xout[n+1]-Xout[n]) for n in range(N)]
    intercept_out = [(Xout[n]*Yout[n+1] - Xout[n+1]*Yout[n])/(Xout[n]-Xout[n+1]) for n in range(N)]

    def outer_bound(x):
        for n in range(N):
            y = slope_out[n]*x+intercept_out[n]
        return sqrt(x**2 + y**2)

    def line_out(x, n):
        return slope_out[n]*x+intercept_out[n]

    x_lin = np.linspace(-1,1,100)

    Nsites = Nr*Nphi
    Nstat = 2*Nsites
    pos = np.zeros([Nsites, 6])
    Nsites_geo = 0

    k = 0
    r = Rext
    d_r = 0

    if (Nr>1): d_r = Rext/(Nr-1)
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
                x_vert_1 = np.round(cos(2*pi*n/N), 5)
                x_vert_2 = np.round(cos(2*pi*(n+1)/N), 5)
                #if x_vert_1 == x_vert_2: print(x_vert_1, x_vert_2)
                bound_out = sqrt(x**2 + line_out(x,n)**2)
                if np.round(x_vert_1, 5) != np.round(x_vert_2, 5):
                    conditions_out.append(r <= bound_out)
                    x_vert = 1
                else:
                    x_vert = x_vert_1
            if (all(conditions_out) and x < x_vert ):
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


def square_lattice_prism_nanotube(Nx = 50, Ny = 50, N = input.vertices, t = 0.2):
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

    Rverts = Rext - t/cos(pi/N)
    Rint = Rverts*cos(pi/N)

    Xin = [Rverts * cos(2*pi*n/N) for n in range(N+1)]
    Yin = [Rverts * sin(2*pi*n/N) for n in range(N+1)]

    slope_in = [(Yin[n+1]-Yin[n])/(Xin[n+1]-Xin[n]) for n in range(N)]
    intercept_in = [(Xin[n]*Yin[n+1] - Xin[n+1]*Yin[n])/(Xin[n]-Xin[n+1]) for n in range(N)]

    slope_out = [(Yout[n+1]-Yout[n])/(Xout[n+1]-Xout[n]) for n in range(N)]
    intercept_out = [(Xout[n]*Yout[n+1] - Xout[n+1]*Yout[n])/(Xout[n]-Xout[n+1]) for n in range(N)]

    def outer_bound(x):
        for n in range(N):
            y = slope_out[n]*x+intercept_out[n]
        return sqrt(x**2 + y**2)

    def inner_bound(x):
        for n in range(N):
            y = slope_in[n]*x+intercept_in[n]
        return sqrt(x**2 + y**2)

    def line_out(x, n):
        return slope_out[n]*x+intercept_out[n]

    def line_in(x, n):
        return slope_in[n]*x+intercept_in[n]

    x_lin = np.linspace(-1,1,100)

    Nsites = Nx*Ny
    pos = np.zeros([Nsites, 4])
    Nstat = 2*Nsites
    x_width = 2.0*Rext
    y_width = 2.0*Rext
    d_x = x_width/Nx
    d_y = y_width/Ny

    x0 = 0.0
    y0 = 0.0

    x_vals = [x0 + i*d_x for i in np.arange(-Nx/2, Nx/2, 1)]
    y_vals = [y0 + j*d_x for j in np.arange(-Ny/2, Ny/2, 1)]

    k = 0
    Nsites_geo = 0
    for i in range(Nx):
        x = x_vals[i]
        for j in range(Ny):
            y = y_vals[j]
            r = sqrt(x**2 + y**2)
            if x!= 0:
                phi = np.arctan(y/x)
            else:
                phi = 0
            k+=1
            conditions_out = []
            conditions_in = []
            for n in range(N):
                if round(cos(2 * pi * n/N),2) == round(cos(2 * pi * (n+1)/N),2):
                    conditions_out.append(x >= Xout[n])
                    conditions_in.append(x <= Xout[n] + t)
                else:
                    bound_out = sqrt(x**2 + line_out(x,n)**2)
                    conditions_out.append(r <= bound_out)
                    bound_in = sqrt(x**2 + line_in(x,n)**2)
                    conditions_in.append(r >= bound_in)
            if (all(conditions_out) and any(conditions_in)) or (abs(r) > Rverts and all(conditions_out)):
                Nsites_geo += 1
                pos[k-1, 0] = r
                pos[k-1, 1] = phi
                pos[k-1, 2] = x
                pos[k-1, 3] = y

    Nstat_geo = 2*Nsites_geo
    pos_geo = np.zeros([Nsites_geo, 4])
    j = 0
    for i in range(Nsites):
        if pos[i,0] != 0:
            j+=1
            pos_geo[j-1,:] = pos[i,:]
        else:
            pass

    return pos_geo, Nstat_geo

def square_lattice_nanotube(Nx = 50, Ny = 50, t = input.t):
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
    
    Rext, Rint = input.Rext, input.Rext - t

    Nsites = Nx*Ny
    pos = np.zeros([Nsites, 4])
    Nstat = 2*Nsites
    x_width = 2.0
    y_width = 2.0
    d_x = x_width/Nx
    d_y = y_width/Ny

    x0 = 0.0
    y0 = 0.0

    x_vals = [x0 + i*d_x for i in np.arange(-Nx/2, Nx/2, 1)]
    y_vals = [y0 + j*d_x for j in np.arange(-Ny/2, Ny/2, 1)]

    k = 0
    Nsites_geo = 0
    for i in range(Nx):
        x = x_vals[i]
        for j in range(Ny):
            y = y_vals[j]
            r = sqrt(x**2 + y**2)
            if x!= 0:
                phi = np.arctan(y/x)
            else:
                phi = 0
            if (r <= Rext) and (r >= Rint):
                k+=1
                Nsites_geo += 1
                pos[k-1, 0] = r
                pos[k-1, 1] = phi
                pos[k-1, 2] = x
                pos[k-1, 3] = y

    Nstat_geo = 2*Nsites_geo
    pos_geo = np.zeros([Nsites_geo, 4])
    j = 0
    for i in range(Nsites):
        if pos[i,0] != 0:
            j+=1
            pos_geo[j-1,:] = pos[i,:]
        else:
            pass

    return pos_geo, Nstat_geo
