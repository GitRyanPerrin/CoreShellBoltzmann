import numpy as np
from numpy import exp

import input_data as ip

def FD(E, mu, T):

    '''
    Fermi Distribution
    '''

    kBT = ip.kB*T
    x = (E-mu)/kBT
    if x > 0:
        return exp(-x)/(1 + exp(-x))
    else:
        return 1/(1+exp(x))
    #return 0.5+0.5*np.tanh(-x/2)

def dFD(E, mu, T):

    '''
    First-Order Derivative of Fermi Distribution
    '''

    return -FD(E,mu,T)*(1.0-FD(E,mu,T))

def ddFD(E, mu, T):

    '''
    Second-Order Derivative of Fermi Distribution
    '''

    kBT = ip.kB*T
    return -(1-2*FD(E,mu,T))*dFD(E,mu,T)/kBT

def fermi_derivative(k, eigvals, mu, T, order=1):

    '''
    Calculates the fermi derivative. The variable 'order'
    determines the number of derivatives taken.
    '''

    # Number of k-points
    Nk = len(k)
    # Number of energy eigenvalues
    Nev = np.shape(eigvals)[1]
    # output array
    fermi = np.zeros([Nk, Nev])

    for a in range(Nev):
        for ik in range(Nk):

            if order == 1:
                fermi[ik,a] = dFD(eigvals[ik,a], mu, T)
            if order == 2:
                fermi[ik,a] = ddFD(eigvals[ik,a], mu, T)

    return fermi
