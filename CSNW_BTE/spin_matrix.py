import numpy as np
from numpy import sqrt, pi, cos, sin, exp
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.linalg as la
from concurrent.futures import ProcessPoolExecutor

from hamiltonian import Hmat, quantum_numbers, SigMat
import input_data as ip
from lattice import nanotube, prismatic_nanotube

def Sijs(polarization, factor=1):

    Nr, Nphi = ip.Nr, ip.Nphi
    Rext, t = ip.Rext, ip.t
    vertices = ip.vertices

    qn = quantum_numbers(Nr, Nphi)

    if ip.vertices == 0:
        pos, Nstat = nanotube.circular(Nr, Nphi, Rext, t)
        d_r = t/(Nr-1)
    else:
        pos, Nstat = prismatic_nanotube(vertices)
        d_r = (Rext*(1-cos(pi/vertices))+t)/(Nr-1)
    d_phi = 2*pi/Nphi

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

            if ii == 0 and (jj == 0 or jj == (Nphi-1)):
                sigma_ijs[a1, a2] = SigMat(polarization, s1, s2, factor*phi1)

            sigma_ijs[a2,a1] = sigma_ijs[a1,a2].conj()
    '''

    Nr, Nphi = ip.Nr, ip.Nphi
    Nsites = ip.Nr*ip.Nphi

    sigma_x = np.array([
        [SigMat('x', -1, -1, 0), SigMat('x', -1, 1, 0)],
        [SigMat('x',  1, -1, 0), SigMat('x',  1, 1, 0)]
    ])

    sigma_y = np.array([
        [SigMat('y', -1, -1, 0), SigMat('y', -1, 1, 0)],
        [SigMat('y',  1, -1, 0), SigMat('y',  1, 1, 0)]
    ])

    Rext, t = ip.Rext, ip.t
    vertices = ip.vertices

    qn = quantum_numbers(Nr, Nphi)

    if ip.vertices == 0:
        pos, Nstat = nanotube.circular(Nr, Nphi, Rext, t)
        d_r = t/(Nr-1)
    else:
        pos, Nstat = prismatic_nanotube(vertices)
        d_r = (Rext*(1-cos(pi/vertices))+t)/(Nr-1)
    d_phi = 2*pi/Nphi

    cos_array = np.zeros([Nsites,Nsites])
    sin_array = np.zeros([Nsites,Nsites])
    for a in range(Nsites):
        q = int(qn[2*a,0])
        phi = pos[q-1,1]
        cos_array[a, a] = cos(phi)
        sin_array[a, a] = sin(phi)

    sigma_ijs = -np.kron(sin_array, sigma_x) + np.kron(cos_array, sigma_y)
    '''

    return sigma_ijs

def unitary_transform(eigvecs, matrix=Sijs('p')):

    Nstat = eigvecs.shape[1]
    Nsing = eigvecs.shape[-1]
    if Nsing > 50: Nsing = 100
    #eigvecs = eigvecs/sqrt(Nsing)
    #Nsing = np.shape(eigvecs[1])[0]
    #mat_trans = eigvecs@matrix[:Nsing, :Nsing]@eigvecs.conj()
    #mat_trans = eigvecs@matrix[:Nsing, :Nsing]@eigvecs.conj().swapaxes(0,1)
    #mat_trans = eigvecs@matrix@eigvecs
    #mat_trans = eigvecs[:,:Nsing]@matrix@eigvecs.conj().T[:Nsing,:]
    #mat_trans = eigvecs.conj().T[:Nsing,:]@matrix@eigvecs[:,:Nsing]
    #mat_trans = eigvecs.conj().T@matrix@eigvecs
    #mat_trans = eigvecs@matrix@eigvecs.conj().swapaxes(0,2)
    #mat_trans = eigvecs@matrix[:Nsing, :Nsing]
    #mat_trans = matrix[:Nsing, :Nsing]
    #mat_trans = inv(eigvecs[:Nsing,:Nsing])@matrix[:Nsing, :Nsing]@eigvecs[:Nsing,:Nsing]
    #mat_trans = np.linalg.inv(eigvecs[:Nsing, :Nsing])@matrix@eigvecs[:Nsing, :Nsing]

    mat_trans = np.zeros([Nsing, Nsing], dtype=complex)
    for n in range(Nstat):
        mat_trans += np.outer(eigvecs[n,:Nsing], eigvecs[n,:Nsing].conj().T)@matrix[:Nsing, :Nsing]

    return mat_trans

def calc_unitary_transform(eigvecs, matrix=Sijs('p')):

    #Nsing = np.shape(eigvecs[1])[0]
    Nk = eigvecs.shape[0]
    #vec_unitary = np.vectorize(unitary_transform, excluded=['matrix'])
    #vec_unitary = np.apply_along_axis(unitary_transform, 0, eigvecs)
    vec_unitary = np.stack([unitary_transform(eigvecs[ik], matrix) for ik in range(Nk)], axis=0)

    return vec_unitary

def main():

    #eigvals, eigvecs = la.eigh(Hmat(0.01), lower=False, turbo=True)
    '''
    suffix = f"{ip.shape}_{ip.R}_{ip.t}_{ip.Nr}_{ip.Nphi}_{ip.B}_{ip.alphaR}_{ip.betaD}_{ip.mag_direction}"
    with np.load(f"/scratch/rperrin/rperrin/" + suffix + "_sparse.npz", allow_pickle=True) as file:
        k = file['arr_0']
        Nk = len(k)
        eigvals = file['arr_1']
        Nstat = len(eigvals[0])
        eigvals = np.stack(eigvals, axis=1)
        eigvecs = file['arr_2']
        eigvecs = np.stack(eigvecs, axis=0)
        Nsing = np.shape(eigvecs[0])[1]
    '''

    Nk = 30
    k = np.linspace(0.0, 0.08, Nk)
    '''
    k = np.r_[
        np.linspace(0.00, 0.02, int(Nk/2), endpoint=False),
        np.linspace(0.02, 0.04, Nk, endpoint=False),
        np.linspace(0.04, 0.05, int(Nk/2), endpoint=False),
        np.linspace(0.05, 0.06, Nk, endpoint=False),
        np.linspace(0.06, 0.08, int(Nk/2))
    ]
    '''

    with ProcessPoolExecutor() as pool:
        H_futures = pool.map(Hmat,k)

    eigvals, eigvecs = np.zeros(Nk, dtype=object), np.zeros(Nk, dtype=object)
    eigsys = [la.eigh(future) for future in H_futures]
    for ik in range(Nk):
        eigvals[ik], eigvecs[ik] = eigsys[ik]
    eigvals = np.stack(eigvals, axis=1)
    eigvecs = np.stack(eigvecs, axis=0)

    sigma_p_ijs = SigMat_ijs('p')
    sigma_p_a = [unitary_transform(eigvecs[ik], sigma_p_ijs) for ik in range(Nk)]
    sigma_p_a = np.stack(sigma_p_a, axis=2)
    sigma_p_exp = np.sum(sigma_p_a, axis=1)
    #sigma_p_exp = np.stack([np.diagonal(sigma_p_a[ik]) for ik in range(Nk)], axis=1)

    plt.ylim([163,168])
    #[plt.plot(k, np.sign(np.real(sigma_p_exp.T[a]))+2*a) for a in range(2)]
    #[plt.scatter(k, eigvals[a], s=1, c=np.sign(np.real(sigma_p_exp[a]))) for a in range(5)]
    [plt.scatter(k, eigvals[a], s=1, c=np.sign(sigma_p_exp[a].real)) for a in range(5)]
    plt.colorbar()

    plt.savefig('./plots/test_sigma.png')

if __name__=='__main__':
    main()
