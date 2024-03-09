import gc
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy import integrate

import input_data as ip
from hamiltonian import Hmat
from fermi_derivative import fermi_derivative
from spin_matrix import Sijs
from spin_velocity_matrix import Vmat

Nev = 20
Nk = 600
k0 = np.pi/ip.R
k = np.linspace(-k0, k0, Nk)

#sz = Sijs('p')
#plt.imshow(abs(sz))
#plt.show()

#H = np.array([Hmat(k[ik]) for ik in range(Nk)])

with ProcessPoolExecutor() as pool:
    H_fut = pool.map(Hmat, k)

H_arr = np.stack([fut for fut in H_fut])
del H_fut
gc.collect()

eigvals = np.zeros(Nk, dtype=object)
eigvecs = np.zeros(Nk, dtype=object)

for ik in range(Nk):
    eigvals[ik], eigvecs[ik] = la.eigh(H_arr[ik], lower=False, subset_by_index=[0,Nev-1])

eigvals = np.stack(eigvals)
eigvecs = np.stack(eigvecs)

v = np.gradient(H_arr, axis=0)
#sz = np.kron(np.array([[1,0],[0,-1]]), np.eye(500))
#sz = np.kron(np.array([[0,1],[1,0]]), np.eye(500))
#sz = Sijs('p')
#sz = np.stack([eigvecs[ik].conj().T@sz@eigvecs[ik] for ik in range(Nk)])
#vs = np.stack([(v[ik]@sz + sz@v[ik])/2 for ik in range(Nk)])

with ProcessPoolExecutor() as pool:
    vs_fut = pool.map(Vmat, k)

vs = np.stack([fut for fut in vs_fut])
del vs_fut
gc.collect()

v = np.stack([eigvecs[ik].conj().T@v[ik]@eigvecs[ik] for ik in range(Nk)])
#vs = np.stack([(v[ik]@sz + sz@v[ik])/2 for ik in range(Nk)])
vs = np.stack([eigvecs[ik].conj().T@vs[ik]@eigvecs[ik] for ik in range(Nk)])
#v = v[:,:Nev,:Nev]
#vs = v[:,:Nev,:Nev]
del eigvecs
gc.collect()

T = 0.3
dmu = 3.0
band_min = np.min(eigvals) - 0.25
band_max = band_min + dmu
Nmu = 100
chem_potential = np.linspace(band_min, band_max, Nmu)

dfermi = np.stack([fermi_derivative(k, eigvals, mu, T, order=2) for mu in chem_potential])

prod = v*v*vs
prod = np.diagonal(prod[:,:Nev, :Nev], axis1=1, axis2=2)
integrand = np.stack([[prod[ik]*dfermi[imu, ik] for ik in range(Nk)] for imu in range(Nmu)])
#integrand = np.stack([[prod[ik]*np.diag(dfermi[imu, ik]) for ik in range(Nk)] for imu in range(Nmu)])
cond = np.stack([integrate.simps(integrand[imu], k, axis=0) for imu in range(Nmu)])
#cond = np.real(np.trace(cond, axis1=1, axis2=2))
cond = np.real(np.sum(cond,axis=1))

fig, ax = plt.subplots()
ax.plot(chem_potential, cond, 'k')
plt.show()
