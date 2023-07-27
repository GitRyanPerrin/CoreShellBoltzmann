from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

import hamiltonian as ham
from fermi_energy import FD, dFD, ddFD
from spin_matrix import Sijs

Nk = 50
k = np.linspace(-0.1, 0.1, Nk)

with ProcessPoolExecutor() as pool:
    H_fut = pool.map(ham.Hmat, k)

Hk = np.stack([fut for fut in H_fut])

eigvals = np.zeros(Nk, dtype=object)
eigvecs = np.zeros(Nk, dtype=object)

for ik in range(Nk):
    eigvals[ik], eigvecs[ik] = eigh(Hk[ik], lower=False)

eigvals = np.stack(eigvals)
vk = np.gradient(eigvals, axis=0)
S = Sijs('z')
S = np.stack([eigvecs[ik].conj().T@S@eigvecs[ik] for ik in range(Nk)])
s_vel = np.stack([np.diag(vk[ik])@S[ik] + S[ik]@np.diag(vk[ik]) for ik in range(Nk)])
print(s_vel.shape)
Nstat = eigvals.shape[1]

mu = np.min(eigvals)+5.0

fermi = np.zeros(Nk)
dfermi = np.zeros(Nk)
for ik in range(Nk):
    for a in range(Nstat):
        #fermi[ik] += FD(eigvals[ik,a], mu, 0.1)
        dfermi[ik] += s_vel[ik,a,a]**2*vk[ik,a]*dFD(eigvals[ik,a], mu, 10.0)

#plt.plot(k, fermi, 'k')
plt.plot(k, dfermi, 'r')
plt.show()
