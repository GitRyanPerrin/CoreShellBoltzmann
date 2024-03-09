from itertools import repeat

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

import input_data as ip
from hamiltonian import quantum_numbers
from fermi_derivative import fermi_derivative
from load_file import load_eigensystem, load_velocity, load_spin_velocity
from k_structure import calc_dispersion
from spin_matrix import Sijs

def main():

    #print(f"Loading Eigensystem: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")
    #k, eigvals, eigvecs, Nstat, Nev = load_eigensystem()
    #print(f"Loading Velocity: {ip.shape}, Nr={ip.Nr}, Nphi={ip.Nphi}, alpha={ip.alphaR}, beta={ip.betaD}...")

    #runs = []

    #diffusive_cond = np.zeros(len(runs))

    #for i,j in enumerate(runs):
    Nk = 80
    NE = 20
    #Nk = 700
    Nev = 6
    k0 = np.pi/ip.R
    k = np.linspace(-k0,k0,Nk)

    #eigvals = calc_dispersion(Nev=Nev, Nk=Nk, k=k, save_file=False, eigvals_only=True)
    eigvals = calc_dispersion(Nev=Nev, Nk=Nk, k=k, E=repeat(0), save_file=False, eigvals_only=True)
    eigvals2 = calc_dispersion(Nev=Nev, Nk=Nk, k=k, E=repeat(ip.E), save_file=False, eigvals_only=True)
    eigvals = np.stack(eigvals, axis=0)
    eigvals2 = np.stack(eigvals2, axis=0)

    #plt.suptitle(r"Internal Only $E = 0$")
    #plt.suptitle(r"Back-Gate $E = E_0 \hat{y}$")
    #plt.suptitle(r"Gate-All-Around $E = E_0 \hat{r}$")
    plt.title(rf'$\alpha_0$={round(ip.alphaR*ip.R*ip.ts)} meV nm and $\alpha_B$={round(ip.rashba_coeff*ip.E*ip.R*ip.ts)}')
    #plt.title(rf'$\alpha_1$={round(ip.rashba_coeff*ip.E*ip.R*ip.ts)} meV nm and $\alpha_2$={round(ip.rashba_coeff*1.4*ip.E*ip.R*ip.ts)}')
    plt.plot(k, eigvals, 'b:', label=r'Internal Only $\alpha_0$')
    plt.plot(k, eigvals2, 'k', label=r'Internal $\alpha_0$ and Back-Gate $\alpha_B$')
    #plt.plot(k, eigvals, 'b:', label=r'$\alpha_1$')
    #plt.plot(k, eigvals2, 'k', label=r'$\alpha_2$')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

if __name__=="__main__":
    main()

