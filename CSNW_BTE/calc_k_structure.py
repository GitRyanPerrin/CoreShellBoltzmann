from gc import collect

import numpy as np

import input_data as ip
from lattice import prismatic_nanotube
from k_structure import calc_dispersion, get_RAM_dumps, split_dispersion_calculation

def main():

    vertices = ip.vertices
    if vertices == 0:
        Nsing = 50
        #Nstat = 2*Nr*Nphi
    else:
        Nsing = 6*vertices
        pos, Nstat = prismatic_nanotube(vertices)
        print(f"Number of states: {Nstat}..")
        print(f"Expected RAM use: {round(Nstat*Nstat*8*6000/1e9)}GB..")
        del pos
        collect()

    Nk = 6000
    k = np.linspace(-2*np.pi/ip.R, 2*np.pi/ip.R, Nk)
    dumps = get_RAM_dumps(Nsing=Nsing, Nk=Nk)
    print(f"Expected number of RAM dumps: {dumps}..")
    if dumps == 0:
        calc_dispersion(Nsing=Nsing, Nk=Nk, save_file=True, k=k)
    else:
        split_dispersion_calculation(Nsing=Nsing, k=k, Nk=Nk)

if __name__=='__main__':
    main()
