from gc import collect

import numpy as np

import input_data as ip
from lattice import prismatic_nanotube
from k_structure import calc_dispersion, get_RAM_dumps, split_dispersion_calculation

def main():

    # Need number of states to determine how much RAM will be used.
    vertices = ip.vertices
    if vertices == 0:
        # Max number of eigenvalues solved for
        Nev = 50
        Nstat = ip.Nr*ip.Nphi*2
    else:
        Nev = 6*vertices
        pos, Nstat = prismatic_nanotube(vertices)
        del pos
        collect()

    Nk = 6000
    k = np.linspace(-2*np.pi/ip.R, 2*np.pi/ip.R, Nk)

    print(f"Number of states: {Nstat}..")
    print(f"Expected RAM use: {round(Nstat*Nstat*8*Nk/1e9)}GB..")

    '''
    To mitigate issues of the RAM running out,
    the calculation is done in chunks equal to
    the integer 'dumps'. This allows the calculation
    to be done on smaller PCs, but will take
    longer to run. The user must also combine
    the files afterwards using "combine_npz.py".
    '''

    dumps = get_RAM_dumps(Nev=Nev, Nk=Nk)
    print(f"Expected number of RAM dumps: {dumps}..")
    if dumps == 1:
        calc_dispersion(Nev=Nev, Nk=Nk, save_file=True, k=k)
    else:
        split_dispersion_calculation(Nev=Nev, k=k, Nk=Nk)

if __name__=='__main__':
    main()
