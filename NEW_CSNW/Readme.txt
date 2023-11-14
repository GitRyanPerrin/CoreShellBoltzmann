Assuming the requirements.txt file is correct, running "bash compile_all" on a unix machine should create a local Python venv, download the requirements, and compile the Cython files. Then, entering the venv using "source ./bin/activate" should allow running particular modules.

The following files should be anticipated to be correctly working:
 * lattice.pyx, lattice_to_c.py
 * hamiltonian.pyx, hamiltonian_to_c.py
 * k_structure.pyx, k_structure_to_c.py, calc_k_structure.py
    * This portion may benefit from optimization, possibly through the interpolation of the band-structure
 * ballistic_cond.py
