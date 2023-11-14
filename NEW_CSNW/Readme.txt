Assuming the requirements.txt file is correct, running "bash compile_all" on a unix machine should create a local Python venv, download the requirements, and compile the Cython files. Then, entering the venv using "source ./bin/activate" should allow running particular modules.

The following files should be anticipated to be correctly working:
 * hamiltonian.pyx, hamiltonian_to_c.py
 * k_structure.pyx, calc_k_structure.py
