Assuming the requirements.txt file is correct, running "bash compile_all" on a unix machine should create a local Python venv, download the requirements, and compile the Cython files. Then, entering the venv using "source ./bin/activate" should allow running particular modules.

The files which need to be worked on are:
   * calc_spin_velocity.py
   * calc_velocity.py
   * new_ballistic_cond.py
   * new_spin_cond.py
   * spin_cond.py
   * spin_matrix.py
   * spin_velocity.pyx
   * spin_velocity_matrix.py
   * thermospin.py
   * velocity.pyx
   * velocity_matrix.py
   

The following files should be anticipated to be correctly working:
   * ballistic_cond.py
   * calc_k_structure.py
   * combine_npz.py
   * compile_all
   * fermi_derivative.py
   * hamiltonian.pyx
   * hamiltonian_to_c.py
   * input_data
   * k_structure.pyx
   * k_structure_to_c.py
   * lattice.pyx
   * lattice_to_c.py
   * load_file.py
   * requirements.txt
   * spin_velocity_to_c.py
   * thermoelec_cond.py
   * velocity_to_c.py
