Assuming the requirements.txt file is correct, running "bash compile_all" on a unix machine should create a local Python venv, download the requirements, and compile the Cython files. Then, entering the venv using "source ./bin/activate" should allow running particular modules.

The files which need to be worked on are:
   * new_ballistic_cond.py 
      * Calculates charge cond. w/ hard-coded velocity matrix, velocity_matrix.py

   * new_spin_cond.py 
      * Calculates spin cond. w/ hard-coded velocity matrices, velocity_matrix.py and spin_matrix.py

   * spin_cond.py 
      * Calculates spin cond. w/ v = dE/dk. Does not work, v is not a matrix in this case

   * spin_matrix.py 
      * Hard-coded spin matrix for anticommutation in spin_cond.py and thermospin_cond.py

   * spin_velocity_matrix.py 
      * Hard-coded spin-velocity matrix for new_spin_cond.py

   * thermospin_cond.py 
      * Calculates spin seebeck w/ v = dE/dk

   * velocity_matrix.py 
      * Hard-coded charge velocity matrix for use in new_ballistic_cond.py and new_spin_cond.py
   

The following files should be anticipated to be correctly working:
   * ballistic_cond.py
      * Calculates the charge conductivity with v = dE/dk

   * calc_k_structure.py
      * Runs the band-structure calculation

   * calc_spin_velocity.py
      * Calculates the hard-coded spin-velocity matrices

   * calc_velocity.py
      * Calculates the hard-coded charge-velocity matrices

   * combine_npz.py
      * Combines multiple compressed numpy arrays

   * compile_all

   * fermi_derivative.py
      * Contains Fermi-Dirac function and its derivatives w.r.t. E(k)

   * hamiltonian.pyx
      * Calculates Hamiltonian matrix

   * hamiltonian_to_c.py
      * Compiles hamiltonian.pyx to C

   * input_data
      * Contains constants, geometries, field values

   * k_structure.pyx
      * Calculates the band-structure

   * k_structure_to_c.py
      * Compiles k_structure.pyx to C

   * lattice.pyx
      * Creates the cross-sectional grid

   * lattice_to_c.py
      * Compiles the grid to C

   * load_file.py
      * Loads eigensystems and velocities

   * requirements.txt

   * spin_velocity.pyx 
      * Calculates the RAM usage and splits the calculation if necessary

   * spin_velocity_to_c.py
      * Compiles spin_velocity_matrix.pyx and spin_velocity.pyx to C

   * thermoelec_cond.py
      * Calculates the Seebeck coefficient using v = dE/dk

   * velocity.pyx
      * Calculates the RAM usage and splits the calculation if necessary
   * velocity_to_c.py
      * Compiles velocity_matrix.pyx and velocity.pyx to C
