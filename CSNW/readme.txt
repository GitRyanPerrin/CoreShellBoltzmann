1) Running the CSNW modules requires gcc and python3.
    a) On a Debian-based system:
        sudo apt update
        sudo apt install build-essentials
        sudo apt install python3

    b) Confirm installations:
        gcc --version
        python3 --version

    *) If not on a Debian-based system, you will need to install a C-compiler and the python3 interpreter in the appropriate manner.

    **) If you are using a development environment, you can find these through the package manager.

2) On Linux or Mac: Run the "compile_all" bash file.
    a) This file compiles the main Cython modules for the Lattice, Hamiltonian, and Band-Structure Calculations:
        bash compile_all

    b) It also creates a python3 virtual environment which can be re-activated:
        source ./bin/activate

    *) compile_all will check for the C-compiler and python3 interpreter, producing an error if not found.

    **) If on Windows, you will need to install each dependency from requirements.txt manually.
        Each line of requirements.txt is a new package. 
        They can be installed in the command line through:
            pip3 install "package_name"

3) The available eigensystems will be found in the eigensystems folder.
    a) From current directory it is possible to check which configurations have been already calculated:
        ls eigensystems

        i) File Name Syntax: Shape_Radius_Thickness_RadialGridPoints_AzimuthalGridPoints_Alpha_Beta.npz

                             Shape = Shape of CSNW Cross-Section
                             Radius = outer radius of CSNW in nanometers
                             Thickness = Thickness of shell as a fraction of radius (Ex: R=30 and t=0.2 implies thickness is 6nm)
                             RadialGridPoints = Number of radial points in polar grid
                             AzimuthalGridPoints = Number of azimuthal points in polar grid
                             Alpha = Dimensionless Rashba Coefficient R/lso_R (lso_R = characteristic RSOC length)
                             Beta = Dimensionless Dresselhaus Coefficient R/lso_D (lso_D = characteristic DSOC length)

    b) To calculate a new eigensystem:

        i) Modify input_data.py for the corresponding parameters.

        ii) Calculate the band-structure (E vs k):
            python calc_k_structure.py

            *) This will be a long process. It is optimized for reducing necessary RAM, but will write to disk several times if RAM is low.
               For instance, a run for a Circular CSNW with Nr=10 and Nphi=50 requires 48GB of RAM.
               When considering polygonal cross-sections it becomes exacerbated: Shape=Square, Nr=10, Nphi=300 requires 269GB of RAM.

            *) If the number of RAM dumps is greater than 1, it will be necessary to combine the output files:
                python combine_npz.py

4) From the python3 virtual environment (2b) you can run the following conductivity calculations:
    a) Diffusive and Ballistic conductivity:
        python3 ballistic_cond.py

    b) Spin Conductivity in Second-Order:
        python3 spin_cond.py

    c) Seebeck Coefficient (Diffusive is currently not working?):
        python3 thermoelec_cond.py

    d) Spin-Seebeck (Does not seem correct):
        python3 thermospin_cond.py

5) All other files in the directory are utilized from within one of these programs.
   Other than for debugging, it will not be necessary to run these directly.
   
   
   
   
   
