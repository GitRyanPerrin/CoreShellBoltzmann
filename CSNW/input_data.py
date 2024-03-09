from numpy import sin, cos, pi, sqrt
import numpy as np

# Units are meV, s, nm, T
e_eV = 1 # "natural"
e_SI = 1.60217663*10**(-19) # C

h_eV = 4.135667696*10**(-12) # meV.s
hbar_eV = 6.582*10**(-13) # meV.s
hbar_SI = 1.054571817e-34 # J.s

c_eV = 2.99792458e17 # [c] = nm/s
c_SI = 2.99792458e8 # m/s

m0_eV = 0.511 * 10**(9)/c_eV**2 # base electron mass in meV/c^2
m0_SI = 9.109383702e-31

muB_eV = 5.788381e-2 # in meV/T
muB_SI = 9.274010078e-24 # J/T

kB_eV = 8.61733262e-2 # meV/K
kB_SI = 1.380649e-23 # J/K

eV_to_J = e_SI
J_to_eV = 1/eV_to_J
J_to_meV = J_to_eV*1e3

tau_e = 0.5e-12
tau_s = 1e-10

# size parameters
R_eV = 50 # nm
R_SI = R_eV*1e-9 # m
Rext = 1.0 # External Radius (unitless)
t = 0.2*Rext # nm
#L = 600 # Length of the nanowire (nm)
L = 'inf' # Length of the nanowire (nm)
Nl = 400 # Number of longitudinal modes

hbar = hbar_eV
e = e_eV
c = c_eV
m0 = m0_eV
R = R_eV
kB = kB_eV
phi_o = hbar_SI/e_SI*2*pi
G_o = e_SI**2/hbar_SI*2*pi

## Polygon shape ##
#vertices, shape, shape_string = 6, "hex", "Hexagonal 3DEG"
#vertices, shape, shape_string = 4, "sqr", "Square 3DEG"
vertices, shape, shape_string = 3, "tri", "Triangular 3DEG"
#vertices, shape, shape_string = 0, "circ", "Circular 3DEG"
#vertices, shape, shape_string = 0, "circ_2D", "Circular 2DEG"

# Number of sites in polar grid #
# Hex and Square
if shape == "hex" or shape == "sqr":
    #Nr, Nphi = int(200), int(10)
    #Nr, Nphi = int(10), int(300)
    #Nr, Nphi = 4, 50
    #Nr, Nphi = 10, 100
    Nr, Nphi = 10, 300
# Triangle
if shape == "tri":
    Nr, Nphi = int(30), int(141)
    #Nr, Nphi = int(15), int(141)
    #Nr, Nphi = int(10), int(141)
# Circle
if shape == "circ":
    Nr, Nphi = 10, 50
    #Nr, Nphi = 5, 10
    #Nr, Nphi = 10, 100
    #Nr, Nphi = 15, 100
# 2D Circle
if shape == "circ_2D":
    Nr, Nphi = 1, 100

################# Material Parameters ######################
material = "InAs"
#material = "InSb"
#material = "InX"

if material == "InAs":
    meff = 0.026*m0 # effective mass of InAs
    Phi0 = h_eV*10**(16)/2 # Flux Quantum in nm**2*T
    n_2D = 1.17 # num. of electrons, units 10**(-3) 1/nm^2
    ns_3D = 1.30 # num. of electrons, units 10**(-4) 1/nm^3
    rashba_coeff = 1.171 # enm^2 form, for translating alpha to electric fields, r = alpha/E for InAs
    #rashba_coeff = 0.0
    geff = -14.9

if material == "InSb":
    meff = 0.014*m0 # effective mass of InSb
    n_2D = 1.0
    n_3D = 1.0
    rashba_coeff = 5.230 # InSb
    geff = -51.6

if material == "InX":
    meff = 0.2*m0
    n_2D = 1.08
    n_3D = 1.15
    rashba_coeff = 3.205
    geff = (geff_InAs + geff_InSb)/2

ts = hbar**2/2/meff/R**2
gamma = geff*meff/2/m0
alphaR_InAs, betaD_InAs = round(20/R/ts,2), round(3/R/ts,2)
alphaR_InSb, betaD_InSb = round(50/R/ts, 2), round(30/R/ts,2)
alphaR_InAs_German, betaD_InAs_German = -round(7.4/R/ts,2), round(4.1/R/ts,2)
alphaR_InX, betaD_InX = round(35/R/ts), round(17/R/ts)

# Area of the nanotube, for use in calculating AB Oscillation period, Phi = BA
# If regular polygon
if vertices > 0:
    # Calculate thickness of polygon corners
    t_corner = R*t/cos(pi/vertices)
    # Calculate area of polygon
    #area = R**2*vertices/2*sin(2*pi/vertices)*(1 - (1 - t_corner)**2)
    area = R**2*cos(pi/vertices)*sin(pi/vertices)*vertices # outter area
    area_inner = (1-t)*R**2*cos(pi/vertices)*sin(pi/vertices)*vertices
    #area -= (R-t)**2*cos(pi/vertices)*sin(pi/vertices)*vertices/2
    circum = 2*R*sin(pi/vertices)*vertices
elif vertices == 0 and Nr > 1:
    area_ring = R**2 * pi * (1 - (1 - t)**2)
    area_enclosed = R**2 * pi
    area = area_ring
    circum = 2*pi*R
elif vertices == 0 and Nr == 1:
    area = R**2 * pi
    circum = 2*pi*R

def calc_circum(R=R):
    return 2*R*sin(pi/vertices)*vertices

def calc_periodicity(R=R):
    return 4*pi**2*hbar/calc_circum(R)**2*10**15

mag_period = 4*pi**2*hbar/circum**2*10**15

#e^2 s/nm^2 1/nm nm^2/s^2 1/meV = e/(nm s mV)
#units_cond = e_SI**2*tau_e*1e-21/hbar_SI**2
#units_spin = e_SI*tau_e**2/4/hbar_SI**2*1e-30

# Diffusive
#units_cond = tau_e*G_o/hbar
units_spin = tau_s/phi_o*100
# Ballistic
units_cond = 2*pi/G_o*100
#units_spin = 2*pi/phi_o*1e-9

# ip.k
k = 0

# Field Parameters
#mag_direction = "z"
mag_direction = "x"
#mag_direction = "xy" # directed at a face of the polygon
#mag_direction = "arbitrary"
B = 0.0 # Manetic Field (T) delta = 0 # Magnetic field directed along vertex
#delta = pi/vertices # Magnetic field directed perp. to face
E = 0.0 # Electric Field (mV/R) **radius is not scaled in the Hamiltonian currently
#E = -0.256 # Electric Field (mV/R) **radius is not scaled in the Hamiltonian currently
#E = -0.0 # Electric Field (mV/R) **radius is not scaled in the Hamiltonian currently

## SOI Model Type ##
#SOI_definition = 'manolescu' # Eur Phys J
#SOI_definition = 'kokurin' # Physica E
SOI_definition = '111' # Equivalent to 'kokurin' and 'german'
#SOI_definition = '001'
#SOI_definition = 'german' # PRB 93
#SOI_definition = 'ring' # PRB 84

# Calculating the External Rashba Parameter
#alphaR, betaD = 0.0, betaD_InAs
#alphaR, betaD = 0.1, betaD_InAs
#alphaR, betaD = 0.2, betaD_InAs
#alphaR, betaD = 0.3, betaD_InAs
#alphaR, betaD = 0.4, betaD_InAs
#alphaR, betaD = 0.5, betaD_InAs
#alphaR, betaD = 0.6, betaD_InAs
#alphaR, betaD = 0.7, betaD_InAs
#alphaR, betaD = 0.8, betaD_InAs
#alphaR, betaD = 0.9, betaD_InAs
#alphaR, betaD = alphaR_InAs, betaD_InAs
#alphaR, betaD = alphaR_InSb, betaD_InSb
#alphaR, betaD = alphaR_InAs_German, betaD_InAs_German
#alphaR, betaD = alphaR_InSb, 0.0
#alphaR, betaD = alphaR_InAs, 0.0
#alphaR, betaD = alphaR_InAs_kokurin, 0.0
#alphaR, betaD = alphaR_InAs, betaD_InAs_German
#alphaR, betaD = 0.0, betaD_InAs
#alphaR, betaD = 0.0, betaD_InAs
#alphaR, betaD = alphaR_InX, betaD_InX
#alphaR, betaD = alphaR_InSb, betaD_InSb
#alphaR, betaD = 0.5, 0.5
#alphaR, betaD = 0.8, 0.0
#alphaR, betaD = 0.78, 0.5
#alphaR, betaD = 0.78, 0.0
alphaR, betaD = 0.63, 0.0
#alphaR, betaD = 0.9, 0.5
#alphaR, betaD = 0.245, 0.245
#alphaR, betaD = 0.5, 0.245
#alphaR, betaD = 0.5, 0.0
#alphaR, betaD = 0.0, 0.5
#alphaR, betaD = 0.81, 0.23
#alphaR, betaD = 0.81, 0.0
#alphaR, betaD = 0.9, 0.08
#alphaR, betaD = 0.9, round(0.9/1.8,2)
#alphaR, betaD = 0.7, 0.0
#alphaR, betaD = 1.0, betaD_InAs_German
#alphaR, betaD = betaD_InAs_German, betaD_InAs_German
#alphaR, betaD = 10*betaD_InAs_German, 0.0
#alphaR, betaD = round(sqrt(12)*betaD_InAs_German, 2), betaD_InAs_German
#alphaR, betaD = 10*betaD_InAs_German, betaD_InAs_German
#alphaR, betaD = 10*betaD_InAs_German, 2*betaD_InAs_German
#alphaR, betaD = 10*betaD_InAs_German, 3*betaD_InAs_German
#alphaR, betaD = 10*betaD_InAs_German, 4*betaD_InAs_German
#alphaR, betaD = 10*betaD_InAs_German, 5*betaD_InAs_German
#alphaR, betaD = 0.0, 0.0
#alphaR, betaD = round(49/R/ts,2), 0.0

if __name__ == "__main__":
    '''
    print(f"InAs: alphaR = {round(hbar**2/2/meff/R*alphaR_InAs)}, betaD = {round(hbar**2/2/meff/R*betaD_InAs)}")
    print(f"InX: alphaR = {round(hbar**2/2/meff/R*alphaR_InX)}, betaD = {round(hbar**2/2/meff/R*betaD_InX)}")
    print(f"InSb: alphaR = {round(hbar**2/2/meff/R*alphaR_InSb)}, betaD = {round(hbar**2/2/meff/R*betaD_InSb)}")
    print(f"Strong: alphaR = {round(hbar**2/2/meff/R*0.9)}, betaD = {round(hbar**2/2/meff/R*0.5)}")

    print(f"Kokurin: alphaR = {round(hbar**2/2/meff/42*0.9)}, betaD = {round(hbar**2/2/meff/42*0.4)}")
    print(f"Kokurin: alphaR = {round(hbar**2/2/meff/37*0.9)}, betaD = {round(hbar**2/2/meff/37*0.4)}")
    print(f"New: alphaR = {round(hbar**2/2/meff/30*0.8)}, betaD = {round(hbar**2/2/meff/30*0.08)}")
    print(ts)
    '''

    print(R*ts*0.59)

