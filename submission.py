"""
A program to modal the masses and radii of white dwarfs based on a simple
internal structure model using the lane-embden equation and mass continuity 
equation.

This version of the code is designed to be self contained for submission. 
For the full version of the code including the used data, please see my github repository:
https://github.com/kl1ps3r/PHYS6013P1

This requires the following packages:
- numpy
- matplotlib
- scipy

Author: Sam Ecclestone-Brown
Date created: 28/01/2025
Licence: see licence
"""

# import necessary libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

class Consts:
    """
    Constants used in the model.
    """
    SPEED_OF_LIGHT = 2.99792458e8
    H_BAR = 1.05457266e-34
    ELECTRON_MASS  = 9.1093897e-31
    PROTON_MASS = 1.6726231e-27
    GRAVITATION = 6.67259e-11

    SOLAR_MASS = 1.98e30
    SOLAR_RADIUS = 6.95e8
    EARTH_RADIUS = 6.378e6

    RHO_0 = PROTON_MASS * ELECTRON_MASS**3 * SPEED_OF_LIGHT**3 / (3 * np.pi**2 * H_BAR**3)

    R_0 = (3*ELECTRON_MASS * SPEED_OF_LIGHT**2 / (4 * np.pi * GRAVITATION * RHO_0 * PROTON_MASS))**0.5

    M_0 = 4 * np.pi * R_0**3 * RHO_0 / 3

consts = Consts()

def gamma(theta):
    """
    Evalutes gamma(theta).

    Parameters
    ----------
    theta : float
        The dimensionless density.

    Returns
    -------
    float
        The evaluated gamma value.

    Notes
    -----
    Due to numpy not liking computing negative numbers to any scalar power, 
    np.sign and np.abs is used to circumvent this.
    """

    return np.abs(theta)**(2/3) / (3 * (1 + np.abs(theta)**(2/3))**0.5)


def dtheta_dx(x, theta, mu):
    """
    Evalutates dtheta/dx.

    Parameters
    ----------
    x : float
        The dimensionless radius.
    theta : float
        The dimensionless density.
    mu : float
        The dimensionless enclosed mass.

    Returns
    -------
    float
        The evaluated dtheta/dx value.
    """
 
    if x < 1e-4:
        return -1 * x * theta**2 / (gamma(theta)) 
    else:
        return -1 * mu * theta / (gamma(theta) * x**2)


def dmu_dx(x, theta):
    """
    Evalutates dmu/dx.

    Parameters
    ----------
    x : float
        The dimensionless radius.
    theta : float
        The dimensionless density.

    Returns
    -------
    float
        The evaluated dmu/dx value.
    """
    return 3 * x**2 * theta


def q(x, y, theta_0):
    """
    Paramaterises the system of ODEs.

    Parameters
    ----------
    x : float
        The dimensionless radius.
    y : ndarray, shape (2,)
        The values of theta and mu (the dimensionless mass and density).

    Returns
    -------
    ndarray, shape (2,)
        The evaluated system of ODEs.
    """
    return np.array([dtheta_dx(x, y[0], y[1]), dmu_dx(x, y[0])])


def event(x, y, theta_0):
    """
    Event function for ODE solver to find when theta = 0.

    Parameters
    ----------
    x : float
        The dimensionless radius (not used but required by solve_ivp).
    y : ndarray, shape (2,)
        The values of theta and mu (the dimensionless mass and density).

    Returns
    -------
    float
        The value of theta.
    """
    return y[0] - 1e-3 * theta_0
# set the terminal flag, makes solve_ivp terminate on finding a root
event.terminal = True

      
def get_mass_radius(theta_0, x_min=0.0, x_max=40.0, terminator=event):
    """
    Estimates white dwarf mass and radius for given initial core density.

    Parameters
    ----------
    theta_0 : float
        The initial dimensionless core density.
    x_min : float, optional
        The lower integration bound, by default 0.0.
    x_max : float, optional
        The upper integration bound, by default 10.0.
    terminator : callable, optional
        Event function for integration termination, by default event.

    Returns
    -------
    ndarray, shape (2,)
        The estimated mass and radius of the white dwarf per Ye^2 and Ye
        respectively.

    Notes
    -----
    Solves the IVP for increasing x until theta = 0, using scipy.integrate.solve_ivp's
    events to find roots between integration steps.
    """
    # the inital values of theta and mu
    q_0 = np.array([theta_0, 0])
    
    # obtaining the solution to the ODE
    sol = integrate.solve_ivp(q, [x_min, x_max], q_0, atol= 1e-8, rtol=1e-6,
                              events=terminator, args=(theta_0,))
    
    # Calculating the mass and radius of the white dwarf
    mass = calc_mass(sol.y_events[0].T[1])
    radius = calc_radius(sol.t_events[0])
    
    return np.array([mass, radius])


def calc_radius(x):
    """
    Calculate the physical radius per solar radius.

    Parameters
    ----------
    x : float
        The dimensionless radius.

    Returns
    -------
    float
        The enclosed radius per solar radius.
    """
    return consts.R_0 * x / consts.SOLAR_RADIUS


def calc_mass(mu):
    """
    Calculate the physical mass per solar mass.

    Parameters
    ----------
    mu : float
        The dimensionless mass.

    Returns
    -------
    float
        The enclosed mass per solar mass.
    """
    return consts.M_0 * mu / consts.SOLAR_MASS


def apply_Ye_scale(data, Ye=0.5):
    """
    Scale masses and radii by electron fraction (Ye).

    Parameters
    ----------
    data : ndarray, shape (N, 2)
        Input data of mass and radii.
    Ye : float, optional
        The electron fraction value to scale with, by default 0.5.

    Returns
    -------
    ndarray, shape (N, 2)
        The scaled masses and radii.

    Notes
    -----
    Mass is proportional to R^3 * rho => Ye^2.
    """ 
    data[:, 0] = data[:, 0] * Ye**2
    data[: ,1] = data[:, 1] * Ye
    
    return data

if __name__=="__main__":
    # ~~~~~ Make initial graphs ~~~~~

    q_0 = np.array([15, 0])
    sol = integrate.solve_ivp(q, [0, 10], q_0, atol=1e-10, rtol=1e-8, 
                              dense_output=True, events=event, args=(q_0[0],),
                              method="RK45")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sol.t, sol.y[0], label="Density")
    ax.set_xlabel("Dimensionless radius")
    ax.set_ylabel("Dimensionless density")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(sol.t, sol.y[1], label="Mass")
    ax.set_xlabel("Dimensionless radius")
    ax.set_ylabel("Dimensionless mass")
    plt.show()

    # ~~~~~~~~~~~ Solving ~~~~~~~~~~~ 

    # create the initial core densities
    initial_densities = np.logspace(-2, 9.5, 150)
    
    data = []

    # estimate the mass and radius of the white dwarf for each initial density
    for theta_0 in initial_densities:
        data.append(get_mass_radius(theta_0))
    data = np.array(data)


    # scaling model data to have  Ye = 0.5, 0.46 corresponding to C/O core and Fe core white dwarfs respectively
    data_5 = apply_Ye_scale(data.copy())
    data_46 = apply_Ye_scale(data.copy(), 0.46)

    # Printing the average and standard deviation of the last 20 points of the models
    print("~~~~~~~~ Maximum Mass estiamtes ~~~~~~~~")
    print(fr"Ye =  0.5: {np.average(data_5[-20:, 0]):.6f} +/- {np.std(data_5[-20:, 0]):.6f} M_solar")
    print(fr"Ye = 0.46: {np.average(data_46[-20:, 0]):.6f} +/- {np.std(data_46[-20:, 0]):.6f} M_solar")

    # ~~~~~~~~~~~ Plotting ~~~~~~~~~~~

    fig, ax = plt.subplots(figsize=(8, 8))

    # plotting Mass vs Radius models for Ye = 0.5, 0.46 respectively
    ax.plot(data_5[:, 0], data_5[:, 1], label="Ye = 0.5")
    ax.plot(data_46[:, 0], data_46[:, 1], label="Ye = 0.46")

    # make graph pretty stuff
    ax.set_xlabel(r"Mass $\frac{M}{M_\odot}$")
    ax.set_ylabel(r"Radius $\frac{R}{R_\odot}$")

    ax.legend()
    plt.show()
    

    