"""
A program to modal the masses and radii of white dwarfs based on a simple
internal structure model using the lane-embden equation and mass continuity 
equation.

Author: Sam Ecclestone-Brown
Date created: 28/01/2025
Licence: see licence
"""

# import necessary libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import consts


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
    return np.sign(theta) * np.abs(theta)**(2/3) / (3 * (1 + np.sign(theta) * np.abs(theta)**(2/3))**0.5)


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
        return - x * theta / (gamma(theta)) 
    else:
        return - mu * theta / (gamma(theta) * x**2)


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


def q(x, y):
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


def event(x, y):
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
    return y[0]
# set the terminal flag, makes solve_ivp terminate on finding a root
event.terminal = True

      
def get_mass_radius(theta_0, x_min=0.0, x_max=10.0, terminator=event):
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
    sol = integrate.solve_ivp(q, [x_min, x_max], q_0, max_step=0.001, 
                              events=terminator)
    
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

    # ~~~~~~~~~~~ Solving ~~~~~~~~~~~ 

    # create the initial core densities
    initial_densities = np.logspace(-1, 4, 30)
    data = []

    # estimate the mass and readius of the white dwarf for each radius
    for theta_0 in initial_densities:
        data.append(get_mass_radius(theta_0))

    data = np.array(data)

    # load in measurements of actual white dwarfs
    wds = np.loadtxt("add_wd_corr.csv", unpack=True, delimiter=",")

    # scaling model data to have Ye = 0.5, 0.46 respectively
    data_5 = apply_Ye_scale(data.copy())
    data_46 = apply_Ye_scale(data.copy(), 0.46)

    # ~~~~~~~~~~~ Plotting ~~~~~~~~~~~

    # plotting Mass vs Radius models for Ye = 0.5, 0.46 respectively
    plt.plot(data_5[:, 0], data_5[:, 1])
    plt.plot(data_46[:, 0], data_46[:, 1])

    # plotting observed white dwarf measurements with their uncertainties
    plt.errorbar(wds[0], wds[2], xerr=wds[1], yerr=wds[3], fmt=".") 

    # make graph pretty stuff
    plt.xlabel(r"Mass $\frac{M}{M_\odot}$")
    plt.ylabel("Radius")

    plt.show()
    #plt.savefig("YES.png")
