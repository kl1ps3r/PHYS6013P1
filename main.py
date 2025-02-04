import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import consts

def calc_radius(red_rad):
    # Calculate the physical radius from the reduced radius
    return consts.R_0 * red_rad / consts.SOLAR_RADIUS

def calc_mass(red_mass):
    # Calculate the physical mass from the reduced mass
    return consts.M_0 * red_mass / consts.SOLAR_MASS

def gamma(rho):
    return np.sign(rho) * np.abs(rho)**(2/3) / (3 * (1 + np.sign(rho) * np.abs(rho)**(2/3))**0.5)

def dtheta_dx(x, theta, mu):
    if x < 1e-4:
        return - x * theta / (gamma(theta)) 
    else:
        return - mu * theta / (gamma(theta) * x**2)

def dmu_dx(x, theta):
    return 3 * x**2 * theta

def q(x, y):
    return np.array([dtheta_dx(x, y[0], y[1]), dmu_dx(x, y[0])])

def event(t, y):
    return y[0]
event.terminal = True

t_min = 0
t_max = 10

#[theta_0, mu_0]
q_0 = np.array([1, 0])       
#t_eval = np.linspace(t_min, t_max, 1000)

"""
sol1 = integrate.solve_ivp(q, [t_min, t_max], q_0, method="RK45", t_eval=t_eval, dense_output=True, events=event)
sol2 = integrate.solve_ivp(q, [t_min, t_max], 2*q_0, method="RK45", t_eval=t_eval, dense_output=True, events=event)
"""
"""
sols = []
for i in range(10, 1000, 10):
    sols.append(integrate.solve_ivp(q, [t_min, t_max], i*q_0, t_eval=t_eval, events=event))
    print(sols[-1])

for sol in sols:
    plt.plot(sol.t, sol.y[0])
plt.show()
for sol in sols:
    plt.plot(sol.t, sol.y[1])
plt.show()
"""

def get_mass_radius(theta_0):
    sol = integrate.solve_ivp(q, [t_min, t_max], theta_0*q_0, max_step=0.001, events=event)
    mass = calc_mass(sol.y_events[0].T[1])
    radius = calc_radius(sol.t_events[0])
    return np.array([radius, mass])

initial_densities = np.logspace(-2, 4, 50)
data = []

for theta_0 in initial_densities:
    data.append(get_mass_radius(theta_0))

data = np.array(data)

wds = np.loadtxt("white_dwarfs.csv", unpack=True, delimiter=",")

#plt.plot(data[:, 1], data[:, 0])
#plt.show()
plt.plot(data[:, 1], data[:, 0])
plt.errorbar(wds[0], wds[2], xerr=wds[1], yerr=wds[3], fmt=".") 
plt.xlabel("Mass")
plt.ylabel("Radius")
plt.show()
