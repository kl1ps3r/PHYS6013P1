import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import main

initial_density = 1

sols = []
event_densities, event_masses, masses, radii = [], [], [], []
for i in range(2, 10):
    sols.append(solve_ivp(main.q, [0,10], [initial_density, 0], dense_output=True, 
                          events=main.event, rtol=10**(-i), atol=1e-6, max_step=1, args=(initial_density,)))

    #print(sols[-1].t_events, sols[-1].y_events)

    event_densities.append(sols[-1].y_events[0][0,0])
    event_masses.append(sols[-1].y_events[0][0,1])

    #print(sol.y[1])
    
    masses.append(sols[-1].y[1])
    radii.append(sols[-1].t)

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
#print(radii, masses)
#plt.plot(event_masses)
for i, (mass, radius) in enumerate(zip(masses, radii)):
    ax.plot(radius, mass, label=f"1e-{i+2}")
ax.legend()
ax.set_xlabel("Dimensionless radius")
ax.set_ylabel("Dimensionless mass")
plt.savefig("errors.png")

"""
sol = solve_ivp(main.q, [0,10], [initial_density, 0], dense_output=True, 
                          events=main.event, rtol=1e-1, atol=1e-3)
print(type(sol.y[1]))
plt.plot(sol.t, sol.y[1])
plt.show()"""