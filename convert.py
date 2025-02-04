import numpy as np
import consts
data = np.loadtxt("add_wd.csv", delimiter=",")

data[:, 2] *= consts.EARTH_RADIUS / consts.SOLAR_RADIUS
data[:, 3] *= consts.EARTH_RADIUS / consts.SOLAR_RADIUS
data = np.append(data, np.loadtxt("white_dwarfs.csv", delimiter=","), axis=0)

np.savetxt("add_wd_corr.csv", data, delimiter=",")
