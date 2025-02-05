import numpy as np
import consts
data = np.loadtxt("add_wd.csv", delimiter=",")

data[:, 2] *= consts.EARTH_RADIUS / consts.SOLAR_RADIUS
data[:, 3] *= consts.EARTH_RADIUS / consts.SOLAR_RADIUS
data = np.append(data, np.loadtxt("white_dwarfs.csv", delimiter=","), axis=0)

data_2 = np.loadtxt("trembley_whites.csv", delimiter=",")
data_2 = np.hstack((data_2, data_2[:, 2].reshape(-1, 1)*0.02))

data = np.append(data, data_2, axis=0)

np.savetxt("add_wd_corr.csv", data, delimiter=",")
