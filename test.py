import pandas as pd

wds_pd = pd.read_csv("parsons_wites.csv")
print(wds_pd[["Mass_Msun", "Mass_err", "Radius_Rsun", "Radius_err"]].to_numpy().T)