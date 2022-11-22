'''
Created on Apr 5, 2019

@author: rene
'''
from _collections import defaultdict
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pprint
import pandas as pd
#import seaborn as sns

ch2018_basepath = os.path.join("mean_grid_v2")

simulations = [("CLMCOM-CCLM4_ECEARTH_EUR11", "RCP45"),
               ("CLMCOM-CCLM4_ECEARTH_EUR11", "RCP85"),
               ("CLMCOM-CCLM4_HADGEM_EUR11", "RCP45"),
               ("CLMCOM-CCLM4_HADGEM_EUR11", "RCP85"),
               ("CLMCOM-CCLM4_HADGEM_EUR44", "RCP85"),
               ("CLMCOM-CCLM4_MPIESM_EUR11", "RCP45"),
               ("CLMCOM-CCLM4_MPIESM_EUR44", "RCP45"),
               ("CLMCOM-CCLM4_MPIESM_EUR11", "RCP85"),
               ("CLMCOM-CCLM4_MPIESM_EUR44", "RCP85"),
               ("CLMCOM-CCLM5_ECEARTH_EUR44", "RCP85"),
               ("CLMCOM-CCLM5_MIROC_EUR44", "RCP85"),
               ("CLMCOM-CCLM5_MPIESM_EUR44", "RCP85"),
               ("CLMCOM-CCLM5_HADGEM_EUR44", "RCP85"),
               ("DMI-HIRHAM_ECEARTH_EUR11", "RCP26"),
               ("DMI-HIRHAM_ECEARTH_EUR11", "RCP45"),
               ("DMI-HIRHAM_ECEARTH_EUR44", "RCP45"),
               ("DMI-HIRHAM_ECEARTH_EUR11", "RCP85"),
               ("DMI-HIRHAM_ECEARTH_EUR44", "RCP85"),
               ("ICTP-REGCM_HADGEM_EUR44", "RCP85"),
               ("KNMI-RACMO_ECEARTH_EUR44", "RCP45"),
               ("KNMI-RACMO_ECEARTH_EUR44", "RCP85"),
               ("KNMI-RACMO_HADGEM_EUR44", "RCP26"),
               ("KNMI-RACMO_HADGEM_EUR44", "RCP45"),
               ("KNMI-RACMO_HADGEM_EUR44", "RCP85"),
               ("MPICSC-REMO1_MPIESM_EUR11", "RCP26"),
               ("MPICSC-REMO1_MPIESM_EUR44", "RCP26"),
               ("MPICSC-REMO1_MPIESM_EUR11", "RCP45"),
               ("MPICSC-REMO1_MPIESM_EUR44", "RCP45"),
               ("MPICSC-REMO1_MPIESM_EUR11", "RCP85"),
               ("MPICSC-REMO1_MPIESM_EUR44", "RCP85"),
               ("MPICSC-REMO2_MPIESM_EUR11", "RCP26"),
               ("MPICSC-REMO2_MPIESM_EUR44", "RCP26")]

periods = [("ref", 1981, 2010),
           ("2030", 2020, 2049),
           ("2060", 2045, 2074),
           ("2085", 2070, 2099)]

rcp_sims = defaultdict(list)
for sim, rcp in simulations:
    rcp_sims[rcp].append(sim)

temp_data = {}
var = "tas"
for sim, rcp in simulations:
    for period, _, _ in periods:
        fname = fname = "{}_{}_{}_{}_v2.tif".format(var, sim, rcp, period)
        temp_path = os.path.join(ch2018_basepath, fname)
        with rasterio.open(temp_path) as src:
            fwd = src.transform
            data = src.read()
            temp_data[(sim, rcp, period)] = (fwd, data)

if __name__ == '__main__':

    loc = (7.43975, 46.94753)
#     loc = (9.02435, 45.83218)
    d = []

    bad_sim = set()

    for period, _, _ in periods:
        for rcp in rcp_sims:
            for sim in rcp_sims[rcp]:
                fwd, data = temp_data[(sim, rcp, period)]

                i, j = loc * ~fwd

#                 print(data.shape)
# 
#                 print(sim, rcp, period, i, j, data[:, int(j), int(i)])

                mean = np.mean(data[:, int(j), int(i)])

                if mean < 0:
                    bad_sim.add(sim)
#                 if sim == "CLMCOM-CCLM4_HADGEM_EUR11":

                if sim in set(['CLMCOM-CCLM4_HADGEM_EUR11', 'ICTP-REGCM_HADGEM_EUR44', 'KNMI-RACMO_HADGEM_EUR44', 'CLMCOM-CCLM5_HADGEM_EUR44', 'CLMCOM-CCLM4_HADGEM_EUR44']):
                    print(rcp, period, sim, mean)

#                 mean = np.mean(data[np.where(data > -500)])
#                 if mean < 0:

                d.append({'rcp': rcp, 'mean': mean, 'period': period})
    df = pd.DataFrame(d)
    
    print(bad_sim)




