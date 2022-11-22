'''
Created on 11.10.2017

@author: rene
'''
from collections import defaultdict
import fiona
import gzip
import json
import logging
import multiprocessing
import os
import queue
import random
import time
from pint import UnitRegistry
import psycopg2
import psycopg2.extras
import rasterio
from shapely.affinity import rotate
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from secrets import dbstring
from pyheat.sia380 import SIAConf, reduktionsfaktor_erdreich_wand, \
    reduktionsfaktor_erdreich_boden
from pyheat.sia380 import SIAPeopleArea, SIAHeatPersonQp, SIAOccupation_tp, \
    SIABuildingType, SIAElectricityRed_fel, SIAAussenluftvolumenstrom_VAE, \
    get_verschattungsfaktor_horizont
import pyheat.sia380
from wallbaum import *
from pyproj import Transformer

"""
The following variables are used to setup the model
"""

# How many times one building should be simulated
number_of_runs = 1

# CH2018 RCP scenario Values: ["RCP26", "RCP45", "RCP85"]
sim_rcp = "RCP26"

# CH2018 periods Values: ["ref", "2030", "2060", "2085"]
sim_period = "ref"

# Path were output data should be written to
out_base_dir = r"output"

# Path to logfile
logfile_path = os.path.join(out_base_dir, r"beef_ch2018.log")

# How many CPU cores should be used
cpus = multiprocessing.cpu_count() - 1

# Path to solar data
sol_cache_path = "data/solar_cache.gz"

# Path to municipalities data
mun_path = r"data/swissBOUNDARIES3D_1_1_TLM_HOHEITSGEBIET.shp"

# Path to solar grid
sol_path = r"data/SISin201312312330-CH.nc"

# PAth to CH2018 temperature data
ch2018_basepath = os.path.join("ch2018", "mean_grid_v2")

"""
End
"""

if not os.path.exists(out_base_dir):
    os.mkdir(out_base_dir)

proj_lv03 = "EPSG:21781"
proj_wg84 = "EPSG:4326"

transformer_lv03_to_wg84 = Transformer.from_crs(proj_lv03, proj_wg84)

ureg = UnitRegistry()

days = [
    31.0,
    28.0,
    31.0,
    30.0,
    31.0,
    30.0,
    31.0,
    31.0,
    30.0,
    31.0,
    30.0,
    31.0]


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=logfile_path,
                    filemode='a')



# Read temperature data
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

rcp_sims = defaultdict(list)
for sim, rcp in simulations:
    rcp_sims[rcp].append(sim)


periods = [("ref", 1981, 2010),
           ("2030", 2020, 2049),
           ("2060", 2045, 2074),
           ("2085", 2070, 2099)]

temp_data = {}
var = "tas"
for sim, rcp in simulations:
    for period, _, _ in periods:
        fname = fname = "{}_{}_{}_{}_v2.tif".format(var, sim, rcp, period)
        temp_path = os.path.join(ch2018_basepath, fname)
        with rasterio.open(temp_path) as src:
            fwd = src.transform
            data = src.read()
            temp_data[(sim, rcp, period)] = (fwd, data, src.meta['nodata'])

# Read Solar data
solar_cache = {}
with gzip.open(sol_cache_path, 'r') as f:
    for line in f:
        d = json.loads(line.decode())
        h = d['h']
        w = d['w']
        solar_cache[(w, h)] = d


with rasterio.open(sol_path) as src:
    sol_fwd = ~src.transform


def unit_vector(vector):
    """ Returns the unit vector of the vector.
    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python"""

    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    a = unit_vector(v1)
    b = unit_vector(v2)

    angle = np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
    if np.isnan(angle):
        if (a == b).all():
            return 0.0
        else:
            return np.pi
    return angle


def get_walls(mar_ratio, mar_angle_rad, area, warea, floor_height=2.8):
    """
    MAR: minimum-area enclosing rectangle

    a := length of long side of mar rectancle
    b := length of short side of mar rectancle

    a * b = area
    a / b = mar_ratio

    ==>
    a = sqrt(area) * sqrt(mar_ratio)
    b = sqrt(area) / sqrt(mar_ratio)
    """

    a = np.sqrt(area) * np.sqrt(mar_ratio)
    b = np.sqrt(area) / np.sqrt(mar_ratio)

    # Estimate number of floors based on flat areas
    floors = area / warea
    volume = a * b * floors * floor_height

    # Calculate angles for side of building
    mar_angle = 90.0 - np.rad2deg(mar_angle_rad)
    x = 0.0
    y = 0.0

    #  Calculate orientation
    if mar_ratio >= 1:
        ls = LineString([(x, y),
                         (x,
                          y + 10)])
        a1, b1 = a, b
    else:
        ls = LineString([(x, y),
                         (x + 10,
                          y)])
        a1, b1 = b, a

    ls_rot = rotate(
        ls, -mar_angle, origin=Point(x, y), use_radians=False)

    pt1 = ls_rot.coords[0]
    pt2 = ls_rot.coords[-1]
    aa = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
    bb = [0, -1]

    ang2 = np.rad2deg(angle_between(aa, bb))

    # Normalize angles to 0 -> 180, beacuse for rectancles that doesent matter and makes calc easier
    if ang2 < 0:
        ang2 += 180.0

    # calc other side
    ang_opposite = ang2 + 90.0

    if ang_opposite > 180:
        ang_opposite -= 180.0

    ang2_round = round(ang2 / 15.0) * 15.0
    ang2_opposite_round = round(ang_opposite / 15.0) * 15.0

    walls = {}
    walls[ang2_round] = b1 * floors * floor_height
    walls[ang2_opposite_round] = a1 * floors * floor_height
    walls[ang2_round - 180.0] = b1 * floors * floor_height
    walls[ang2_opposite_round - 180.0] = a1 * floors * floor_height

    return walls, volume, a, b


def process_mun(bfsnr, period, rcp, outdir):

    # get all building data for municipality
    conn = psycopg2.connect(dbstring)
    c = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    sql = generate_sql(bfsnr)
    print(sql)
    c.execute(sql)

    # Process all buildings
    q = multiprocessing.Queue()
    outq = multiprocessing.Queue()

    ps = [
        multiprocessing.Process(
            target=worker,
            args=(
                q, outq, rcp, period,
                i)) for i in range(cpus)]

    pw = multiprocessing.Process(
        target=writer,
        args=(outq, bfsnr, outdir))
    ps.append(pw)

    for p in ps:
        p.start()

    t = 0
    for row in c:
        t += 1
        try:
            data = {}

            data = dict(row)
#             for key in row.keys():
#                 data[key] = row[key]
            bkey = (row['btype'], row['bid'])
#             if bkey in allowed_bkeys:
            q.put(data)
        except Exception as e:
            print(str(e), "!!!")

    for _ in range(cpus):
        q.put('killitwithfire_worker')

    for p in ps:
        p.join()


def get_avg_slope(gkat, gbaup):
    """
    slops from following query:
SELECT gkat,
       gbaup,
       avg(b.slopeavg) as avgslope
FROM   heat.buildinginfo AS b,
       heat.gwr AS gwr,
       heat.gwrmatches2 AS gm
WHERE  b.bid = gm.bid
       AND b.btype = gm.btype
       AND gm.gwrid = gwr.gwrid
group by gwr.gkat, gwr.gbaup
order by gwr.gkat, gwr.gbaup
    """

    avg_slope = {}
    avg_slope[(0, 0)] = 15.5300275
    avg_slope[(1010, 0)] = 19.9050641474359
    avg_slope[(1010, 8011)] = 34.1084
    avg_slope[(1010, 8017)] = 13.3657
    avg_slope[(1021, 0)] = 25.4395236144578
    avg_slope[(1021, 8011)] = 32.8658835969203
    avg_slope[(1021, 8012)] = 33.6043755799038
    avg_slope[(1021, 8013)] = 26.9191257689555
    avg_slope[(1021, 8014)] = 22.3441899414525
    avg_slope[(1021, 8015)] = 24.1716564912901
    avg_slope[(1021, 8016)] = 28.8157756864219
    avg_slope[(1021, 8017)] = 29.2531463449444
    avg_slope[(1021, 8018)] = 30.4916838487736
    avg_slope[(1021, 8019)] = 29.6694166639136
    avg_slope[(1021, 8020)] = 24.0231922454795
    avg_slope[(1021, 8021)] = 19.5413031037272
    avg_slope[(1021, 8022)] = 19.1415948347581
    avg_slope[(1025, 0)] = 27.0830329411765
    avg_slope[(1025, 8011)] = 33.1523275548547
    avg_slope[(1025, 8012)] = 33.3493508893338
    avg_slope[(1025, 8013)] = 25.3929008619639
    avg_slope[(1025, 8014)] = 20.106621047987
    avg_slope[(1025, 8015)] = 20.3107104568338
    avg_slope[(1025, 8016)] = 25.6599808975988
    avg_slope[(1025, 8017)] = 27.2620168781767
    avg_slope[(1025, 8018)] = 29.0205121062806
    avg_slope[(1025, 8019)] = 26.8486109904723
    avg_slope[(1025, 8020)] = 22.260362099214
    avg_slope[(1025, 8021)] = 19.391502160637
    avg_slope[(1025, 8022)] = 19.7833159174421
    avg_slope[(1030, 0)] = 26.7203523529412
    avg_slope[(1030, 8011)] = 34.0105884182909
    avg_slope[(1030, 8012)] = 33.1707572798362
    avg_slope[(1030, 8013)] = 27.3022945429909
    avg_slope[(1030, 8014)] = 22.2712296431082
    avg_slope[(1030, 8015)] = 23.4299156521023
    avg_slope[(1030, 8016)] = 27.4754489640063
    avg_slope[(1030, 8017)] = 29.1376595835716
    avg_slope[(1030, 8018)] = 29.6305426551072
    avg_slope[(1030, 8019)] = 27.9599389404669
    avg_slope[(1030, 8020)] = 24.0280939774475
    avg_slope[(1030, 8021)] = 20.9405127853774
    avg_slope[(1030, 8022)] = 21.9000333911735
    avg_slope[(1040, 0)] = 22.17387845
    avg_slope[(1040, 8011)] = 32.6134795964637
    avg_slope[(1040, 8012)] = 30.412409644375
    avg_slope[(1040, 8013)] = 25.1991505316156
    avg_slope[(1040, 8014)] = 20.7813869563264
    avg_slope[(1040, 8015)] = 20.6891546926921
    avg_slope[(1040, 8016)] = 22.8978933362828
    avg_slope[(1040, 8017)] = 23.6804928157233
    avg_slope[(1040, 8018)] = 23.8869346644087
    avg_slope[(1040, 8019)] = 23.175836331448
    avg_slope[(1040, 8020)] = 18.3773099350428
    avg_slope[(1040, 8021)] = 15.790879847541
    avg_slope[(1040, 8022)] = 16.7717028811659
    avg_slope[(1060, 0)] = 22.521181332013
    avg_slope[(1060, 8011)] = 30.8094603192383
    avg_slope[(1060, 8012)] = 29.306118666036
    avg_slope[(1060, 8013)] = 24.581364490381
    avg_slope[(1060, 8014)] = 20.4552399192752
    avg_slope[(1060, 8015)] = 19.7517610946774
    avg_slope[(1060, 8016)] = 20.7188467260423
    avg_slope[(1060, 8017)] = 21.0898104735037
    avg_slope[(1060, 8018)] = 21.2365056292896
    avg_slope[(1060, 8019)] = 19.8308558955187
    avg_slope[(1060, 8020)] = 19.1604929794732
    avg_slope[(1060, 8021)] = 17.2226042705376
    avg_slope[(1060, 8022)] = 19.2442630192157
    avg_slope[(1080, 0)] = 29.0132292463136
    avg_slope[(1080, 8011)] = 32.9057601606425
    avg_slope[(1080, 8012)] = 32.4961627030114
    avg_slope[(1080, 8013)] = 27.2592693827161
    avg_slope[(1080, 8014)] = 23.3464506545031
    avg_slope[(1080, 8015)] = 24.7323318485597
    avg_slope[(1080, 8016)] = 26.2479866219512
    avg_slope[(1080, 8017)] = 26.3506210653061
    avg_slope[(1080, 8018)] = 26.7602672380952
    avg_slope[(1080, 8019)] = 26.02517725369
    avg_slope[(1080, 8020)] = 25.606957991242
    avg_slope[(1080, 8021)] = 23.5605023961312
    avg_slope[(1080, 8022)] = 24.7968045348838

    return avg_slope[(gkat, gbaup)]


def sample_flat_roof(gkat, gbaup):
    """
    Stats retrieved from following query

SELECT gkat,
       gbaup,
       (count(*) filter (where b.slope50 < 5))::float / (count(*))::float as flat_roof_share
FROM   heat.buildinginfo AS b,
       heat.gwr AS gwr,
       heat.gwrmatches2 AS gm
WHERE  b.bid = gm.bid
       AND b.btype = gm.btype
       AND gm.gwrid = gwr.gwrid
group by gwr.gkat, gwr.gbaup
    """
    flat_roof_shares = {}
    flat_roof_shares[(1080, 8019)] = 0.1963133640553
    flat_roof_shares[(1025, 8020)] = 0.3215221287743
    flat_roof_shares[(1080, 8017)] = 0.164625850340136
    flat_roof_shares[(1060, 8022)] = 0.377450980392157
    flat_roof_shares[(1040, 8011)] = 0.033010093014051
    flat_roof_shares[(1080, 8014)] = 0.238169123351435
    flat_roof_shares[(1040, 8018)] = 0.215131830340084
    flat_roof_shares[(1030, 8021)] = 0.357378595002357
    flat_roof_shares[(1040, 8016)] = 0.222550675675676
    flat_roof_shares[(1030, 8015)] = 0.210048731305663
    flat_roof_shares[(1080, 8011)] = 0.076305220883534
    flat_roof_shares[(1040, 8017)] = 0.214357429718876
    flat_roof_shares[(0, 0)] = 0.25
    flat_roof_shares[(1040, 8019)] = 0.220588235294118
    flat_roof_shares[(1060, 8014)] = 0.301434281683517
    flat_roof_shares[(1030, 8012)] = 0.025162744116174
    flat_roof_shares[(1030, 8013)] = 0.065186339436437
    flat_roof_shares[(1025, 0)] = 0.073529411764706
    flat_roof_shares[(1030, 8020)] = 0.267793138760881
    flat_roof_shares[(1010, 8011)] = 0
    flat_roof_shares[(1021, 0)] = 0.052208835341366
    flat_roof_shares[(1040, 8022)] = 0.477578475336323
    flat_roof_shares[(1060, 8011)] = 0.0933846073897
    flat_roof_shares[(1080, 8013)] = 0.175925925925926
    flat_roof_shares[(1025, 8018)] = 0.078011592083964
    flat_roof_shares[(1080, 8012)] = 0.105919003115265
    flat_roof_shares[(1021, 8018)] = 0.029232535547117
    flat_roof_shares[(1025, 8019)] = 0.144167521573834
    flat_roof_shares[(1025, 8017)] = 0.094123006833713
    flat_roof_shares[(1080, 8020)] = 0.192368839427663
    flat_roof_shares[(1021, 8016)] = 0.028467743451855
    flat_roof_shares[(1025, 8016)] = 0.139663503522152
    flat_roof_shares[(1060, 8021)] = 0.411249628970021
    flat_roof_shares[(1021, 8017)] = 0.028023973575261
    flat_roof_shares[(1021, 8019)] = 0.051460989250085
    flat_roof_shares[(1060, 8015)] = 0.316194137433926
    flat_roof_shares[(1060, 8013)] = 0.190504852907983
    flat_roof_shares[(1030, 8014)] = 0.242526964560863
    flat_roof_shares[(1060, 8012)] = 0.125610048803904
    flat_roof_shares[(1025, 8022)] = 0.396751169086882
    flat_roof_shares[(1060, 8020)] = 0.357318357318357
    flat_roof_shares[(1030, 8011)] = 0.011367824564637
    flat_roof_shares[(1040, 0)] = 0.25
    flat_roof_shares[(1080, 8015)] = 0.205592105263158
    flat_roof_shares[(1080, 8021)] = 0.273338940285955
    flat_roof_shares[(1021, 8022)] = 0.329733529298997
    flat_roof_shares[(1030, 8018)] = 0.074320676624078
    flat_roof_shares[(1030, 8017)] = 0.070927513639907
    flat_roof_shares[(1030, 8019)] = 0.11901983663944
    flat_roof_shares[(1040, 8021)] = 0.505323505323505
    flat_roof_shares[(1030, 8016)] = 0.108678655199375
    flat_roof_shares[(1040, 8015)] = 0.317027574388534
    flat_roof_shares[(1080, 0)] = 0.149406332453826
    flat_roof_shares[(1021, 8014)] = 0.090820336335457
    flat_roof_shares[(1040, 8013)] = 0.110632981676846
    flat_roof_shares[(1040, 8012)] = 0.067882682143227
    flat_roof_shares[(1025, 8014)] = 0.269349284031176
    flat_roof_shares[(1040, 8020)] = 0.430740037950664
    flat_roof_shares[(1025, 8011)] = 0.01891369442444
    flat_roof_shares[(1030, 8022)] = 0.349047141424273
    flat_roof_shares[(1060, 0)] = 0.221271095959691
    flat_roof_shares[(1010, 0)] = 0.215351812366738
    flat_roof_shares[(1021, 8011)] = 0.009752601449736
    flat_roof_shares[(1060, 8018)] = 0.260617760617761
    flat_roof_shares[(1025, 8015)] = 0.304704070014065
    flat_roof_shares[(1025, 8021)] = 0.409754147511493
    flat_roof_shares[(1060, 8016)] = 0.26363976083707
    flat_roof_shares[(1080, 8022)] = 0.166368515205725
    flat_roof_shares[(1021, 8021)] = 0.311047746162713
    flat_roof_shares[(1060, 8017)] = 0.265524625267666
    flat_roof_shares[(1010, 8017)] = 0
    flat_roof_shares[(1060, 8019)] = 0.303337453646477
    flat_roof_shares[(1021, 8015)] = 0.093679478946886
    flat_roof_shares[(1040, 8014)] = 0.278572440164283
    flat_roof_shares[(1025, 8012)] = 0.018373688710922
    flat_roof_shares[(1021, 8013)] = 0.017402176809396
    flat_roof_shares[(1025, 8013)] = 0.052649442958112
    flat_roof_shares[(1080, 8018)] = 0.179347826086957
    flat_roof_shares[(1021, 8012)] = 0.016088008436899
    flat_roof_shares[(1030, 0)] = 0.117647058823529
    flat_roof_shares[(1080, 8016)] = 0.194782608695652
    flat_roof_shares[(1021, 8020)] = 0.205043193162771

    return random.uniform(0, 1) <= flat_roof_shares[(gkat, gbaup)]


def sample_gbaup(gkat):
    """
    Weights based on following query

WITH total AS (
    SELECT gwr.gkat, count(*)::float as tot
    FROM heat.gwr
    where gwr.gbaup > 0
    GROUP BY gwr.gkat
)

SELECT gwr.gkat, gwr.gbaup,
       count(*)::float / tot as gbaup_share
FROM   heat.gwr AS gwr, total
where gwr.gkat=total.gkat and gwr.gbaup > 0
group by gwr.gkat, gwr.gbaup, total.tot
order by gwr.gkat, gwr.gbaup

    """

    gbaups = defaultdict(list)
    gbaups[1010].append((8011, 0.166666666666667))
    gbaups[1010].append((8013, 0.166666666666667))
    gbaups[1010].append((8017, 0.333333333333333))
    gbaups[1010].append((8022, 0.333333333333333))
    gbaups[1021].append((8011, 0.131131523392065))
    gbaups[1021].append((8012, 0.113232376044247))
    gbaups[1021].append((8013, 0.113818619894663))
    gbaups[1021].append((8014, 0.09812100530426))
    gbaups[1021].append((8015, 0.128630632072736))
    gbaups[1021].append((8016, 0.061517145034026))
    gbaups[1021].append((8017, 0.079529591282013))
    gbaups[1021].append((8018, 0.053435503301779))
    gbaups[1021].append((8019, 0.071835660465024))
    gbaups[1021].append((8020, 0.063999326443236))
    gbaups[1021].append((8021, 0.060964163246045))
    gbaups[1021].append((8022, 0.023784453519906))
    gbaups[1025].append((8011, 0.201183854107932))
    gbaups[1025].append((8012, 0.121407184595167))
    gbaups[1025].append((8013, 0.126863868284723))
    gbaups[1025].append((8014, 0.127998471852396))
    gbaups[1025].append((8015, 0.116436102026867))
    gbaups[1025].append((8016, 0.043773143726425))
    gbaups[1025].append((8017, 0.057878590513978))
    gbaups[1025].append((8018, 0.0490019861316))
    gbaups[1025].append((8019, 0.041869863502358))
    gbaups[1025].append((8020, 0.036325725582434))
    gbaups[1025].append((8021, 0.051800521503384))
    gbaups[1025].append((8022, 0.025460688172736))
    gbaups[1030].append((8011, 0.477410427678267))
    gbaups[1030].append((8012, 0.133451694790231))
    gbaups[1030].append((8013, 0.087000757988127))
    gbaups[1030].append((8014, 0.073631678816521))
    gbaups[1030].append((8015, 0.067176061819273))
    gbaups[1030].append((8016, 0.028376226643537))
    gbaups[1030].append((8017, 0.043490204656794))
    gbaups[1030].append((8018, 0.031255564090694))
    gbaups[1030].append((8019, 0.028549190377112))
    gbaups[1030].append((8020, 0.010850930697502))
    gbaups[1030].append((8021, 0.012041328157987))
    gbaups[1030].append((8022, 0.006765934283956))
    gbaups[1040].append((8011, 0.35751819952835))
    gbaups[1040].append((8012, 0.142776581564647))
    gbaups[1040].append((8013, 0.105852045524454))
    gbaups[1040].append((8014, 0.104480672613555))
    gbaups[1040].append((8015, 0.093676304726751))
    gbaups[1040].append((8016, 0.033528145186097))
    gbaups[1040].append((8017, 0.058392289551933))
    gbaups[1040].append((8018, 0.037770429611402))
    gbaups[1040].append((8019, 0.025146108889572))
    gbaups[1040].append((8020, 0.014944119758023))
    gbaups[1040].append((8021, 0.018032912949862))
    gbaups[1040].append((8022, 0.007882190095355))
    gbaups[1060].append((8011, 0.177128095161712))
    gbaups[1060].append((8012, 0.116731813154807))
    gbaups[1060].append((8013, 0.120556179348423))
    gbaups[1060].append((8014, 0.118242986551401))
    gbaups[1060].append((8015, 0.118648217406354))
    gbaups[1060].append((8016, 0.050763606892302))
    gbaups[1060].append((8017, 0.058184396923622))
    gbaups[1060].append((8018, 0.039636643000059))
    gbaups[1060].append((8019, 0.039037239027108))
    gbaups[1060].append((8020, 0.054419126896354))
    gbaups[1060].append((8021, 0.069589957028645))
    gbaups[1060].append((8022, 0.037061738609214))
    gbaups[1080].append((8011, 0.051278788649956))
    gbaups[1080].append((8012, 0.068647410612037))
    gbaups[1080].append((8013, 0.072655554141748))
    gbaups[1080].append((8014, 0.11031937905586))
    gbaups[1080].append((8015, 0.118335666115282))
    gbaups[1080].append((8016, 0.059931288968062))
    gbaups[1080].append((8017, 0.073291767400433))
    gbaups[1080].append((8018, 0.068711031937906))
    gbaups[1080].append((8019, 0.092569029138567))
    gbaups[1080].append((8020, 0.11070110701107))
    gbaups[1080].append((8021, 0.114709250540781))
    gbaups[1080].append((8022, 0.058849726428299))

    baups = [a[0] for a in gbaups[gkat]]
    ws = [a[1] for a in gbaups[gkat]]
    return np.random.choice(baups, 1, p=ws)[0]


def worker(q, outq, rcp, period, wid):

    finished = False
    while not finished:
        try:

            f = q.get()

            if f == 'killitwithfire_worker':
                logging.info(
                    "worker {} finished - recieved good kill".format(wid))
                finished = True
                outq.put("killitwithfire_thedb")
                break

            data = f

            data["sia_building_type"] = gkat2siabuildingtype(data)

            if data['gwr_ratio'] is None:
                data['gwr_ratio'] = 1.0
            else:
                data['gwr_ratio'] = float(data['gwr_ratio'])


            # Fix DSM data not available
            if data['delta10'] < 0.9:
                for i in np.arange(-180.0, 180.1, 15.0):
                    data["a_wall_{}".format(i).replace('.', '_').replace('-', 'm')] = 0.0
                    data["a_shared_{}".format(i).replace('.', '_').replace('-', 'm')] = 0.0

                warea = float(data['wareas'])
                if data['gkat'] == 1021:
                    warea = 1.15 * warea
                else:
                    warea = 1.2 * warea

                walls, volume, a, b = get_walls(mar_ratio=data['mar_ratio'],
                                                mar_angle_rad=data['mar_angle'],
                                                area=data['footprint_area'],
                                                warea=warea)

                for i in walls:
                    data["a_wall_{}".format(i).replace('.', '_').replace('-', 'm')] = walls[i]

                data['volume'] = volume

                # needs to be sampled in every run
                # data["flat_roof"] = sample_flat_roof(gkat=data['gkat'],
                #                                      gbaup=data['gbaup'])
                # See below

                data['ebf'] = warea

                # calc roof area from average slope
                slope_angle = get_avg_slope(gkat=data['gkat'],
                                            gbaup=data['gbaup'])

                data['a_roof'] = data['footprint_area'] / np.cos(np.deg2rad(slope_angle))

                data['wall_method'] = "GWR_FOOTPRINTS"
            else:
                data["wall_method"] = "DSM"

                # Detect flat roof
                if data["slope50"] < 5:
                    data["flat_roof"] = True
                else:
                    data["flat_roof"] = False

                # Estimate ebf from height model
                ebf_floors = calc_ebf3(
                    data, f_height=2.8)

                nr_of_floors = min(len(ebf_floors), data["gastw"])
                data["nr_of_floors"] = nr_of_floors
                if len(ebf_floors) > 1:
                    ebf = sum(ebf_floors[:-1])
                else:
                    ebf = sum(ebf_floors)

                if ebf == 0.0:
                    logging.info(
                        "0 ebf for building {}, {}".format(
                            data['btype'],
                            data['bid']))
                    continue

                data["ebf"] = ebf


            # Detect basement
            if data["gkat"] >= 1020:
                data["has_basement"] = True
            else:
                data["has_basement"] = False

            lat, lon = transformer_lv03_to_wg84.transform(data["x"], data["y"])
            data["lon"] = lon
            data["lat"] = lat

            ww, hh = sol_fwd * (lon, lat)
            hh = int(hh) + 0.5
            ww = int(ww) + 0.5

            # Horizontal radiation -> alpha = 0, beta = 0
            for month in range(12):
                data["G_sH{}".format(month)] = solar_cache[(ww, hh)]["G_sH{}".format(month)]

            # Tilted radiation on walls -> alpha = *, beta = 90
            for a in range(-180, 181, 15):
                for month in range(12):
                    data["G_s{}_{}".format(int(a), month)] = solar_cache[(ww, hh)]["G_s{}_{}".format(int(a), month)]

            d2 = data.copy()

            ts = time.time()

            heat_results = {}
            heat_results['bid'] = data['bid']
            heat_results['btype'] = data['btype']
            heat_results['ebf'] = data["ebf"]
            heat_results['gwr_ratio'] = data['gwr_ratio']
            heat_results['heatdemand'] = defaultdict(list)
            heat_results['heatdemandY'] = []
            heat_results['coolingdemand'] = defaultdict(list)
            heat_results['coolingdemandY'] = []
            heat_results['energydemand'] = defaultdict(list)
            heat_results['energydemandY'] = []
            heat_results['egid'] = data['egid']
            heat_results['x'] = data['x']
            heat_results['y'] = data['y']
            heat_results['wall_method'] = data['wall_method']

            for r in range(number_of_runs):

                data = d2.copy()

                sim = random.choice(rcp_sims[rcp])
                temp_fwd, tdata, temp_nodata = temp_data[(sim, rcp, period)]

                data['climate_scen'] = "{}_{}_{}".format(sim, rcp, period)

                ww, hh = ~temp_fwd * (lon, lat)
                data["temps"] = list(tdata[:, int(hh), int(ww)])
                if temp_nodata in data["temps"]:
                    raise Exception('Temperature should not include nodata: {}: {}'.format(temp_nodata, str(data["temps"])))

                data["run"] = r

                # FIX gbaup = 0
                if d2['gbaup'] == 0:
                    data['gbaup'] = sample_gbaup(gkat=data['gkat'])

                # Fix DSM data not available
                if data['delta10'] < 0.9:
                    data["flat_roof"] = sample_flat_roof(gkat=data['gkat'],
                                                         gbaup=data['gbaup'])

                # Create config file
                conf = SIAConf()

                conf = config_climate(conf, data)

                # Config Weather and such
                conf = config_utilization(conf, data)

                conf = config_misc(conf, data)

                conf = config_special(conf, data)

                # Sampling
                if data["gbaup"] == 8011:  # <1919
                    data["r_win"] = gimme_normal(15.4, 4.6, 5.0, 30.0) / 100.0
                # 1919 - 1970
                elif data["gbaup"] > 8011 and data["gbaup"] <= 8014:
                    data["r_win"] = gimme_normal(13.6, 4.2, 6.0, 25.0) / 100.0
                # 1971 - 1980
                elif data["gbaup"] > 8014 and data["gbaup"] <= 8015:
                    data["r_win"] = gimme_normal(15.2, 4.1, 10.0, 25.0) / 100.0
                # 1981 - 2000
                elif data["gbaup"] > 8015 and data["gbaup"] <= 8019:
                    data["r_win"] = gimme_normal(18.5, 4.1, 10.0, 30.0) / 100.0
                else:
                    data["r_win"] = gimme_normal(16.4, 3.9, 10.0, 28.0) / 100.0

                conf = config_windows(conf, data)
                conf = config_areas(conf, data)
                hts = pyheat.sia380.sia380(conf)

                for i, ht in enumerate(hts):
                    heat_results['heatdemand'][i].append(max(0, ht['Q_h'] * data['gwr_ratio']))
                heat_results['heatdemandY'].append(sum([max(0, ht['Q_h'] * data['gwr_ratio']) for ht in hts]))

                for i, ht in enumerate(hts):
                    heat_results['energydemand'][i].append(ht['Q_h'] * data['gwr_ratio'])
                heat_results['energydemandY'].append(sum([ht['Q_h'] * data['gwr_ratio'] for ht in hts]))

                for i, ht in enumerate(hts):
                    heat_results['coolingdemand'][i].append(min(0, ht['Q_h'] * data['gwr_ratio']))
                heat_results['coolingdemandY'].append(sum([min(0, ht['Q_h'] * data['gwr_ratio']) for ht in hts]))

            logging.info("building {}/{} took {}".format(data['btype'], data['bid'], time.time() - ts))

            outq.put(heat_results)

        except queue.Empty:
            logging.info("{}: worker finished".format(wid))
            finished = True
            break
        except Exception as e:
            logging.exception("{}: {}".format(wid, str(e)))


# ----- HEAT MODEL

def calc_ebf3(data, f_height=2.8):
    """ Estimates EBF and floors based on percentile data
        Returns ebf and number of floors

        data needs to be a dict like object with keys area
        where X is in range(5, 105, 5).
    """
    i = 1
    ebf_tot = 0.0
    rs = []
    ebf_floors = []

    while f_height * i < data["p99"]:
        new_r = 0.0

        for p in range(5, 105, 5):
            d_p = data["p{}".format(p)]
            d_pe = data["pe{}".format(p)]
            d_a = (d_p + d_pe) * 0.5

            if d_a >= f_height * i:
                new_r += 0.05

        if new_r == 0.0 or (new_r < 0.1 and data["footprint_area"] * new_r < 10.0):
            i -= 1
            break

        rs.append(new_r)
        new_ebf = data["footprint_area"] * new_r

        ebf_floors.append(new_ebf)

        ebf_tot += new_ebf
        if f_height * (i + 1) < data["p99"]:
            i += 1
        else:
            break

    return ebf_floors


def gimme_normal(mu, sigma, xmin, xmax):
    assert xmin < xmax
    x = 0
    while x <= xmin or x >= xmax:
        x = np.random.normal(mu, sigma, size=None)
    return x


class GWRGKAT:
    PROV = 1010
    EFH = 1021
    MFH = 1025
    WOHNNEB = 1030
    WOHNTEIL = 1040


def config_utilization(conf, data):
    # Raumtemperatur C
    conf.theta_o_C = np.random.normal(20.0, 1.5)

    # Regelungszuschlag fuer die Raumteperatur K
    # - Ab  baujahr / renovation 2006 (8021) -> 0.0
    # - sonst samplen 1.0 / 2.0
    renp_hydro = get_ren_period(
        hydronicsystemrenewalrates,
        data['gbaup'],
        data["sia_building_type"])
    data["renp_hydro"] = renp_hydro
    if renp_hydro <= 8020:
        conf.delta_theta_o_K = float(np.random.randint(1, 2))
    else:
        conf.delta_theta_o_K = 0.0

    # Personenflaeche m^2 / P
    conf.A_P = SIAPeopleArea[data["sia_building_type"]]

    # Waermeabgabe pro Person W / P
    conf.Q_P = SIAHeatPersonQp[data["sia_building_type"]]

    # Prasenezzeit pro Tag h / d
    conf.t_p = SIAOccupation_tp[data["sia_building_type"]]

    # Elektrizitaetsbedarf pro Jahr MJ/m^2
    # SIAElectricity_Efel[SIABuildingType.MFH] = 100.0
    # SIAElectricity_Efel[SIABuildingType.EFH] = 80.0
    # TODO Werte 2015?? -> Licht
    if data["sia_building_type"] == SIABuildingType.MFH:
        conf.Q_El = np.random.normal(100, 7)
    elif data["sia_building_type"] == SIABuildingType.EFH:
        conf.Q_El = np.random.normal(80, 7)

    # Reduktionsfaktor Elektrizitaet -
    conf.f_El = SIAElectricityRed_fel[data["sia_building_type"]]

    # fleachenbezogener Aussenluft-Volumenstrom m^3 / (h*m^2)
    conf.V_A_E = SIAAussenluftvolumenstrom_VAE[data["sia_building_type"]]
    # Assumption Heeren et al. 2015: m3/m3h, Lognormal, mu = -0,638, sigma =
    # 0,821, Mdn: 0.53, 95% CI (0.11, 2.64). Sources: Murrey et al. 1995,
    # Hellweg et al. 2009

    # http://www.enev-online.info/enev/21_anhang_1.htm
    red_factor = 0.8
    if data["pe99"] is None or data["pe99"] < 2.8 * 3:
        red_factor = 0.76

    """
    Residential Air Exchange Rates in the United States:
    Empirical and Estimated Parametric Distributions by
    Season and Climatic Region
    Donald M. Murray and David E. Burmaster
    """
    #R1
#     V_A_E_S1 = np.random.lognormal(-1.305, 0.799) * data['volume'] * red_factor / data["ebf"]
#     V_A_E_S2 = np.random.lognormal(-1.011, 0.627) * data['volume'] * red_factor / data["ebf"]
#     V_A_E_S3 = np.random.lognormal(-0.441, 0.736) * data['volume'] * red_factor / data["ebf"]
#     V_A_E_S4 = np.random.lognormal(-1.531, 0.559) * data['volume'] * red_factor / data["ebf"]

    # R2
#     V_A_E_S1 = np.random.lognormal(-0.798, 0.673) * data['volume'] * red_factor / data["ebf"]
#     V_A_E_S2 = np.random.lognormal(-1.177, 0.807) * data['volume'] * red_factor / data["ebf"]
#     V_A_E_S3 = np.random.lognormal(-0.588, 0.612) * data['volume'] * red_factor / data["ebf"]
#     V_A_E_S4 = np.random.lognormal(-1.173, 0.540) * data['volume'] * red_factor / data["ebf"]

    R2_ALL = np.random.lognormal(-0.844, 0.698) * data['volume'] * red_factor / data["ebf"]
    V_A_E_S1 = R2_ALL
    V_A_E_S2 = R2_ALL
    V_A_E_S3 = R2_ALL
    V_A_E_S4 = R2_ALL

    conf.V_A_Es = [V_A_E_S1, #jan
                   V_A_E_S1, #feb
                   V_A_E_S2,
                   V_A_E_S2,
                   V_A_E_S2,
                   V_A_E_S3,
                   V_A_E_S3,
                   V_A_E_S3,
                   V_A_E_S4,
                   V_A_E_S4,
                   V_A_E_S4,
                   V_A_E_S1] # dec

#     conf.V_A_E = np.random.lognormal(-0.638, 0.821) * data['volume'] * red_factor / data["ebf"]

    #http://www.geak.ch/Resources/Documents/UploadDocuments/Manual_Sep10_Version_2.0.1_DE.pdf
#     conf.V_A_E = np.random.triangular(left=0.35, mode=0.7, right=1.4)

    return conf


def config_climate(conf, data):

    # Klimadaten
    # Laenge der Berechnungsperiode d
    conf.t_c = days

    # Hoehenlage in Meter ueber Meer m
    conf.h = data["elevation"]

    # Aussentemperatur C
    conf.theta_e = data["temps"]

    # Globale Sonnenstrahlung horizontal MJ / m^2
    conf.G_sH = [data["G_sH{}".format(i)] for i in range(12)]

    return conf


def config_areas(conf, data):
    '''Flachen, Laengen, Anzahl
    Needs to be run after config_windows!
    '''

    # Energiebezugsflache m^2
    conf.A_E = data["ebf"]

    # Dach gegen Aussenluft m^2
    conf.A_Re = data["a_roof"]

    # Decke gegen unbeheizte Raeume m^2
    conf.A_Ru = 0.0

    # Wand gegen Aussenluft m^2
    conf.A_We = sum([max(0.0,
                         (data["a_wall_{}".format(i).replace('.', '_').replace('-', 'm')] -
                          data["a_shared_{}".format(i).replace('.', '_').replace('-', 'm')]))
                     for i in np.arange(-180.0, 180.1, 15.0)]) - data["win_area"]

    # Wand gegen unbeheizte Raueme m^2
    conf.A_Wu = 0.0

    # Wand gegen Erdreich m^2
    conf.A_WG = 0.0

    # Wand gegen benachbarten beheitzten Raum
    conf.A_Wn = sum([max(0.0,
                         min(data["a_wall_{}".format(i).replace('.', '_').replace('-', 'm')],
                             data["a_shared_{}".format(i).replace('.', '_').replace('-', 'm')]))
                     for i in np.arange(-180.0, 180.1, 15.0)])

    # Boden gegen Aussenluft m^2
    conf.A_Fe = 0.0

    # Boden gegen unbeheizte Raume m^2
    if data["has_basement"]:
        conf.A_Fu = data["footprint_area"]
    else:
        conf.A_Fu = 0.0

    # Boden gegen Erdreich mit Bauteilheizung m^2
    if data["has_basement"]:
        conf.A_FG = 0.0
    else:
        conf.A_FG = data["footprint_area"]

    # Waermebruecke Decke/Wand m
    conf.l_RW = 0.0

    # Waermebruecke Gebaudesockel m
    conf.l_WF = 0.0

    # Waermebruecke Balkon m
    conf.l_B = 0.0

    # Waermebruecke Boden / Keller-Innenwand m
    conf.l_F = 0.0

    # Waermebruecke Stuetzen, Traeger, Konsolen
    conf.z = 0.0

    return conf


def config_misc(conf, data):


    # Diverses
    # Dach gegen Aussenluft W / (m^2 * K)

    renp_flat_roof = None
    if data["flat_roof"]:
        renp_flat_roof = get_ren_period(
            flatroofrenewalrates,
            data['gbaup'],
            data["sia_building_type"])
        data["renp_flat_roof"] = renp_flat_roof
        conf.U_Re = flatroofuvalues[
            data["sia_building_type"]][
            data['gbaup']][renp_flat_roof]
        data["renp_tilted_roof"] = None
    else:
        renp_tilted_roof = get_ren_period(
            slantedroofrenewalrates,
            data['gbaup'],
            data["sia_building_type"])
        data["renp_tilted_roof"] = renp_tilted_roof
        conf.U_Re = slantedroofuvalues[
            data["sia_building_type"]][
            data['gbaup']][renp_tilted_roof]

        data["renp_flat_roof"] = None

    # Decke gegen unbeheizte Raeume W / (m^2 * K)
    if renp_flat_roof is None:
        renp_flat_roof = get_ren_period(
            flatroofrenewalrates,
            data['gbaup'],
            data["sia_building_type"])
        data["renp_flat_roof"] = renp_flat_roof
    conf.U_Ru = flatroofuvalues[
        data["sia_building_type"]][
        data['gbaup']][renp_flat_roof]

    # Reduktionsfaktor Decke gegen unbeheizte Raeume -
    '''
    unbeheizter Raum :    b uR , b uW , b uF
    Estrichraum, Schraegdach ungedaemmt: 0.9
    Estrichraum, Schraegdach gedaemmt: U e < 0.4 W/m K: 0.7
    Kellerraum ganz im Erdreich: 0.7
    Kellerraum teilweise oder ganz ueber dem Erdreich: 0.8
    angebauter Raum: 0.8
    Glasvorbau: 0.9
    '''
    conf.b_uR = 0.9

    # Wand gegen Aussenluft W / (m^2 * K)
    renp_walls = get_ren_period(
        wallsrenewalrates,
        data['gbaup'],
        data["sia_building_type"])
    data["renp_walls"] = renp_walls
    conf.U_We = wallsuvalues[
        data["sia_building_type"]][
        data['gbaup']][renp_walls]

    # Wand gegen unbehizte Raume W / (m^2 * K)
    conf.U_Wu = wallsuvalues[
        data["sia_building_type"]][
        data['gbaup']][renp_walls]

    # Reduktionsfaktor Wand gegen unbehizte Raume
    conf.b_uW = 0.8

    # Wand gegen Erdreich W / (m^2 * K)
    conf.U_WG0 = wallsuvalues[
        data["sia_building_type"]][
        data['gbaup']][renp_walls]

    # Reduktionsfaktor Wand gegen Erdreich -
    conf.b_GW = reduktionsfaktor_erdreich_wand(0.0, conf.U_WG0)

    # Wand gegen benachbarten beheizten Raum W / (m^2 * K)
    conf.U_Wh = wallsuvalues[
        data["sia_building_type"]][
        data['gbaup']][renp_walls]

    # Raumteperatur des benachbarten beheizten Raumes C
    conf.theta_on = np.random.normal(20.0, 1.5)

    # Boden gegen Aussenluft W / (m^2 * K)
    renp_floor = get_ren_period(
        floorrenewalrates,
        data['gbaup'],
        data["sia_building_type"])
    data["renp_floor"] = renp_floor
    conf.U_Fe = flooruvalues[
        data["sia_building_type"]][
        data['gbaup']][renp_floor]

    # Boden gegen unbeheizte Raeume W / (m^2 * K)
    conf.U_Fu = flooruvalues[
        data["sia_building_type"]][
        data['gbaup']][renp_floor]

    # Reduktionsfaktor Boden gegen unbeheizte Raume -
    conf.b_uF = 0.7

    # Boden gegen Erdreich mit Bauteilheizung W / (m^2 * K)
    conf.U_FG0 = flooruvalues[
        data["sia_building_type"]][
        data['gbaup']][renp_floor]

    # Reduktionsfaktor Boden gegen Erdreich -
    # TODO Tiefe, 0m?
    conf.b_GF = reduktionsfaktor_erdreich_boden(0.0,
                                                data["footprint_area"],
                                                data["perimeter_length"],
                                                conf.U_FG0)

    # Temperaturzuschlag Bauteilheizung K
    conf.delta_theta = 0.0

    # Waermebruecke Stuetzen, Traeger, Konsolen W/ K
    conf.chi = 0.0

    # Waermebruecke Decke/Wand W / (m * K)
    conf.psi_RW = 0.0

    # Waermebruecke Gebaeudesockel W / (m * K)
    conf.psi_WG = 0.0

    # Waermebruecke Balkon W / (m * K)
    conf.psi_B = 0.0

    # Waermebruecke Boden / Keller - Innenwand W / (m * K)
    conf.psi_F = 0.0

    return conf


def config_windows(conf, data):

    # Fenster

    # Abminderungsfaktor fuer Fensterrahmen -
    # TODO http://www.energie-zentralschweiz.ch/pdf/Merkblatt_Fenster_de.pdf
    conf.F_F = 0.7

    # Verschattungsfaktor horizontal -
    conf.F_SH = 0.8

    # Fenster horizontal m^2
    conf.A_wH = 0.0

    def a_deg(deg):
        return max(
            0.0, data["a_wall_{}".format(deg)] - data["a_shared_{}".format(deg)])

    # Fenster horizontal W / (m^2 * K)
    renp_windows = get_ren_period(
        windowsrenewalrates,
        data['gbaup'],
        data["sia_building_type"])

    conf.U_wH = windowsuvalues[
        data["sia_building_type"]][
        data['gbaup']][renp_windows]

    # Gesamtenergiedurchlassgrad Fenster (senkrecht)
    conf.g_90deg = windowsgvalues[
        data["sia_building_type"]][
        data['gbaup']][renp_windows]

    F_S2 = 0.9 + np.random.rand() / 10.0
    F_S3 = 1.0

    for a in np.arange(-180.0, 181.0, 15.0):
        degk = str(a).format(a).replace('.', '_').replace('-', 'm')
        A = a_deg(degk) * data["r_win"]

        if A > 0:
            #ts3 = time.time()
            F = get_verschattungsfaktor_horizont(data["hor_{}".format(degk)],
                                                 a) * F_S2 * F_S3
            #print("F: ", round(time.time()-#ts3,6))
            #ts3 = time.time()
    #         #print(data["lat"], data["lon"], a)
#             G = read_solar(data["lon"], data["lat"], a, 90.0)
            G = [data["G_s{}_{}".format(int(a), i)] for i in range(12)]
            #print("G: ", round(time.time()-#ts3,6))
            #ts3 = time.time()

            conf.add_wall(a=degk,
                          F=F,
                          G=G,
                          A=A,
                          U=windowsuvalues[data["sia_building_type"]][data['gbaup']][renp_windows])
            #print("A: ", round(time.time()-#ts3,6))
            #ts3 = time.time()

    # Waermebruecke Fensteranschlag m
    data["win_area"] = sum(conf.As)

    # https://www.gr.ch/DE/institutionen/verwaltung/bvfd/aev/dokumentation/EnergieeffizienzVollzugsformulare/d2_1-waermeschutz.pdf
    # Um die Berechnung der Laenge der Waermebruecken der Fenster zu
    # vereinfachen, ist es zulaessig, eine Waermebrueckenlaenge von 3 m pro
    # Quadratmeter Fensterflaeche einzusetzen. (siehe SIA 380/1, 3.5.3.4)
    conf.l_w = data["win_area"] * 3.0

    # Waermebruecke Fensteranschlag W / (m * K)
    conf.psi_W = 0.10  # 0.10
    return conf


def config_special(conf, data):

    # Spezielle Engabedaten
    #     waermespeicherfaehigkeitstyp = WaermeSpeicherfaehigkeitType.Mittel
    #
    #     # 3-eck verteilung 0.1, 0.4, 0,5
    #
    #     conf.C_AE = SIAWaermespeicherfaehigkeit[waermespeicherfaehigkeitstyp]
    conf.C_AE = np.random.triangular(0.1, 0.4, 0.5, size=None)
    # Waermespeichergaehigkeit pro Energiebezugsflaeche MJ / (m^2 * K)

    conf.a_0 = 1.0
    # numerischer Parameter fuer Ausnutzungsgrad -

    conf.tau_0 = 15.0
    return conf


def gkat2siabuildingtype(data):
    if data["gkat"] == GWRGKAT.EFH:
        return SIABuildingType.EFH
    elif data["gkat"] == GWRGKAT.MFH:
        return SIABuildingType.MFH
    elif data["gkat"] == GWRGKAT.WOHNNEB:
        return SIABuildingType.MFH
    #print("no sia kat found", data["gkat"])
    return SIABuildingType.MFH

# ---- HEAT model


def writer(q, bfsnr, outdir):
    finished = False
    i = 0

    killret = 0

    outpath = os.path.join(outdir, "{}.txt.gz".format(bfsnr))

    fields = ['btype', 'bid', 'bfsnr', 'ebf', 'egid', 'x', 'y', 'wall_method', 'gwr_ratio']
    for m in range(12):
        for p in range(0, 101, 5):
            fields.append('heatdemand_{}_{}'.format(m, p))
    for p in range(0, 101, 5):
        fields.append('heatdemandY_{}'.format(p))

    for m in range(12):
        for p in range(0, 101, 5):
            fields.append('coolingdemand_{}_{}'.format(m, p))
    for p in range(0, 101, 5):
        fields.append('coolingdemandY_{}'.format(p))

    for m in range(12):
        for p in range(0, 101, 5):
            fields.append('energydemand_{}_{}'.format(m, p))
    for p in range(0, 101, 5):
        fields.append('energydemandY_{}'.format(p))

    logging.info("dbworker: open {}".format(outpath))
    with gzip.open(outpath, 'wb') as ff:
        while not finished:
            try:
                i += 1
                if (i % 100) == 0:
                    logging.info("writer: wrote {}".format(i))
                    ff.flush()

                res = q.get()

                if res == "killitwithfire_thedb":
                    killret += 1
                    logging.info("writer: recieved kill of worker, now: {}".format(killret))

                    if killret == cpus:
                        logging.info("writer: all done")
                        finished = True
                        break
                    continue

                res["bfsnr"] = bfsnr

                for m in range(12):
                    for p in range(0, 101, 5):
                        res['heatdemand_{}_{}'.format(m, p)] = np.percentile(res['heatdemand'][m], p)

                for p in range(0, 101, 5):
                    res['heatdemandY_{}'.format(p)] = np.percentile(res['heatdemandY'], p)

                for m in range(12):
                    for p in range(0, 101, 5):
                        res['coolingdemand_{}_{}'.format(m, p)] = np.percentile(res['coolingdemand'][m], p)

                for p in range(0, 101, 5):
                    res['coolingdemandY_{}'.format(p)] = np.percentile(res['coolingdemandY'], p)

                for m in range(12):
                    for p in range(0, 101, 5):
                        res['energydemand_{}_{}'.format(m, p)] = np.percentile(res['energydemand'][m], p)

                for p in range(0, 101, 5):
                    res['energydemandY_{}'.format(p)] = np.percentile(res['energydemandY'], p)

                # Change here if res is not a dictionary with the fields of "fields" list from above
                line = ",".join([str(res[f]) for f in fields]) + "\n"
                line = line.replace('--', 'None')
                line = line.replace("'NaN'", 'None')

                ff.write(line.encode())
                ff.flush()

            except queue.Empty:
                logging.info("writer: writer finished")
                finished = True
                ff.flush()
                break
            except Exception as e:
                logging.exception("writer: {}".format(str(e)))
                try:
                    ff.flush()
                    logging.error(line)
                except:
                    pass


def generate_sql(bfsnr):
    sql = """
SELECT L.*,
       gws.wareas
FROM   (SELECT b.x,
               b.y,
               w.a_roof,
               a_wall_m180_0,
               a_wall_m165_0,
               a_wall_m150_0,
               a_wall_m135_0,
               a_wall_m120_0,
               a_wall_m105_0,
               a_wall_m90_0,
               a_wall_m75_0,
               a_wall_m60_0,
               a_wall_m45_0,
               a_wall_m30_0,
               a_wall_m15_0,
               a_wall_0_0,
               a_wall_15_0,
               a_wall_30_0,
               a_wall_45_0,
               a_wall_60_0,
               a_wall_75_0,
               a_wall_90_0,
               a_wall_105_0,
               a_wall_120_0,
               a_wall_135_0,
               a_wall_150_0,
               a_wall_165_0,
               a_wall_180_0,
               a_shared_m180_0,
               a_shared_m165_0,
               a_shared_m150_0,
               a_shared_m135_0,
               a_shared_m120_0,
               a_shared_m105_0,
               a_shared_m90_0,
               a_shared_m75_0,
               a_shared_m60_0,
               a_shared_m45_0,
               a_shared_m30_0,
               a_shared_m15_0,
               a_shared_0_0,
               a_shared_15_0,
               a_shared_30_0,
               a_shared_45_0,
               a_shared_60_0,
               a_shared_75_0,
               a_shared_90_0,
               a_shared_105_0,
               a_shared_120_0,
               a_shared_135_0,
               a_shared_150_0,
               a_shared_165_0,
               a_shared_180_0,
               g.gbaup,
               g.gkat,
               g.gastw,
               g.garea,
               g.egid,
               b.bid,
               b.btype,
               b.volume,
               b.p0,
               b.p5,
               b.p10,
               b.p15,
               b.p20,
               b.p25,
               b.p30,
               b.p35,
               b.p40,
               b.p45,
               b.p50,
               b.p55,
               b.p60,
               b.p65,
               b.p70,
               b.p75,
               b.p80,
               b.p85,
               b.p90,
               b.p95,
               b.p99,
               b.p100,
               b.pe0,
               b.pe5,
               b.pe10,
               b.pe15,
               b.pe20,
               b.pe25,
               b.pe30,
               b.pe35,
               b.pe40,
               b.pe45,
               b.pe50,
               b.pe55,
               b.pe60,
               b.pe65,
               b.pe70,
               b.pe75,
               b.pe80,
               b.pe85,
               b.pe90,
               b.pe95,
               b.pe99,
               b.pe100,
               b.area      AS footprint_area,
               b.slope50,
               b.perimeter AS perimeter_length,
               b.elevation,
               b.delta10,
               b.mar_ratio,
               b.mar_angle,
               hor_m180_0,
               hor_m165_0,
               hor_m150_0,
               hor_m135_0,
               hor_m120_0,
               hor_m105_0,
               hor_m90_0,
               hor_m75_0,
               hor_m60_0,
               hor_m45_0,
               hor_m30_0,
               hor_m15_0,
               hor_0_0,
               hor_15_0,
               hor_30_0,
               hor_45_0,
               hor_60_0,
               hor_75_0,
               hor_90_0,
               hor_105_0,
               hor_120_0,
               hor_135_0,
               hor_150_0,
               hor_165_0,
               hor_180_0,
               r.ratio as gwr_ratio
        FROM   heat.walls AS w,
               heat.buildinginfo AS b,
               heat.horizons AS h,
               heat.gwrmatches2 AS gm,
               heat.gwr AS g,
               heat.mview_warea_ratio as r
        WHERE  w.bid = b.bid
               AND b.bid = h.bid
               AND w.btype = b.btype
               AND b.btype = h.btype
               AND g.gwrid = gm.gwrid
               AND gm.bid = b.bid
               AND b.bfsnr = BFSNR
               AND gm.btype = b.btype
               AND r.egid = g.egid
               AND g.gkat > 1020 and g.gkat < 1060) L
       LEFT JOIN heat.view_bld_gws_wareas AS gws using (btype, bid)
    """
    sql = sql.replace("BFSNR", str(bfsnr))

    return sql


if __name__ == '__main__':

    dir_name = "period_{}_rcp_{}_numberofruns_{}".format(sim_period, sim_rcp, number_of_runs)
    outdir = os.path.join(out_base_dir, dir_name)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    with fiona.open(mun_path) as src:
        for f in src:
            bfsnr = f['properties']['BFS_NUMMER']
            outpath = os.path.join(outdir, "{}.txt.gz".format(bfsnr))

            # We check if municpality already exists and skip it if already simulated
            if not os.path.exists(outpath):
                process_mun(bfsnr, sim_period, sim_rcp, outdir)
            else:
                logging.info("skip {} / {}".format(bfsnr, outpath))
