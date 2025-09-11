"""

This script simulates a Solar-Wind-BESS DVPP with production patterns to participate in FFR and FCR services

The reward mechanisms are evaluated, namely Shapely Value and Nash Bargaining Solution

Setup:
- Finnish Grid
- Services: FFR, FCR-D up and both
- Device and Size:
    - 1 MW Solar PV (LPF)
    - 1 MW Wind Turbine (LPF)
    - 1 MW / 1 MWh Battery (HPF)
- DC gains:
    - Solar PV: time varying based on avg. production
    - Wind Turbine: time varying based on avg. production
    - Battery: 1 until energy limit reached, then 0
- Time Frame: 1 Week

"""

import control as ct
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

from check_qualification import sympy_to_tf, create_curve
from get_max_reward import simulate_devices_and_limits

# import procduction data
from src.time_varying_dc_gain import get_wind_solar_dc_gains
from src.get_required_services import *
from src.game_theory_helpers import get_shapley_values

wind_solar_dc_gains, probs_15_min = get_wind_solar_dc_gains()  # time series of 15-minute dc gain

power_ratings_dict = {  # in MVA
    # 'Hydro': 1,
    'BESS': 1,
    # 'SC': .2
    'PV': 1,
    'Wind': 1
}


all_shapely_values = []
all_values = []
my_path = 'pics/varying'
SERVICE = 'FCR'
service_diff = 5  # difference of service compared to capacity: if service_diff = 2, then service rating is half of capacity

my_names = list(power_ratings_dict.keys())

# set colors
cmap = plt.colormaps['tab20']
color_dict = {k: (cmap((i)*2), cmap((i)*2+1)) for i, k in enumerate(my_names)}

pi_params = {}

tau_PV, K_PV = 1.5, 1
tau_WTG, K_WTG = 2, 1
tau_BESS, K_BESS = 0.1, 1
tau_SC = 0.01

pi_params['PV'] = {"kp": 11.9, "ki": 157.9}
pi_params['Wind'] = {"kp": 11.9, "ki": 118}
pi_params['BESS'] = {"kp": 12, "ki": 2370}

# now select Sx scenarios where we simulate wind solar and battery
Sx = 5  # number of scenarios
# selected_indices = np.random.randint(0, high=len(wind_solar_dc_gains), size=Sx)
ps = probs_15_min.iloc[:, 0].values if SERVICE=='FFR' else probs_15_min.iloc[:, 1].values
selected_indices = np.random.choice(len(wind_solar_dc_gains), size=Sx, p=ps)

# run scenario simulations
for i, idx in enumerate(selected_indices):
    G_PV = ct.tf([K_PV], [tau_PV, 1], inputs=['u'], outputs=['y'])
    G_WTG = ct.tf([K_WTG], [tau_WTG, 1], inputs=['u'], outputs=['y'])
    G_BESS = ct.tf([K_BESS], [tau_BESS, 1], inputs=['u'], outputs=['y'])

    # todo: integrate restricted battery model

    # now get restricted power rating due to time varying dc gain
    dc_gain_Wind = wind_solar_dc_gains['Wind'].iloc[idx] * power_ratings_dict['Wind']
    dc_gain_PV = wind_solar_dc_gains['Solar'].iloc[idx] * power_ratings_dict['PV']

    # main dict: name -> (system, device_type, rating)
    IO_dict = {('Wind'): (G_WTG, 'lpf', dc_gain_Wind),
            ('PV'): (G_PV, 'lpf', dc_gain_PV),
            ('BESS'): (G_BESS, 'hpf', power_ratings_dict['BESS']),
            }

    # PI and saturation controller parameters
    for name in IO_dict.keys():
        pi_params[name]['saturation_limits'] = (-power_ratings_dict[name], power_ratings_dict[name])

    if SERVICE == 'FCR':
        T_MAX = 60
        ts, input, requirement_curve = get_fcr()
    elif SERVICE == 'FFR':
        T_MAX = 10
        ts, input, requirement_curve = get_ffr()
    elif SERVICE == 'FFR-FCR':
        T_MAX = 60
        ts, input, requirement_curve = get_ffr_fcr()

    # service response
    VALUE, ENERGY, PEAK_POWER, SHAPELY_VALS = simulate_devices_and_limits(
                                IO_dict=IO_dict,
                                pi_params=pi_params,
                                input_service_max=input,
                                curve_service_min=requirement_curve,
                                title=f'{SERVICE} Response of',
                                service_diff=service_diff,
                                T_MAX=T_MAX,
                                save_path=my_path,
                                STATIC_PF=False
    )

    all_values.append(VALUE)
    all_shapely_values.append(SHAPELY_VALS)

# create final shapely value by averaging over scenarios
final_vals = {k: 0 for k in all_values[0].keys()}   # dict to compute final shapely value
for vals in all_values:
    for k, v in vals.items():
        final_vals[k] += v

final_vals = {k: v / len(all_values) for k, v in final_vals.items()}

final_shapely = get_shapley_values(final_vals, my_names)
print('Final Shapely Values for', SERVICE, final_shapely)

pd.DataFrame(all_shapely_values).to_csv(f'{my_path}/shapely_values_{SERVICE}.csv')
