"""

This script simulates a Solar-Wind-BESS DVPP with production patterns to participate in FFR and FCR services

The reward mechanisms are evaluated, namely Shapely Value and Nash Bargaining Solution

Setup:
- Finnish Grid
- Services: FFR, FCR, FCR-D up and combination
- Device and Size:
    - Solar PV (LPF)
    - Wind Turbine (LPF)
    - Battery (HPF/BPF)
    - Hydro (LPF)
    - Supercapacitor (HPF)
- DC gains:
    - Solar PV: time varying based on avg. production
    - Wind Turbine: time varying based on avg. production
    - Battery: 1 until energy limit reached, then decaying fast
    - Hydro: constant
    - Supercapacitor: 1 until energy limit reached, then decaying fast
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
from src.game_theory_helpers import get_shapely_value, get_nash_bargaining_solution
from src.get_device_systems import get_hydro_tf, get_bess_io_sys, get_sc_io_sys

wind_solar_dc_gains, probs_15_min, prices = get_wind_solar_dc_gains()  # time series of 15-minute dc gain

power_ratings_dict = {  # in MVA
    'Hydro': 2,
    'BESS': 1,    # in 2:3 to solar PV: 8cent / kWh
    'SC': 1,
    # 'PV': 3,      # 5cent / kWh
    'Wind': 1.5   # 7cent / kWh
                  # s
}


all_shapely_values = []
all_values = []
SERVICE = 'FCR'  # options: 'FCR', 'FFR', 'FFR-FCR', 'FCR-D'
my_path = 'pics/v3/FCR' if SERVICE=='FCR' else 'pics/v3/FFR' if SERVICE=='FFR' else 'pics/v3/FFR_FCR' if SERVICE=='FFR-FCR' else 'pics/v3/FCR_D'
service_diff = 0.1  # minimum fraction of capacity that can be provided as service (1MW)

my_names = list(power_ratings_dict.keys())
pi_params = {}
tau_BESS = 0.1
tau_SC = 0.01
tau_WTG, K_WTG = 2, 1
# K_BESS = 1.5 ?

# now select Sx scenarios where we simulate wind solar and battery
Sx = 1  # number of scenarios
selected_indices = np.random.randint(0, high=len(wind_solar_dc_gains), size=Sx)
ps = probs_15_min.iloc[:, 0].values if SERVICE=='FFR' else probs_15_min.iloc[:, 1].values

# get control systems
G_Hydro, pi_params_hydro, hydro_t_constant = get_hydro_tf()
# todo: corret me
G_BESS = get_bess_io_sys(tau_BESS=tau_BESS, t_drop=30)
G_SC = get_sc_io_sys(tau_SC=tau_SC)
G_WTG = ct.tf([K_WTG], [tau_WTG, 1], inputs=['u'], outputs=['y'])
# G_BESS = ct.tf([1], [tau_BESS, 1])
# G_SC = ct.tf([1], [tau_SC, 1])

pi_params['Hydro'] = pi_params_hydro
pi_params['BESS'] = {"kp": 12, "ki": 2370}
pi_params['SC'] = {"kp": 12, "ki": 2370}
pi_params['Wind'] = {"kp": 11.9, "ki": 118}

# define 1st order transfer function for ADPFs
Gs_diff = {
    'Wind': ct.tf([K_WTG], [tau_WTG, 1]),
    'Hydro': ct.tf([1], [hydro_t_constant, 1]),
    'BESS': ct.tf([1], [tau_BESS, 1]),
    'SC': ct.tf([1], [tau_SC, 1]),
}

# set price to average non-zero price
price = prices[prices>0].mean()
price = price.values[0] if SERVICE=='FFR' else price.values[1]

# run scenario simulations
for i, idx in enumerate(selected_indices):
    # now get restricted power rating due to time varying dc gain
    dc_gain_Wind = wind_solar_dc_gains['Wind'].iloc[idx] * power_ratings_dict['Wind']

    # main dict: name -> (system, device_type, rating)
    IO_dict = { ('Hydro'): (G_Hydro, 'lpf', power_ratings_dict['Hydro']),
                ('Wind'): (G_WTG, 'lpf', dc_gain_Wind),
                ('BESS'): (G_BESS, 'bpf', power_ratings_dict['BESS']),
                ('SC'): (G_SC, 'hpf', power_ratings_dict['SC']),
            }
    
    my_names = list(IO_dict.keys())

    # PI and saturation controller parameters
    for name, vals in IO_dict.items():
        pi_params[name]['saturation_limits'] = (-vals[2], vals[2])  # -rating, +rating

    if SERVICE == 'FCR':
        T_MAX = 60
        ts, input, requirement_curve = get_fcr()
    elif SERVICE == 'FFR':
        T_MAX = 10
        ts, input, requirement_curve = get_ffr()
    elif SERVICE == 'FFR-FCR':
        T_MAX = 60
        ts, input, requirement_curve = get_ffr_fcr()
    elif SERVICE == 'FCR-D':
        T_MAX = 60
        ts, input, requirement_curve = get_fcr_d()

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
                                STATIC_PF=False,
                                save_data=False,
                                x_scenario=i+1,
                                Gs_diff=Gs_diff,
                                price=price  # specify scenario if needed from 1...Sx,
    )

    all_values.append(VALUE)
    all_shapely_values.append(SHAPELY_VALS)

# create final shapely value by averaging over scenarios
final_vals = {k: 0 for k in all_values[0].keys()}   # dict to compute final shapely value
for vals in all_values:
    for k, v in vals.items():
        final_vals[k] += v

# normalize to expected reward value
final_vals = {k: v / len(all_values) for k, v in final_vals.items()}

final_shapely = get_shapely_value(final_vals, my_names)
print('Final Shapely Values for', SERVICE, final_shapely)

# also compute Nash Bargaining Solution
final_nash = get_nash_bargaining_solution(final_vals, my_names)
print('Final Nash Bargaining Solution for', SERVICE, final_nash)

pd.DataFrame(all_shapely_values).to_csv(f'{my_path}/shapely_values_{SERVICE}.csv')
