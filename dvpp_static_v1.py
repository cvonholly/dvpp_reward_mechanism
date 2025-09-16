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
    - Solar PV: STATIC
    - Wind Turbine: STATIC
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
from itertools import chain, combinations

from src.get_device_systems import get_bess_io_sys
from get_max_reward import simulate_devices_and_limits
from src.get_controllers import get_pi_params
from src.get_device_systems import get_time_constants, get_pv_sys, get_wind_sys

# import procduction data
from src.time_varying_dc_gain import get_wind_solar_dc_gains
from src.get_required_services import *
from src.game_theory_helpers import get_shapely_value, get_nash_bargaining_solution

def powerset(iterable):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def run_dvpp_simulation(create_io_dict,
                        save_path='pics/new',
                        services_input={'FCR': get_fcr(), 'FFR': get_ffr(), 'FFR-FCR': get_ffr_fcr(), 'FCR-D': get_fcr_d()},
                        service_diff = 0.1,  # minimum fraction of capacity that can be provided as service (1MW)
                        Sx=1,   # number of scenarios to average over
                        make_PV_Wind_stochastic=False,   # take dc gain from probability distribution
                        ):
    
    wind_solar_dc_gains, probs_15_min, prices = get_wind_solar_dc_gains()  # time series of 15-minute dc gain

    # get all orders of coalitions
    # all_shapely_values = pd.DataFrame(pd.NA, columns=list(powerset(create_io_dict().keys())), index=list(services_input.keys()))
    # all_values = pd.DataFrame(pd.NA, columns=list(powerset(create_io_dict().keys())), index=list(services_input.keys()))
    all_values, all_shapely_values = {}, {}
    power_ratings_dict = {name: vals[2] for name, vals in create_io_dict().items()}
    my_names = list(power_ratings_dict.keys())

    pi_params = get_pi_params()   # get all pi parameters
    
    time_constants = get_time_constants()  # get dict with time constants

    Gs_diff = {k: ct.tf([1], [time_constants[k], 1]) for k in my_names}   # get 1st order transfer functions / reference

    # set scenarios
    if Sx > 1:
        ps = probs_15_min.iloc[:, 0].values if service=='FFR' else probs_15_min.iloc[:, 1].values
        selected_indices = np.random.choice(len(wind_solar_dc_gains), size=Sx, p=ps)
    else:
        # set price to average non-zero price
        price = prices[prices>0].mean()
        prices = pd.DataFrame([price], columns=prices.columns, index=[0])
        selected_indices = [0]

    for i, idx in enumerate(selected_indices):
        for service, (ts, input, requirement_curve) in services_input.items():
            print(f'========================\n Simulating {service} for scenario {i+1}/{Sx}, index {idx} \n')
            IO_dict = create_io_dict()  # reset io dict for each service
            if make_PV_Wind_stochastic:   # adjust dc gain
                dc_gain_Wind = wind_solar_dc_gains['Wind'].iloc[idx] * power_ratings_dict['Wind']
                dc_gain_PV = wind_solar_dc_gains['Solar'].iloc[idx] * power_ratings_dict['PV']
                # adjust current io_dict
                IO_dict['Wind'] = (IO_dict['Wind'][0], IO_dict['Wind'][1], dc_gain_Wind)
                IO_dict['PV'] = (IO_dict['PV'][0], IO_dict['PV'][1], dc_gain_PV)
            price = prices.iloc[idx, 0] if service=='FFR' else prices.iloc[idx, 1]
            my_path = save_path + '/' + service.replace('-', '_')

            # service response
            VALUE, ENERGY, PEAK_POWER, SHAPELY_VALS = simulate_devices_and_limits(
                                        IO_dict=IO_dict,
                                        pi_params=pi_params,
                                        input_service_max=input,
                                        curve_service_min=requirement_curve,
                                        title=f'{service} Response of',
                                        service_diff=service_diff,
                                        T_MAX=ts[-1],
                                        save_path=my_path,
                                        STATIC_PF=False,
                                        save_data=False,
                                        x_scenario=i+1,
                                        price=price,  # specify scenario if needed from 1...Sx,
                                        Gs_diff=Gs_diff
            )

            # all_values.loc[service, :] = VALUE
            # all_shapely_values.loc[service, :] = SHAPELY_VALS

            all_values[service] = VALUE
            all_shapely_values[service] = SHAPELY_VALS

    # create final shapely value by averaging over scenarios
    indexes = list(services_input.keys())
    if Sx > 1:
        # make multiindex
        indexes = pd.MultiIndex.from_product([list(range(0, Sx)), list(services_input.keys())], names=['Scenario', 'Service'])
        
    # save values and shapely values
    df = pd.DataFrame.from_dict(all_values, orient='index')
    df.index = indexes
    df.to_csv(f'{save_path}/values.csv', float_format='%.4f')
    df = pd.DataFrame.from_dict(all_shapely_values, orient='index')
    df.index = indexes
    df.to_csv(f'{save_path}/shapely.csv', float_format='%.4f')

if __name__ == '__main__':
    # run simulation with default parameters
    def get_io_dict():
        return {'PV': (get_pv_sys(), 'lpf', 1),
                'Wind': (get_wind_sys(), 'lpf', 1),
                'BESS': (get_bess_io_sys(t_drop=20), 'hpf', 1)}
                
    run_dvpp_simulation(get_io_dict)
