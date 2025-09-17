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


from dvpp_static_v1 import run_dvpp_simulation


if __name__ == '__main__':
    # run simulation with default parameters
    def get_io_dict():
        return {'Hydro': (get_hydro_tf(), 'lpf', 1),
                'BESS': (get_bess_io_sys(t_drop=20), 'bpf', 1),
                'SC': (get_sc_io_sys(t_drop=5), 'hpf', 1),
                }
                
    run_dvpp_simulation(get_io_dict,
                        save_path='pics/v3',
                        )