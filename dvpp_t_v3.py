"""
DVPP consisting of Solar PV, Wind and Battery 
    with stochastic PV=Wind generation
    and Adaptive Dynamic Participation Factor (ADPF)

time horizon: 1 week
"""

import numpy as np

# import procduction data
from src.get_required_services import get_fcr, get_ffr
from src.get_device_systems import get_pv_sys, get_wind_sys, get_bess_io_sys, get_bess_energy_sys

from run_dvpp_simulation import run_dvpp_simulation


if __name__ == '__main__':
    # run simulation with default parameters
    def get_io_dict():
        return {'PV': (get_pv_sys(), 'lpf', 1),
                'Wind': (get_wind_sys(), 'lpf', 1),
                'BESS': (get_bess_energy_sys(e_max=4), 'hpf', 1),
                }
    
    def adaptive_pv_func(t):
        return np.where(t<5, 1, np.exp(-.1*(t-5)))
        # return 1 + .5 * np.sin(t / 5)
    
    def adaptive_wind_func(t):
        return 1
        # return 1 + .5 * np.sin(t / 5 + .1)

    pfs_to_run = [True , False, 'ADPF']

    set_service_rating = {
        'FCR': 1.8,   # 1 MW
        'FFR': 2.0,   # 1 MW
        'FFR-FCR': 1.8,   # 1.5 MW
        'FCR-D': 1.8,   # 1 MW
    }
    
    for pf in pfs_to_run:
        run_dvpp_simulation(get_io_dict,
                            save_path='pics/v3/PV_drop_to_zero',
                            STATIC_PF=pf,   # options: False (DPF), True (SPF), 'ADPF' (ADPF)
                            make_PV_Wind_stochastic=False,
                            Sx=1,
                            save_pics=True,
                            adaptive_func={'PV': adaptive_pv_func, 'Wind': adaptive_wind_func},
                            set_service_rating=set_service_rating
                            )