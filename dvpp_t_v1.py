"""
DVPP consisting of Solar PV, Wind and Battery 
    non-stochastic PV=Wind generation
"""

# import procduction data
from src.get_device_systems import get_pv_sys, get_wind_sys, get_bess_io_sys

from run_dvpp_simulation import run_dvpp_simulation


if __name__ == '__main__':
    # run simulation with default parameters
    def get_io_dict():
        return {'PV': (get_pv_sys(), 'lpf', 1),
                'Wind': (get_wind_sys(), 'lpf', 1),
                'BESS': (get_bess_io_sys(t_drop=8), 'hpf', 1),
                }
                
    run_dvpp_simulation(get_io_dict,
                        save_path='pics/v1',
                        STATIC_PF=False
                        )