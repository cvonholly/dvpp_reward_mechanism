"""
DVPP consisting of Solar PV, Wind and Battery 
    non-stochastic PV=Wind generation
"""

# import procduction data
from src.get_device_systems import get_pv_sys, get_wind_sys, get_bess_io_sys, get_bess_energy_sys

from run_dvpp_simulation import run_dvpp_simulation
from src.get_required_services import *


if __name__ == '__main__':
    # run simulation with default parameters
    def get_io_dict():
        return {'PV': (get_pv_sys(), 'lpf', 1),
                'Wind': (get_wind_sys(), 'lpf', 1),
                'BESS': (get_bess_energy_sys(e_max=3), 'hpf', 1)
                }
                
    run_dvpp_simulation(get_io_dict,
                        # services_input={'FFR': get_ffr()},
                        save_path='pics/v1',
                        STATIC_PF=False
                        )