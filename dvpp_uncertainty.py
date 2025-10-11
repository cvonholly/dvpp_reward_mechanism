"""
DVPP consisting of Solar PV, Wind and Battery 
    with uncertainty in Pv and Wind generation
"""

# import procduction data
from src.get_device_systems import get_pv_sys, get_wind_sys, get_bess_io_sys, get_bess_energy_sys

from run_dvpp_simulation import run_dvpp_simulation


if __name__ == '__main__':
    # run simulation with default parameters
    def get_io_dict():
        return {'PV': (get_pv_sys(), 'lpf', 1),
                'Wind': (get_wind_sys(), 'lpf', 1),
                'BESS': (get_bess_energy_sys(e_max=10), 'hpf', 1),
                }
    
    # run scenarios with uncertainties
    # 1. stage: coalition forming based on forecasted generation
    # 2. stage: real-time operation based on actual generation

    def save_pics(i):
        return i < 2  # save pics for first 2 scenarios only
                
    run_dvpp_simulation(get_io_dict,
                        save_path='pics/vu2',
                        STATIC_PF=True,
                        make_PV_Wind_stochastic=True,
                        Sx=3,
                        save_pics=save_pics,
                        calc_1st_stage_reward=True
                        )