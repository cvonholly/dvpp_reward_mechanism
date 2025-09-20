"""
DVPP consisting of Hydro, BESS and Supercapacitor
"""

# import procduction data
from src.get_device_systems import get_hydro_tf, get_bess_io_sys, get_sc_io_sys, get_bess_energy_sys

from run_dvpp_simulation import run_dvpp_simulation


if __name__ == '__main__':
    # run simulation with default parameters
    def get_io_dict():
        return {'Hydro': (get_hydro_tf(), 'lpf', 1),
                'BESS': (get_bess_energy_sys(e_max=1.8), 'hpf', 1),
                'SC': (get_sc_io_sys(t_drop=5), 'hpf', 1),
                }
                
    run_dvpp_simulation(get_io_dict,
                        save_path='pics/v3',
                        STATIC_PF=True
                        )