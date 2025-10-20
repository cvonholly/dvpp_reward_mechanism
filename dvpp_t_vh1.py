"""
DVPP consisting of Hydro, BESS and Supercapacitor

    Battery: is very big such that the services can be provided by the battery alone
    SC: quickly discharges
    Hydro: slow response

"""
import numpy as np

# import procduction data
from src.get_device_systems import get_hydro_tf, get_bess_io_sys, get_bess_energy_sys, get_sc_time_sys

from run_dvpp_simulation import run_dvpp_simulation


if __name__ == '__main__':
    # run simulation with default parameters
    t_drop, drop_exp = 7, 1
    def get_io_dict():
        return {'Hydro': (get_hydro_tf(), 'lpf', 1.5),
                'BESS': (get_bess_energy_sys(e_max=4), 'bpf', 1),
                'SC': (get_sc_time_sys(t_drop=t_drop, drop_exp=drop_exp), 'hpf', .2),
                }
    
    # 1.56
    # def adaptive_SC_func(t):
    #     t_drop, drop_exp = 3, 1
    #     return np.power(np.where(t > t_drop+1, t - t_drop, 1), -drop_exp)
    #     # return 1 / (np.max(1, t-t_drop)**drop_exp)

    # def adaptive_hydro_func(t):
    #     return 1   
    # change_roles_for_services = {
    #     'FFR': {'BESS': 'hpf', 'SC': 'lpf'},
    # }

    # manually set reference service value
    # set_service_rating = {
    #     'FCR': 1.5,   
    #     'FFR': 1,    
    #     'FFR-FCR': 1.5,
    #     'FCR-D': 1.5,
    # }
                
    run_dvpp_simulation(get_io_dict,
                        save_path='pics/vh1',
                        STATIC_PF=False
                        )