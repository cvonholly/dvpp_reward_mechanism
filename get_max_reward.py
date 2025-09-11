# Import the packages needed for the examples included in this notebook
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import control as ct
import pandas as pd
import itertools

# from check_qualification import *
from src.dvpp_helper import DVPP_2_devices, DVPP_3_devices, DVPP_4_devices, simualte_DVPP_2_devices
from plot_max_reward import plot_reward_value
from src.get_controllers import get_pi_controller
from src.game_theory_helpers import get_shapely_value
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)

"""

in this script the reward is calculated for the selected provision provision,
trying to maximize the reward for the DVPP coalition

PV, Wind, BESS parameters from https://www.researchgate.net/publication/353419981_Automatic_Load_Frequency_Control_in_an_Isolated_Micro-grid_with_Superconducting_Magnetic_Energy_Storage_Unit_using_Integral_Controller
Hydro, SC from Verena paper
"""

def simulate_devices_and_limits(IO_dict: dict, 
                              pi_params: dict,
                              input_service_max: np.ndarray,
                              curve_service_min: np.ndarray,
                              title: str='',
                              T_MAX=60,
                              save_path='pics/FFR',
                              service_diff = 1, # derating to ensure some margin
                              STATIC_PF = False,   # use static instead of dynamic
                              Gs_diff = {}  # specify different transfer functions for some devices
                            ):
    """
    IO_dict: dict of IO systems with entries: 
        {(name): (ct.tf(...), device_type, rating)} 
            where ct.tf(...) is the transfer function of the device,
            where device_type is 'lpf', 'bpf' or 'hpf'
            where rating is the power rating in MW
    saturation_dict: dict of saturation limits (min, max power)
    pi_params: dict of pi parameters (kp, ki)

    input_service_max: maximum service curve a.k.a. reference
    curve_service_min: minimum service curve a.k.a. hard constraint to be upheld
    scales_hard_constrains: values for scaling hard constraints
    T_MAX: time horizon of the service in seconds
    save_path: for the plots

    OUTPUT:
        plots of the responses
        VALUE: dict with value function for each coalition
        ENERGY: dict with energy provision for each coalition
        PEAK_POWER: dict with peak power for each coalition
        SHAPELY_VALS: dict with shapely values for each player
    """
    # scales of the reference/hard constraints to test
    scales_hard_constrains = np.linspace(1, service_diff, int(service_diff*10))  # 10 per unit

    my_names = list(IO_dict.keys())
    # get PI controller with physical saturation
    Gs = {name: v[0] for name, v in IO_dict.items()}  # get only the transfer functions
    PIs = {name: get_pi_controller(params=pi_params[name]) for name in my_names}  

    # define error signal
    error = ct.summing_junction(['yref', '-y'], 'e')

    # define u, e as output as well
    u_out = ct.summing_junction(['u'], 'u_out')
    err_out = ct.summing_junction(['e'], 'err_out')

    # create closed-loop systems
    Closed_Loop_systems = {name: ct.interconnect([PIs[name], Gs[name], error, u_out, err_out], inputs=['yref'], outputs=['y', 'u_out', 'err_out']) for name in my_names}

    ## SET PARAMS FOR DVPP
    # example: Solar PV LPF, Wind LPF and Battery HPF
      # scales of the reference/hard constraints to test
    pf_name = 'Static PF' if STATIC_PF else 'Dynamic PF'

    lpf_device_names = {name for name in my_names if IO_dict[name][1] == 'lpf'}
    bpf_device_names = {name for name in my_names if IO_dict[name][1] == 'bpf'}
    hpf_device_names = {name for name in my_names if IO_dict[name][1] == 'hpf'}
    my_names = my_names   # specify otherwise if needed
    tau_c = 0             # 0.081 in Verena paper


    # create value function
    VALUE = {}
    ENERGY = {}
    PEAK_POWER = {}

    for name in my_names:
        # check if fulfills requirements
        g_cl = Closed_Loop_systems[name]
        reward, energy_dict, get_peak_power = plot_reward_value(g_cl, input_service_max, curve_service_min,
                            name, tlim=[0, T_MAX],title=f'{title} {name}',
                            service_rating=IO_dict[name][2]/service_diff,
                            save_path=save_path,
                            scales_hard_constrains=scales_hard_constrains,
                            print_total_energy=True,
                            get_peak_power=True)
        VALUE[(name)] = reward
        ENERGY[(name)] = energy_dict
        PEAK_POWER[(name)] = get_peak_power

    for subset in itertools.combinations(my_names, 2):
        print('Evaluating subset:', subset)
        g_cl, name = DVPP_2_devices(lpf_devices={k: Gs[k] for k in subset if k in lpf_device_names}, 
                            bpf_devices={k: Gs[k] for k in subset if k in bpf_device_names}, 
                            hpf_devices={k: Gs[k] for k in subset if k in hpf_device_names},
                            pi_params=pi_params, tau_c=tau_c,
                            STATIC_PF=STATIC_PF,
                            Gs_diff=Gs_diff)
        reward, energy_dict, get_peak_power = plot_reward_value(g_cl, input_service_max, curve_service_min,
                            name, tlim=[0, T_MAX],title=f'{title} {name} with {pf_name}',
                            service_rating=sum(IO_dict[k][2] for k in subset)/service_diff,
                            save_path=save_path,
                            scales_hard_constrains=scales_hard_constrains,
                            print_total_energy=True,
                            get_peak_power=True)
        VALUE[(name)] = reward
        ENERGY[(name)] = energy_dict    
        PEAK_POWER[(name)] = get_peak_power

    for subset in itertools.combinations(my_names, 3):
        g_cl, name = DVPP_3_devices(lpf_devices={k: Gs[k] for k in subset if k in lpf_device_names}, 
                            bpf_devices={k: Gs[k] for k in subset if k in bpf_device_names}, 
                            hpf_devices={k: Gs[k] for k in subset if k in hpf_device_names},
                            pi_params=pi_params, tau_c=tau_c,
                            STATIC_PF=STATIC_PF,
                            Gs_diff=Gs_diff)
        reward, energy_dict, get_peak_power = plot_reward_value(g_cl, input_service_max, curve_service_min,
                            name, tlim=[0, T_MAX],title=f'{title} {name} with {pf_name}',
                            service_rating=sum(IO_dict[k][2] for k in subset)/service_diff,
                            save_path=save_path,
                            scales_hard_constrains=scales_hard_constrains,
                            print_total_energy=True,
                            get_peak_power=True)
        VALUE[(name)] = reward
        ENERGY[(name)] = energy_dict
        PEAK_POWER[(name)] = get_peak_power

    for subset in itertools.combinations(my_names, 4):
        g_cl, name = DVPP_4_devices(lpf_devices={k: Gs[k] for k in subset if k in lpf_device_names}, 
                            bpf_devices={k: Gs[k] for k in subset if k in bpf_device_names}, 
                            hpf_devices={k: Gs[k] for k in subset if k in hpf_device_names},
                            STATIC_PF=STATIC_PF,
                            pi_params=pi_params, tau_c=tau_c)
        reward, energy_dict, get_peak_power = plot_reward_value(g_cl, input_service_max, curve_service_min,
                            name, tlim=[0, T_MAX],title=f'{title} {name} with {pf_name}',
                            service_rating=sum(IO_dict[k][2] for k in subset)/service_diff,
                            save_path=save_path,
                            scales_hard_constrains=scales_hard_constrains,
                            print_total_energy=True,
                            get_peak_power=True,
                            Gs_diff=Gs_diff)
        VALUE[(name)] = reward
        ENERGY[(name)] = energy_dict
        PEAK_POWER[(name)] = get_peak_power

    # save to path
    is_static = '_static_pf' if STATIC_PF else ''
    pd.DataFrame(VALUE, index=[0]).to_csv(f'{save_path}/value_function{is_static}.csv')
    
    # reformat for shapely value
    VALUE ={tuple(k.replace(' ', '').split('+')): v for k, v in VALUE.items()}
    VALUE[()] = 0  # empty coaltion: value zero
    print('Value function from S -> U:')
    print(VALUE)

    print('Energy provision for each case:')
    print(ENERGY)
    # save energy dict to csv]
    name = 'energy_provision' if not STATIC_PF else 'energy_provision_static_pf'
    pd.DataFrame(ENERGY).to_csv(f'{save_path}/{name}.csv')

    print('Peak power for each case:')
    print(PEAK_POWER)
    name = 'peak_power' if not STATIC_PF else 'peak_power_static_pf'
    pd.DataFrame(PEAK_POWER).to_csv(f'{save_path}/{name}.csv')

    print('Shapely value:')
    SHAPELY_VALS = get_shapely_value(VALUE, my_names)
    print(SHAPELY_VALS)
    # reformat for saving
    name = 'shapely_value' if not STATIC_PF else 'shapely_value_static_pf'
    pd.DataFrame(SHAPELY_VALS, index=[0]).to_csv(f'{save_path}/{name}.csv')

    return VALUE, ENERGY, PEAK_POWER, SHAPELY_VALS
