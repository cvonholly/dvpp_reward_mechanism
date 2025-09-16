# Import the packages needed for the examples included in this notebook
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import control as ct
import pandas as pd
import itertools

# from check_qualification import *
from src.dvpp_helper import get_DVPP   # DVPP_2_devices, DVPP_3_devices, DVPP_4_devices, plot_DVPP_2_devices
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
                              service_diff = 0.1, # fraction of minimum service to supply. at 1 MW -> 0.1 * 1 MW = 0.1 MW
                              STATIC_PF = False,   # use static instead of dynamic
                              Gs_diff = {},  # specify different transfer functions for some devices
                              save_plots = True,
                              save_data = False,
                              x_scenario = 0,  # specify scenario if needed from 1...Sx
                              price = 1
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
    price: price of the service in EUR/MW

    OUTPUT:
        plots of the responses
        VALUE: dict with value function for each coalition
        ENERGY: dict with energy provision for each coalition
        PEAK_POWER: dict with peak power for each coalition
        SHAPELY_VALS: dict with shapely values for each player
    """
    # scales of the reference/hard constraints to test
    scales_hard_constrains = np.linspace(service_diff, 1, int(3/service_diff))

    my_names = list(IO_dict.keys())
    # get PI controller with physical saturation
    for name, specs in IO_dict.items():
        pi_params[name]['saturation_limits'] = (-specs[2], specs[2])  # -rating, +rating
    PIs = {name: get_pi_controller(params=pi_params[name]) for name in my_names}  

    # define error signal
    error = ct.summing_junction(['yref', '-y'], 'e')

    # define u, e as output as well
    u_out = ct.summing_junction(['u'], 'u_out')
    err_out = ct.summing_junction(['e'], 'err_out')

    # create closed-loop systems
    Closed_Loop_systems = {name: ct.interconnect([PIs[name], IO_dict[name][0], error, u_out, err_out], inputs=['yref'], outputs=['y', 'u_out', 'err_out']) for name in my_names}

    ## SET PARAMS FOR DVPP
    # example: Solar PV LPF, Wind LPF and Battery HPF
      # scales of the reference/hard constraints to test
    pf_name = 'Static PF' if STATIC_PF else 'Dynamic PF'
    my_names = my_names   # specify otherwise if needed
    tau_c = 0.081             # 0.081 in Verena paper


    # create value function
    VALUE = {}
    ENERGY = {}
    PEAK_POWER = {}

    for name in my_names:
        # check if fulfills requirements
        g_cl = Closed_Loop_systems[name]
        reward, energy_dict, get_peak_power = plot_reward_value(g_cl, input_service_max, curve_service_min,
                            name, tlim=[0, T_MAX],title=f'{title} {name} with {pf_name} Scenario {x_scenario}',
                            service_rating=IO_dict[name][2],
                            save_path=save_path,
                            scales_hard_constrains=scales_hard_constrains,
                            print_total_energy=True,
                            get_peak_power=True,
                            save_plots=save_plots,
                            price=price)
        VALUE[(name,)] = reward
        ENERGY[(name,)] = energy_dict
        PEAK_POWER[(name,)] = get_peak_power

    for i in range(2, len(my_names)+1):
        for subset in itertools.combinations(my_names, i):
            print('Evaluating subset:', subset)
            subset_io_dict = {k: v for k, v in IO_dict.items() if k in subset}
            reward, energy_dict, get_peak_power = get_DVPP(
                            IO_dict=subset_io_dict,
                            pi_params=pi_params, Gs_diff=Gs_diff, 
                            vref=input_service_max, min_hard_constrains=curve_service_min,
                            scales_hard_constrains=scales_hard_constrains,
                            tlim=[0, T_MAX],
                            print_total_energy=False,
                            get_peak_power=False,
                            save_plots=save_plots,
                            save_path=save_path,
                            tau_c=tau_c,
                            title=f'{title} {"+".join(subset)} with {pf_name} Scenario {x_scenario}',
                            STATIC_PF=STATIC_PF,
                            price=price,
                            )

            VALUE[subset] = reward
            ENERGY[subset] = energy_dict    
            PEAK_POWER[subset] = get_peak_power

    # save to path
    if save_data:
        is_static = '_static_pf' if STATIC_PF else ''
        pd.DataFrame(VALUE, index=[0]).to_csv(f'{save_path}/value_function{is_static}.csv')
        
    # reformat for shapely value
    # VALUE ={tuple(k.replace(' ', '').split('+')): v for k, v in VALUE.items()}
    VALUE[()] = 0  # empty coaltion: value zero
    print('Value function from S -> U:')
    print(VALUE)

    # print('Energy provision for each case:')
    # print(ENERGY)
    # # save energy dict to csv
    # if save_data:
    #     name = 'energy_provision' if not STATIC_PF else 'energy_provision_static_pf'
    #     pd.DataFrame(ENERGY).to_csv(f'{save_path}/{name}.csv')

    # print('Peak power for each case:')
    # print(PEAK_POWER)
    # if save_data:
    #     name = 'peak_power' if not STATIC_PF else 'peak_power_static_pf'
    #     pd.DataFrame(PEAK_POWER).to_csv(f'{save_path}/{name}.csv')

    print('Shapely value:')
    SHAPELY_VALS = get_shapely_value(VALUE, my_names)
    print(SHAPELY_VALS)
    # reformat for saving
    if save_data:
        name = 'shapely_value' if not STATIC_PF else 'shapely_value_static_pf'
        pd.DataFrame(SHAPELY_VALS, index=[0]).to_csv(f'{save_path}/{name}.csv')

    return VALUE, ENERGY, PEAK_POWER, SHAPELY_VALS
