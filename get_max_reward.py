# Import the packages needed for the examples included in this notebook
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import control as ct
import pandas as pd
import itertools

# from check_qualification import *
from src.dvpp_helper import get_DVPP   # DVPP_2_devices, DVPP_3_devices, DVPP_4_devices, plot_DVPP_2_devices
from src.get_single_cl import get_single_cl
from src.get_controllers import get_pi_controller
from src.get_device_systems import get_time_constants
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
                              dpfs: dict,  # specify dynamic participation factors
                              title: str='',
                              T_MAX=60,
                              save_path='pics/FFR',
                              service_diff = 0.1, # fraction of minimum service to supply. at 1 MW -> 0.1 * 1 MW = 0.1 MW
                              STATIC_PF = False,   # use static instead of dynamic
                              
                              save_plots = True,
                              save_pics = True,
                              x_scenario = 0,  # specify scenario if needed from 1...Sx
                              price = 1,
                              adaptive_func={},
                              service='FCR',
                              set_service_rating=None,
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
    # scales of the reference/hard constraints to test: from service_diff to 1 in 50 steps
    scales_hard_constrains = np.linspace(service_diff, 1, 50)

    my_names = list(IO_dict.keys())
    # get PI controller with physical saturation
    for name, specs in IO_dict.items():
        pi_params[name]['saturation_limits'] = (-specs[2], specs[2])  # -rating, +rating
    PIs = {name: get_pi_controller(params=pi_params[name]) for name in my_names}  

    # same delay as for dpf's
    small_delay = ct.tf([1], [get_time_constants()['tau_c'], 1], inputs='yref', outputs='yref_delay')

    # create summing junction for error
    error = ct.summing_junction(['yref_delay', '-y'], 'e')

    # define u, e as output as well
    u_out = ct.summing_junction(['u'], 'u_out')
    err_out = ct.summing_junction(['e'], 'err_out')

    # create closed-loop systems
    Closed_Loop_systems = {name: ct.interconnect([small_delay, error, PIs[name], IO_dict[name][0], u_out, err_out], inputs=['yref'], outputs=['y', 'u_out', 'err_out']) for name in my_names}

    ## SET PARAMS FOR DVPP
    # example: Solar PV LPF, Wind LPF and Battery HPF
      # scales of the reference/hard constraints to test
    pf_name = 'SPF' if STATIC_PF else 'DPF'
    pf_name = 'ADPF' if (adaptive_func!={} and STATIC_PF!='DPF' and STATIC_PF) else pf_name
    if x_scenario > 1:
        pf_name += f' Scenario {x_scenario}'
    my_names = my_names       # specify otherwise if needed
    tau_c = get_time_constants()['tau_c']             # 0.081 in Verena paper


    # create value function
    VALUE = {}
    ENERGY = {}
    PEAK_POWER = {}

    for name in my_names:
        # check if fulfills requirements
        g_cl = Closed_Loop_systems[name]
        reward, energy_dict, get_peak_power = get_single_cl(g_cl, input_service_max, curve_service_min,
                            name, tlim=[0, T_MAX],title=f'{title} {name} {pf_name}',
                            service_rating=IO_dict[name][2],
                            save_path=save_path,
                            scales_hard_constrains=scales_hard_constrains,
                            print_total_energy=True,
                            get_peak_power=True,
                            save_plots=save_plots,
                            price=price,
                            save_pics=save_pics,
                            adaptive_func=adaptive_func)
        VALUE[(name,)] = reward
        ENERGY[(name,)] = energy_dict
        PEAK_POWER[(name,)] = get_peak_power

    for i in range(2, len(my_names)+1):
        for subset in itertools.combinations(my_names, i):
            print('Evaluating subset:', subset)
            subset_io_dict = {k: v for k, v in IO_dict.items() if k in subset}
            # set service rating
            # sum of ratings of LPF devices only, except for FFR where all devices can have enough energy to contribute
            if set_service_rating:
                sum_service_rating = set_service_rating[service]
            else:
                sum_service_rating = sum([v[2] for v in subset_io_dict.values()])   # sum([v[2] for v in subset_io_dict.values() if v[1]=='lpf']) if service!='FFR' else 
            reward, energy_dict, get_peak_power = get_DVPP(
                            IO_dict=subset_io_dict,
                            pi_params=pi_params, dpfs=dpfs, 
                            vref=input_service_max, min_hard_constrains=curve_service_min,
                            scales_hard_constrains=scales_hard_constrains,
                            tlim=[0, T_MAX],
                            print_total_energy=False,
                            get_peak_power=False,
                            save_plots=save_plots,
                            save_path=save_path,
                            tau_c=tau_c,
                            title=f'{title} {"+".join(subset)} {pf_name}',
                            STATIC_PF=STATIC_PF,
                            price=price,
                            save_pics=save_pics,
                            adaptive_func=adaptive_func,
                            sum_service_rating=sum_service_rating
                            )

            VALUE[subset] = reward
            ENERGY[subset] = energy_dict    
            PEAK_POWER[subset] = get_peak_power

    return VALUE, ENERGY, PEAK_POWER
