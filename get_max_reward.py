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
from src.get_controllers import get_pi_controller, get_pi_controller_adaptive
from src.get_device_systems import get_time_constants
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm", # 'cm' stands for Computer Modern
    "font.size": 15,  # plt.rcParams.update({'font.size': 15})
})

"""

This is the main script for simulaing the DVPP services and calculating the value functions.

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
                              pf_name = False,   # use static instead of dynamic
                              save_pics = True,
                              time_stamp = None, 
                              price = 1,
                              adaptive_func={},
                              service='FCR',
                              set_service_rating={},
                              bid_received={},
                              rating_threshold=0.09,
                              min_service_rating=0.1,
                              print_debugging=False,
                              set_special_ratings={},
                              HPF_DC_factor=0,
                              debug_grand_coalition=False
                            ):
    """
    IO_dict: dict of IO systems with entries: 
        {(name): (ct.tf(...), device_type, rating)} 
            where ct.tf(...) is the transfer function of the device,
            where device_type is 'lpf', 'bpf' or 'hpf'
            where rating is the power rating in MW
    saturation_dict: dict of saturation limits (min, max power)
    pi_params: dict of pi parameters (kp, ki) for each device
    input_service_max: maximum service curve a.k.a. reference
    curve_service_min: minimum service curve a.k.a. hard constraint to be upheld
    scales_rating: values for scaling hard constraints
    T_MAX: time horizon of the service in seconds
    save_path: for the plots
    price: price of the service in EUR/MW

    # for realized case:
    set_service_rating:
        None: sum of ratings of devices in IO_dict
        dict with entries {(name1, name2, ...): rating} to set the service rating for each coalition
     bid_received:
        dict with entries {(name1, name2, ...): bid} to set the received bid for each coalition
        if empty: forecasted, no bids are considered and maximum reward is calculated

    # general:
    rating_threshold:
        minimum rating in MW for a coalition to be considered
        if the sum of ratings of the devices in the coalition is below this threshold, the coalition is skipped
        and 0 value is assigned
    min_service_rating:
        minimum service rating in MW for each device to be considered
    set_special_ratings:
        dict with entries {(coalition_tuple): service_rating}
        to set special service ratings for specific coalitions and services. service_rating is a percentage of the DC gain i.e., 0.5 means 50% of the DC gain

    OUTPUT:
        (plots of the responses saved in save_path)
        VALUE: dict with value function for each coalition
        ENERGY: dict with energy provision for each coalition
        VALS: dict with values for each coalition
    """
    # scales of the reference/hard constraints to test: from service_diff to 1 in 50 steps
    scales_rating = np.linspace(0.1, 1, 1000)

    my_names = list(IO_dict.keys())
    # get PI controller with physical saturation
    for name, specs in IO_dict.items():
        pi_params[name]['saturation_limits'] = (-specs[2], specs[2])  # -rating, +rating


    ## SET PARAMS FOR DVPP
    # example: Solar PV LPF, Wind LPF and Battery HPF
    # scales of the reference/hard constraints to test
    # my_names = my_names       # specify otherwise if needed
    tau_c = get_time_constants()['tau_c']             # 0.081 in Verena paper

    # create value function
    VALUE = {}
    ENERGY = {}
    PEAK_POWER = {}
    reference_ratings = {}

    # run for all coalitions
    for i in range(1, len(my_names)+1):
        for subset in itertools.combinations(my_names, i):
            # print('Evaluating subset:', subset)
            subset_io_dict = {k: v for k, v in IO_dict.items() if k in subset}
            # calculate actual dc gain
            if subset in set_special_ratings:
                lpf_dc_gain = sum([v[2] for v in subset_io_dict.values()]) * set_special_ratings[subset]
            else:
                lpf_dc_gain = sum([v[2] for v in subset_io_dict.values() if v[1]=='lpf'] + [HPF_DC_factor * v[2] for v in subset_io_dict.values() if v[1]=='hpf']) if service!='FFR' else sum([v[2] for v in subset_io_dict.values()]) #sum([v[2] for v in subset_io_dict.values() if v[1]=='lpf'] + [v[2] * set_special_ratings['hpf'] for v in subset_io_dict.values() if v[1]=='hpf']) 
            all_devices_dc_gain = sum([v[2] for v in subset_io_dict.values()])
            # 2 cases: 
            #     forecasted or realized
            # forecasted: maximize reward
            if bid_received=={}:
                min_rating_i = min_service_rating   # minimum bid to achievew
                reference_rating = lpf_dc_gain    # reference rating
                
            # realized: follow bid
            else:
                # reference_rating is the rating we want to provide
                min_rating_i = bid_received[subset]  # minimum bid to achieve
                reference_rating = set_service_rating[subset]   # or lpf_dc_gain  # reference rating
                # new method: take max of set service rating and lpf dc gain
                #     reasoning: if lpf_dc_gain too low, the refernce would be too low. However, sometimes set_service_rating[subset] is also too low, so choose max of both
                # reference_rating = max(set_service_rating[subset], lpf_dc_gain)
                
            sub_title = f'{title} {"+".join(subset)}'
            if time_stamp is not None:
                # print day and hour from time_stamp
                sub_title += f' on {time_stamp.day}.{time_stamp.month}.{time_stamp.year} at {time_stamp.hour}:00'
            # make cases to skip simulation
            if min_rating_i < min_service_rating:
                if print_debugging: print(f'Skipping {subset} due to low min rating {min_rating_i:.3f} MW < {min_service_rating} MW')
                reward, energy_dict, get_peak_power = 0, {}, 0
            elif reference_rating < rating_threshold:
                if print_debugging: print(f'Skipping {subset} due to low setpoint {reference_rating:.3f} MW < {rating_threshold} MW')
                reward, energy_dict, get_peak_power = -3 * price * min_rating_i, {}, 0
            elif all_devices_dc_gain < min_rating_i:
                # here, the dvpp will fail for sure, no simulation needed
                if print_debugging: print(f'Skipping {subset} due to low all dc gain {all_devices_dc_gain:.3f} MW < {min_rating_i} MW')
                reward, energy_dict, get_peak_power = -3 * price * min_rating_i, {}, 0
            else:
                reward, energy_dict, get_peak_power = get_DVPP(
                                IO_dict=subset_io_dict,
                                pi_params=pi_params, dpfs=dpfs, 
                                vref=input_service_max, min_hard_constrains=curve_service_min,
                                scales_rating=scales_rating,
                                tlim=[0, T_MAX],
                                print_total_energy=False,
                                get_peak_power=False,
                                save_path=save_path,
                                tau_c=tau_c,
                                title=sub_title,
                                pf_name=pf_name,
                                price=price,
                                save_pics=save_pics,
                                adaptive_func=adaptive_func,
                                reference_rating=reference_rating,
                                total_dc_gain=lpf_dc_gain,
                                min_service_rating=min_rating_i,
                                debug_grand_coalition=debug_grand_coalition
                                )

            VALUE[subset] = reward
            ENERGY[subset] = energy_dict    
            PEAK_POWER[subset] = get_peak_power
            reference_ratings[subset] = reference_rating

    return VALUE, ENERGY, reference_ratings
