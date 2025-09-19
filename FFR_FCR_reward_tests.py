# Import the packages needed for the examples included in this notebook
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import control as ct
import pandas as pd

from check_qualification import *
from src.dvpp_helper import DVPP_2_devices, DVPP_3_devices, DVPP_4_devices
from get_single_cl import plot_reward_value
from src.game_theory_helpers import get_shapley_value

"""

in this script the reward is calculated for FFR-FCR provision,
for the same numbers used in Verena's paper

"""

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Dark2.colors)

def create_curve(params, t_max, n_points=1000):
    t = np.linspace(0, t_max, n_points)
    v = np.zeros_like(t)
    for (t_start, t_end), (v1, v2) in params.items():
        mask = (t >= t_start) & (t < t_end)
        v[mask] = np.linspace(v1, v2, np.sum(mask))
    return t, v

# dict with {(t_start, t_end): (v1, v2)}
# FFR
Dp = 0.04
MP_max = 49.167
T_MAX_FFR = 10
require_ffr = {(0, .7): (0, 0), (.7, 5): (1/Dp, 1/Dp), (5, T_MAX_FFR): (1/Dp, 0)}
require_ffr_max = {(0, .7): (0, MP_max), (.7, 5): (MP_max, MP_max), (5, 10): (MP_max, 0)}
ts_ffr_max, input_ffr_max = create_curve(require_ffr_max, t_max=T_MAX_FFR)
_, curve_ffr_min = create_curve(require_ffr, t_max=T_MAX_FFR)

# FCR
ti, ta, Dp = 2, 30, 0.06
RP_MAX = 32.56
T_MAX_FCR = 60
require_fcr = {(ti, ta): (0, 1/Dp), (ta, T_MAX_FCR): (1/Dp, 1/Dp)}
require_fcr_max = {(ti, ta): (0, RP_MAX), (ta, T_MAX_FCR): (RP_MAX, RP_MAX)}
ts_fcr_max, input_fcr_max = create_curve(require_fcr_max, t_max=T_MAX_FCR)
_, curve_fcr_min = create_curve(require_fcr, t_max=T_MAX_FCR)  # minimum curve

# FCR-D (finnish market)
ti = 7.5
T_MAX_FCR_D = 60
require_fcr = {(0, ti): (0, 0.86*1/Dp), (ti, T_MAX_FCR_D): (0.86*1/Dp, 0.86*1/Dp), (ta, T_MAX_FCR): (1/Dp, 1/Dp)}


# FFR-FCR
# add two curves together
ts_ffr_fcr_max = ts_fcr_max
input_ffr_fcr_max = create_curve(require_ffr_max, t_max=T_MAX_FCR)[1] + input_fcr_max
curve_ffr_fcr_min = create_curve(require_ffr, t_max=T_MAX_FCR)[1] + curve_fcr_min  # minimum curve

# set saturation limits and default pi params
saturation_limits = 33 * np.array([-1, 1])
KP, KI = 16, 92  # if not specified otherwise
params = {}

"""
Define systems: PV, Wind, BESS, SC, Hydro
parameters from https://www.researchgate.net/publication/353419981_Automatic_Load_Frequency_Control_in_an_Isolated_Micro-grid_with_Superconducting_Magnetic_Energy_Storage_Unit_using_Integral_Controller


"""
tau_PV, K_PV = 1.5, 1
tau_WTG, K_WTG = 2, 1
tau_BESS, K_BESS = 0.1, 1
taus = [tau_PV, tau_WTG, tau_BESS]
Kss = [K_PV, K_WTG, K_BESS]

my_names = ['PV', 'Wind', 'BESS']
# for n in names:
#     params[n] = {"kp": KP, "ki": KI}
params['PV'] = {"kp": 11.9, "ki": 157.9}
params['Wind'] = {"kp": 11.9, "ki": 118}
params['BESS'] = {"kp": 12, "ki": 2370}

Gs = {}  # transfer functions
for tau, K, name in zip(taus, Kss, my_names):
    G = ct.tf(K, [tau, 1], inputs=['u'], outputs=['y'])
    Gs[name] = G

# build restricted battery model
p_BESS = {'tau': tau_BESS}  # maximum energy
def update(t, x, u, params={}):
    tau = params.get('tau')
    x0 = - 1/tau * x[0] + u[0]         # np.clip(u[0], -np.inf, E_max - x[1])
    x1 = x[0] #+ x[1]                  # energy state
    return [x0, x1]
def output(t, x, u, params={}):
    tau = params.get('tau')
    return 1/tau * x[0] / (1 + t**1.2)

Gs['BESS'] = ct.NonlinearIOSystem(
    update, output, inputs=['u'], outputs=['y'], states=2, params=p_BESS
)

if 'BESS' not in my_names: my_names.append('BESS')

# build restricted supercap model
tau_SC = 0.01
p_SC = {'tau': tau_SC}  # maximum energy
def update(t, x, u, params={}):
    tau = params.get('tau')
    x0 = - 1/tau * x[0] + u[0]         # np.clip(u[0], -np.inf, E_max - x[1])
    x1 = x[0] #+ x[1]                  # energy state
    return [x0, x1]
def output(t, x, u, params={}):
    tau = params.get('tau')
    return 1/tau * x[0] / (1 + t**3)

Gs['SC'] = ct.NonlinearIOSystem(
    update, output, inputs=['u'], outputs=['y'], states=2, params=p_SC
)

if 'SC' not in my_names: my_names.append('SC')

params["SC"] = params['BESS']   # set same PI as for BESS

# # get hydro system from Verena paper
# Rg, Rt = 0.03, 0.38
# taug, taur, tauw = 0.2, 5, 1
# s = sp.symbols('s')

# def get_hydro_tf():
#     T_hydro = -1/Rg / (taug*s + 1) * (taur*s+1) / (Rt/Rg*taur*s+1) * (1-tauw*s) / (1+1/2*tauw*s)
    # return sympy_to_tf(sp.simplify(T_hydro))

# T_hydro = get_hydro_tf() * Gs['Wind'] # add typical delay
# T_hydro = ct.tf(T_hydro.num, T_hydro.den, inputs=['u'], outputs=['y'])
# my_names.append('Hydro')
# Gs['Hydro'] = T_hydro

# # set pi params for hydro
# # calculated and tuned in matlab
# params["Hydro"] = {"kp": -0.0796, "ki": -0.09788}

#
#  create PI controller with anti-windup
#
def pi_update(t, x, u, params={}):
    # Get the controller parameters that we need
    kp = params.get('kp')
    ki = params.get('ki', 0.1)
    kaw = params.get('kaw', 15 * ki)  # anti-windup gain

    err = u[0]   # error signal
    z = x[0]     # integrated error

    # Compute unsaturated output
    u_a = kp * err + ki * z

    # Compute anti-windup compensation (scale by ki to account for structure)
    u_aw = kaw/ki * (u_a - np.clip(u_a, saturation_limits[0], saturation_limits[1])) if ki != 0 else 0

    # State is the integrated error, minus anti-windup compensation
    return err - u_aw

def pi_output(t, x, u, params={}):
    # Get the controller parameters that we need
    kp = params.get('kp')
    ki = params.get('ki')

    # Assign variables for inputs and states (for readability)
    err = u[0]  # error signal
    z = x[0]    # integrated error

    # PI controller with saturation
    return np.clip(kp * err + ki * z, saturation_limits[0], saturation_limits[1]) # account for saturation

def get_pi_controller(params):
    # old: without saturation
    # return ct.tf(
    #     [params['kp'], params['ki']], [1, 0],
    #     inputs=['e'], outputs=['u']
    # )
    return ct.NonlinearIOSystem(
        pi_update, pi_output, name='control',
        inputs=['e'], outputs=['u'], states=['z'],
        params=params)

PIs = {name: get_pi_controller(params=params[name]) for name in my_names}

# define error signal
error = ct.summing_junction(['yref', '-y'], 'e')

# define u, e as output as well
u_out = ct.summing_junction(['u'], 'u_out')
err_out = ct.summing_junction(['e'], 'err_out')

# create closed-loop systems
Closed_Loop_systems = {name: ct.interconnect([PIs[name], Gs[name], error, u_out, err_out], inputs=['yref'], outputs=['y', 'u_out', 'err_out']) for name in my_names}


t = np.linspace(0, T_MAX_FCR, 1000)

## SET PARAMS FOR DVPP
# example: Solar PV LPF, Wind LPF and Battery HPF
scales_hard_constrains = [1.2, 1.4, 1.6, 1.8, 2, 3]
STATIC_PF = False   # use static instead of dynamic
lpf_devices = {'PV': Gs['PV'], 'Wind': Gs['Wind']}
bpf_devices = {'BESS': Gs['BESS']}
hpf_devices = {'SC': Gs['SC']}
my_names = my_names   # specify otherwise if needed
tau_c = 0   # 0.081

# run FFR-FCR test

# run for FFR-FCR control
import itertools

# create value function
VALUE = {}
ENERGY = {}
PEAK_POWER = {}

for name in my_names:
    # check if fulfills requirements
    g_cl = Closed_Loop_systems[name]
    reward, energy_dict, get_peak_power = plot_reward_value(g_cl, input_ffr_fcr_max, curve_ffr_fcr_min,
                                                            name, tlim=[0, T_MAX_FCR], title=f'FFR-FCR Response of {name}; sat_lim={saturation_limits}',
                         scales_hard_constrains=scales_hard_constrains,
                         print_total_energy=True,
                         get_peak_power=True)
    VALUE[(name)] = reward
    ENERGY[(name)] = energy_dict
    PEAK_POWER[(name)] = get_peak_power

for subset in itertools.combinations(my_names, 2):
    g_cl, name = DVPP_2_devices(lpf_devices={k: Gs[k] for k in subset if k in lpf_devices}, 
                          bpf_devices={k: Gs[k] for k in subset if k in bpf_devices}, 
                          hpf_devices={k: Gs[k] for k in subset if k in hpf_devices},
                          pi_params=params, tau_c=tau_c,
                          STATIC_PF=STATIC_PF)
    reward, energy_dict, get_peak_power = plot_reward_value(g_cl, input_ffr_fcr_max, curve_ffr_fcr_min,
                                                            name, tlim=[0, T_MAX_FCR], title=f'FFR-FCR Response of {name}; sat_lim={saturation_limits}',
                         scales_hard_constrains=scales_hard_constrains,
                         print_total_energy=True,
                         get_peak_power=True)
    VALUE[(name)] = reward
    ENERGY[(name)] = energy_dict
    PEAK_POWER[(name)] = get_peak_power

for subset in itertools.combinations(my_names, 3):
    g_cl, name = DVPP_3_devices(lpf_devices={k: Gs[k] for k in subset if k in lpf_devices}, 
                          bpf_devices={k: Gs[k] for k in subset if k in bpf_devices}, 
                          hpf_devices={k: Gs[k] for k in subset if k in hpf_devices},
                          pi_params=params, tau_c=tau_c)
    reward, energy_dict, get_peak_power = plot_reward_value(g_cl, input_ffr_fcr_max, curve_ffr_fcr_min,
                                                            name, tlim=[0, T_MAX_FCR], title=f'FFR-FCR Response of {name}; sat_lim={saturation_limits}',
                         scales_hard_constrains=scales_hard_constrains,
                         print_total_energy=True,
                         get_peak_power=True)
    VALUE[(name)] = reward
    ENERGY[(name)] = energy_dict
    PEAK_POWER[(name)] = get_peak_power

for subset in itertools.combinations(my_names, 4):
    g_cl, name = DVPP_4_devices(lpf_devices={k: Gs[k] for k in subset if k in lpf_devices}, 
                          bpf_devices={k: Gs[k] for k in subset if k in bpf_devices}, 
                          hpf_devices={k: Gs[k] for k in subset if k in hpf_devices},
                          pi_params=params, tau_c=tau_c)
    reward, energy_dict, get_peak_power = plot_reward_value(g_cl, input_ffr_fcr_max, curve_ffr_fcr_min,
                                                            name, tlim=[0, T_MAX_FCR], title=f'FFR-FCR Response of {name}; sat_lim={saturation_limits}',
                         scales_hard_constrains=scales_hard_constrains,
                         print_total_energy=True,
                         get_peak_power=True)
    VALUE[(name)] = reward
    ENERGY[(name)] = energy_dict
    PEAK_POWER[(name)] = get_peak_power

# reformat for shapely value
VALUE ={tuple(k.replace(' ', '').split('+')): v for k, v in VALUE.items()}
VALUE[()] = 0  # empty coaltion: value zero
print('Value function from S -> U:')
print(VALUE)

print('Energy provision for each case:')
print(ENERGY)
# save energy dict to csv]
name = 'energy_provision' if not STATIC_PF else 'energy_provision_static_pf'
pd.DataFrame(ENERGY).to_csv(f'data/{name}.csv')

print('Peak power for each case:')
print(PEAK_POWER)
name = 'peak_power' if not STATIC_PF else 'peak_power_static_pf'
pd.DataFrame(PEAK_POWER).to_csv(f'data/{name}.csv')

print('Shapely value:')
SHAPELY_VALS = get_shapley_value(VALUE, my_names)
print(SHAPELY_VALS)
# reformat for saving
name = 'shapely_value' if not STATIC_PF else 'shapely_value_static_pf'
pd.DataFrame(SHAPELY_VALS, index=[0]).to_csv(f'data/{name}.csv')
