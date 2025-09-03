# Import the packages needed for the examples included in this notebook
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import control as ct
import pandas as pd

from check_qualification import *

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)

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

# FFR-FCR
# add two curves together
ts_ffr_fcr_max = ts_fcr_max
input_ffr_fcr_max = create_curve(require_ffr_max, t_max=T_MAX_FCR)[1] + input_fcr_max
curve_ffr_fcr_min = create_curve(require_ffr, t_max=T_MAX_FCR)[1] + curve_fcr_min  # minimum curve

# set saturation limits and default pi params
saturation_limits = 33 * np.array([-1, 1])
KP, KI = 16, 92  # if not specified otherwise
params = {}

# get sample systems: PV, Wind, BESS
# parameters from https://www.researchgate.net/publication/353419981_Automatic_Load_Frequency_Control_in_an_Isolated_Micro-grid_with_Superconducting_Magnetic_Energy_Storage_Unit_using_Integral_Controller
tau_PV, K_PV = 1.5, 1
tau_WTG, K_WTG = 2, 1
tau_BESS, K_BESS = 0.1, 1
taus = [tau_PV, tau_WTG, tau_BESS]
Kss = [K_PV, K_WTG, K_BESS]

names = ['PV', 'Wind', 'BESS']
# for n in names:
#     params[n] = {"kp": KP, "ki": KI}
params['PV'] = {"kp": 11.9, "ki": 157.9}
params['Wind'] = {"kp": 11.9, "ki": 118}
params['BESS'] = {"kp": 12, "ki": 2370}

Gs = {}  # transfer functions
for tau, K, name in zip(taus, Kss, names):
    G = ct.tf(K, [tau, 1], inputs=['u'], outputs=['y'])
    Gs[name] = G

# build restricted battery model
p_BESS = {'tau': tau_BESS, 'E_max': .01}  # maximum energy
def update(t, x, u, params={}):
    tau = params.get('tau')
    x0 = - 1/tau * x[0] + u[0]         # np.clip(u[0], -np.inf, E_max - x[1])
    x1 = x[0] #+ x[1]                  # energy state
    return [x0, x1]
def output(t, x, u, params={}):
    tau = params.get('tau')
    return 1/tau * x[0] / (1 + t**2)

Gs['BESS'] = ct.NonlinearIOSystem(
    update, output, inputs=['u'], outputs=['y'], states=2, params=p_BESS
)

if 'BESS' not in names: names.append('BESS')

# PI controller already set

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

PIs = {name: get_pi_controller(params=params[name]) for name in names}

# define error signal
error = ct.summing_junction(['yref', '-y'], 'e')

# define u, e as output as well
u_out = ct.summing_junction(['u'], 'u_out')
err_out = ct.summing_junction(['e'], 'err_out')

# create closed-loop systems
Closed_Loop_systems = {name: ct.interconnect([PIs[name], Gs[name], error, u_out, err_out], inputs=['yref'], outputs=['y', 'u_out', 'err_out']) for name in names}


t = np.linspace(0, T_MAX_FCR, 1000)

# example: Solar PV LPF, Wind LPF and Battery HPF
lpf_devices = {'PV': Gs['PV'], 'Wind': Gs['Wind']}
bpf_devices = {}
hpf_devices = {'BESS': Gs['BESS']}
my_names = ['PV', 'Wind', 'BESS']
tau_c = 0   # 0.081

# run FFR-FCR test

# run for FFR-FCR control
import itertools

# create value function
VALUE = {}
ENERGY = {}

for name in my_names:
    # check if fulfills requirements
    g_cl = Closed_Loop_systems[name]
    dict_out, energy_dict = check_control_system([g_cl], input_ffr_fcr_max, [name], tlim=[0, T_MAX_FCR], title=f'FFR-FCR Response of {name}; sat_lim={saturation_limits}',
                         plot_hard_constrains=curve_ffr_fcr_min,
                         print_total_energy=True)
    VALUE[(name)] = dict_out[name]
    ENERGY[(name)] = energy_dict

for subset in itertools.combinations(my_names, 2):
    g_cl, name = DVPP_2_devices(lpf_devices={k: Gs[k] for k in subset if k in lpf_devices}, 
                          bpf_devices={k: Gs[k] for k in subset if k in bpf_devices}, 
                          hpf_devices={k: Gs[k] for k in subset if k in hpf_devices},
                          Gs=Gs, PIs=PIs, tau_c=tau_c)
    dict_out, energy_dict = check_control_system([g_cl], input_ffr_fcr_max, [name], tlim=[0, T_MAX_FCR], title=f'FFR-FCR Response of {name}; sat_lim={saturation_limits}',
                         plot_hard_constrains=curve_ffr_fcr_min,
                         print_total_energy=True)
    VALUE[(subset)] = dict_out[name]
    ENERGY[(subset)] = energy_dict

for subset in itertools.combinations(my_names, 3):
    g_cl, name = DVPP_3_devices(lpf_devices={k: Gs[k] for k in subset if k in lpf_devices}, 
                          bpf_devices={k: Gs[k] for k in subset if k in bpf_devices}, 
                          hpf_devices={k: Gs[k] for k in subset if k in hpf_devices},
                          Gs=Gs, PIs=PIs, tau_c=tau_c)
    dict_out, energy_dict = check_control_system([g_cl], input_ffr_fcr_max, [name], tlim=[0, T_MAX_FCR], title=f'FFR-FCR Response of {name}; sat_lim={saturation_limits}',
                         plot_hard_constrains=curve_ffr_fcr_min,
                         print_total_energy=True)
    VALUE[(subset)] = dict_out[name]
    ENERGY[(subset)] = energy_dict


for key, value in VALUE.items():
    # convert True/False to 1/0 float
    VALUE[key] = 1.0 if value else 0.0

print('Value function from S -> U:')
print(VALUE)

print('Energy provision for each case:')
print(ENERGY)
# save energy dict to csv]
pd.DataFrame(ENERGY).to_csv('data/energy_provision.csv')

print('Shapely value:')
print(get_shapely_value(VALUE, my_names))