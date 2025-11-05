import sympy as sp
import control as ct
import numpy as np

from src.get_controllers import sympy_to_tf


def get_time_constants() -> dict:
    constants = {}
    constants['PV'] = 1.5
    constants['Wind'] = 2
    constants['BESS'] = 0.1
    constants['SC'] = 0.01
    Rg, Rt, taur = 0.03, 0.38, 5
    constants['Hydro'] = 5 # test Rt/Rg*taur  # from Verena paper
    constants['tau_c'] = 0.081   # time constant used in verenas paper
    return constants

def get_special_pfs() -> dict:
    # get adpf for systems
    # only needed if adpf is not just 1st order time constant
    Rg, Rt = 0.03, 0.38
    taug, taur, tauw = 0.2, 5, 1
    adpfs = {}
    s = ct.tf('s')
    adpfs['Hydro'] = (taur*s+1) / (Rt/Rg*taur*s+1) * (1-tauw*s) / (1+1/2*tauw*s)
    return adpfs


def get_pv_sys():
    tau_PV = get_time_constants()['PV']
    G_PV = ct.tf([1], [tau_PV, 1], inputs=['u'], outputs=['y'])
    return G_PV

def get_wind_sys():
    tau_WTG = get_time_constants()['Wind']
    G_WTG = ct.tf([1], [tau_WTG, 1], inputs=['u'], outputs=['y'])
    return G_WTG


def get_hydro_tf():
    """
    return:
        hydro IO control system
    """
    # get hydro system from Verena paper
    Rg, Rt = 0.03, 0.38
    taug, taur, tauw = 0.2, 5, 1

    def get_hydro_tf():
        s = sp.symbols('s')
        T_hydro = -1/Rg / (taug*s + 1) * (taur*s+1) / (Rt/Rg*taur*s+1) * (1-tauw*s) / (1+1/2*tauw*s)
        return sympy_to_tf(sp.simplify(T_hydro))

    T_hydro = get_hydro_tf()
    T_hydro = ct.tf(T_hydro.num, T_hydro.den, inputs=['u'], outputs=['y'])
    
    return T_hydro

def get_bess_io_sys(tau_BESS=0.1, t_drop=7, drop_exp=1.2):
    """
    params:
        tau_BESS: time constant of the battery
        t_drop: time after which the battery output starts to drop (in seconds)
    """
    p_BESS = {'tau': tau_BESS, 't_drop': t_drop, 'drop_exp': drop_exp} 

    def update(t, x, u, params={}):
        tau = params.get('tau')
        x0 = - 1/tau * x[0] + u[0]         # np.clip(u[0], -np.inf, E_max - x[1])
        x1 = x[0]                          # energy state
        return [x0, x1]
    
    def output(t, x, u, params={}):
        return 1/params.get('tau') * x[0] / (max(1, t-params.get('t_drop'))**params.get('drop_exp'))

    return ct.NonlinearIOSystem(
        update, output, inputs=['u'], outputs=['y'], states=2, params=p_BESS
    )

def get_bess_energy_sys(tau_BESS=0.1, e_max=10):
    """
    OLD VERSION: use get_bess_sys for new version

    params:
        tau_BESS: time constant of the battery
        e_max: maximum energy capacity of the battery
    """
    p_BESS = {'tau': tau_BESS, 'e_max': e_max} 

    def update(t, x, u, params={}):
        tau = params.get('tau')
        x0 = - 1/tau * x[0] + u[0]         # np.clip(u[0], -np.inf, E_max - x[1])
        x1 = x[0]                          # energy state
        return [x0, x1]
    
    def output(t, x, u, params={}):
        return 1/params.get('tau') * x[0] * (x[1] < params.get('e_max'))

    return ct.NonlinearIOSystem(
        update, output, inputs=['u'], outputs=['y'], states=2, params=p_BESS
    )

def get_bess_sys(tau_BESS=0.1, e_max=10):
    """
    NEW VERSION

    params:
        tau_BESS: time constant of the battery
        e_max: maximum energy capacity of the battery in Watts * seconds
    """
    p_BESS = {'tau': tau_BESS, 'e_max': e_max} 

    def update(t, x, u, params={}):
        tau = params.get('tau')
        x0 = - 1/tau * x[0] + u[0]         # np.clip(u[0], -np.inf, E_max - x[1])
        x1 = x[0] / tau                    # energy state
        return [x0, x1]
    
    def output(t, x, u, params={}):
        return [1/params.get('tau') * x[0] * (x[1] < params.get('e_max')), x[1]]

    return ct.NonlinearIOSystem(
        update, output, inputs=['u'], outputs=['y', 'energy'], states=2, params=p_BESS
    )

def get_sc_time_sys(t_drop, drop_exp):
    """
    params:
        t_drop: maximum energy capacity of the supercapacitor
        drop_exp: exponent of the drop function (higher -> faster drop)
    """
    p_SC = {'tau': get_time_constants()['SC'], 't_drop': t_drop, 'drop_exp': drop_exp} 

    def update(t, x, u, params={}):
        tau = params.get('tau')
        x0 = - 1/tau * x[0] + u[0]         # np.clip(u[0], -np.inf, E_max - x[1])
        x1 = x[0]                         # energy state
        return [x0, x1]
    
    def output(t, x, u, params={}):
        return 1/params.get('tau') * x[0] / (max(1, t-params.get('t_drop'))**params.get('drop_exp'))

    return ct.NonlinearIOSystem(
        update, output, inputs=['u'], outputs=['y'], states=2, params=p_SC
    )

# def get_sc_time_sys(t_max):
#     """
#     params:
#         t_max: maximum time duration of the supercapacitor
#     """
#     p_SC = {'tau': get_time_constants()['SC'], 't_max': t_max} 

#     def update(t, x, u, params={}):
#         tau = params.get('tau')
#         x0 = - 1/tau * x[0] + u[0]         # np.clip(u[0], -np.inf, E_max - x[1])
#         x1 = x[0]                         # energy state
#         return [x0, x1]
    
#     def output(t, x, u, params={}):
#         return 1/params.get('tau') * x[0] * (t <= params.get('t_max'))

#     return ct.NonlinearIOSystem(
#         update, output, inputs=['u'], outputs=['y'], states=2, params=p_SC
#     )

def get_ADPF(lpf_devices, bpf_devices, hpf_devices, 
             IO_dict, sum_service_rating, dpfs, adaptive_func,
             tau_c=0,
             T_END=60):
    """
    built non-linear time variant ADPFs

    important:
        adaptive_func gains always have to add up to 1 (e.g. sin(t)^2 + cos(t)^2 = 1)
            otherwise the DVPP will not track the reference properly

    """
    # restricted_devices = list(adaptive_func.keys())  # devices which are restricted by time-varying production
    mks = {}
    time_constants = get_time_constants()
    lpf_names = list(lpf_devices.keys())
    thetas = {}
    for name, _  in lpf_devices.items():
        theta_i = IO_dict[name][2] / sum_service_rating
        thetas[name] = theta_i
        if name not in adaptive_func:
            mks[name] = dpfs[name] * theta_i  # ADPF is DPF
        else:
            mks[name] = get_adaptive_dc_sys({'theta': theta_i, 'tau': time_constants[name], 'gain': adaptive_func[name]})  # Define steady-state ADPFs as LPFs
    # todo: implement bpf devices
    # for name, g in bpf_devices.items():
    #     if name in Gs_diff: g = Gs_diff[name]  # take other tranfer function if specified
    #     mks[name] = g * (g - Gs_sum)   # Fix intermediate ADPFs as BPFs
    #     Gs_sum += g * (g - Gs_sum)
    for name, _ in hpf_devices.items():
        if len(lpf_names) == 1:
            mks[name] = get_adaptive_hpf_for_1lpf(time_constants[lpf_names[0]], adaptive_func[lpf_names[0]], thetas[lpf_names[0]])   # Fix fastest device’s ADPF as HPF
        elif len(lpf_names) == 2:
            mks[name] = get_adaptive_hpf_for_2lpf(time_constants[lpf_names[0]], time_constants[lpf_names[1]], adaptive_func[lpf_names[0]], adaptive_func[lpf_names[1]],
                                                  thetas[lpf_names[0]], thetas[lpf_names[1]],
                                                  D=np.array([[.75 if T_END>1e3 else 1]])
                                                  )   # Fix fastest device’s ADPF as HPF
    return mks


# def get_static_pf_varying_ref(IO_dict, adaptive_func):
#     """
#     static PF for the time-varying production case

#     set theta_i = rating_i / sum(rating_j) for all i, j in devices
#     for time-varying production, get_adaptive_dc_sys(...) is used to scale the steady-state gain
#     """
#     mks = {}
#     time_constants = get_time_constants()
#     sum_service_rating = sum([specs[2] for _, specs in IO_dict.items()])
#     for name, _ in IO_dict.items():
#         theta_i = IO_dict[name][2] / sum_service_rating
#         if name in adaptive_func:
#             mks[name] = get_adaptive_dc_sys({'theta': theta_i, 'tau': time_constants[name], 'gain': adaptive_func[name]})  # Define steady-state ADPFs as LPFs
#         else:
#             mks[name] = ct.tf([theta_i], [time_constants[name], 1]) 
#     return mks


def get_adaptive_hpf_for_1lpf(tau1, adaptive_func1, theta1):
    """
    params:
        gain: function t -> theta where theta is time-varying dc gain
    """
    A = np.array([[-1/tau1]])
    B = np.vstack([[1/tau1]])
    D = np.array([[1]])

    def update(t, x, u, params={}):
        return params['A'] @ x + params['B'] @ u      
    
    def output(t, x, u, params={}):
        C = np.array([[-params['theta1'] * params['adaptive_func1'](t)]])
        return C @ x + D @ u
    
    params = {'A': A, 'B': B, 'D': D, 'adaptive_func1': adaptive_func1, 'theta1': theta1}
    
    return ct.NonlinearIOSystem(
        update, output, inputs=['u'], outputs=['y'], states=1, params=params
    )

def get_adaptive_hpf_for_2lpf(tau1, tau2, adaptive_func1, adaptive_func2,
                              theta1, theta2, D=None):
    """
    params:
        gain: function t -> theta where theta is time-varying dc gain
    """
    A = np.array([[-1/tau1, 0], [0, -1/tau2]])
    B = np.array([[1/tau1], [1/tau2]])
    D = np.array([[1]]) if D is None else D


    def update(t, x, u, params={}):
        return params['A'] @ x + params['B'] @ u
    
    def output(t, x, u, params={}):
        C = np.array([[-params['theta1'] * params['adaptive_func1'](t), -params['theta2'] * params['adaptive_func2'](t)]])
        return C @ x + params['D'] @ u
    
    params = {'A': A, 'B': B, 'D': D, 'adaptive_func1': adaptive_func1, 'adaptive_func2': adaptive_func2, 
              'theta1': theta1, 'theta2': theta2}
    return ct.NonlinearIOSystem(
        update, output, inputs=['u'], outputs=['y'], states=2, params=params
    )

def get_adaptive_dc_sys(params):
    """
    get adaptive dc gain for all times:
        theta_i(t) = theta_i * gain_i(t) where gain_i(t) is a the relative gain of nominal capacity depending on e.g. weather conditions

    params:
        gain: function t -> theta where theta is time-varying dc gain
    """
    def update(t, x, u, params={}):
        x0 = - 1/params.get('tau') * x[0] + u[0]
        return x0
    
    def output(t, x, u, params={}):
        return params.get('theta')/params.get('tau') * x[0] * params.get('gain')(t)
    
    return ct.NonlinearIOSystem(
        update, output, inputs=['u'], outputs=['y'], states=1, params=params
    )

def get_adaptive_saturation_sys(params):
    """
    params:
        gain: function t -> theta where theta is time-varying dc gain
        saturation_limits: float
    
    clips input u to +/- gain(t)
    """
    def update(t, x, u, params={}):
        return x[0]
    
    def output(t, x, u, params={}):
        return np.clip(u[0], -params.get('gain')(t) * params.get('saturation_limits'), params.get('gain')(t) * params.get('saturation_limits'))
    
    return ct.NonlinearIOSystem(
        update, output, inputs=['u'], outputs=['y'], states=1, params=params
    )
