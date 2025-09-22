"""

build time varying DVPP

"""

import control as ct
import numpy as np
import matplotlib.pyplot as plt

from src.get_controllers import get_pi_controller

def build_first_order_time_varying_plant(tau, K, dc_gain) -> ct.NonlinearIOSystem:
    """
    
    build first order system with time varying dc gain

    dc_gain must must be mapping from t -> dc gain value
    
    """
    params = {'tau': tau, 'K': K, 'dc_gain': dc_gain}

    def update(t, x, u, params={}):
        x0 = - 1/params['tau'] * x[0] + u[0]
        return x0
    
    def output(t, x, u, params={}):
        return 1/params['tau'] * x[0] * params['dc_gain'][t]  # time varying dc gain
    
    G = ct.NonlinearIOSystem(
        update, output, inputs=['u'], outputs=['y'], states=1, params=params
    )
    return G


def DVPP_2_devices(lpf_devices, bpf_devices, hpf_devices,
                   pi_params, 
                   dc_gains=None,
                   tau_c=0,
                   STATIC_PF=False,
                   Gs_diff={}):
    mks = {}
    Gs_sum = ct.tf([0], [1])
    if len(lpf_devices)==0:
        if len(bpf_devices)!=0:
            # make one bpf device to an lpf device
            bpf_name, bpf_g = list(bpf_devices.items())[0]
            lpf_devices[bpf_name] = bpf_g
            del bpf_devices[bpf_name]
        elif len(hpf_devices)!=0:
            # make one hpf device to an lpf device
            hpf_name, hpf_g = list(hpf_devices.items())[0]
            lpf_devices[hpf_name] = hpf_g
            del hpf_devices[hpf_name]
        else:
            print(f'BPF devices: {bpf_devices}, HPF devices: {hpf_devices}')
            raise ValueError('At least one LPF device is needed')
    # assume constant and same capacity, thus theta_i is:
    if not STATIC_PF:
        theta_i = 1 / len(lpf_devices)
        for name, g in lpf_devices.items():
            if name in Gs_diff: g = Gs_diff[name]  # take other tranfer function if specified
            mks[name] = theta_i * g  # Define steady-state ADPFs as LPFs
            Gs_sum += theta_i * g
        for name, g in bpf_devices.items():
            if name in Gs_diff: g = Gs_diff[name]  # take other tranfer function if specified
            mks[name] = g * (g - Gs_sum)   # Fix intermediate ADPFs as BPFs
            Gs_sum += g * (g - Gs_sum)
        for name, g in hpf_devices.items():
            mks[name] = ct.tf([1], [tau_c, 1]) - Gs_sum   # Fix fastest device’s ADPF as HPF
    else:
        # every device is treated equally
        theta_i = 1 / (len(lpf_devices) + len(bpf_devices) + len(hpf_devices))
        for name, g in lpf_devices.items():
            mks[name] = ct.tf([theta_i], [1]) 
        for name, g in bpf_devices.items():
            mks[name] = ct.tf([theta_i], [1]) 
        for name, g in hpf_devices.items():
            mks[name] = ct.tf([theta_i], [1])  

    # next, build the system assuming 2 devices
    name1, name2 = [k for k in lpf_devices.keys()] + [k for k in bpf_devices.keys()] + [k for k in hpf_devices.keys()]
    all_devices = [G for G in lpf_devices.values()] + [G for G in bpf_devices.values()] + [G for G in hpf_devices.values()]
    G1, G2 = all_devices[0], all_devices[1] 
    # each plant has their own desired transfer function, namely mks[name] from y -> y_i
    T1_des = mks[name1]
    T1_des.name = f'T_des1'
    T2_des = mks[name2]
    T2_des.name = f'T_des2'
    T1_des.input_labels = ['yref']
    T1_des.output_labels = ['ref_y1']
    T2_des.input_labels = ['yref']
    T2_des.output_labels = ['ref_y2']
    # error = ct.summing_junction(['yref', '-y'], 'e')  # error signal
    error1 = ct.summing_junction(['ref_y1', '-y1'], 'e1', name='err1')  # error signal
    error2 = ct.summing_junction(['ref_y2', '-y2'], 'e2', name='err2')  # error signal

    # PI1, PI2 = PIs[name1], PIs[name2]  # get PI controller
    PI1, PI2 = get_pi_controller(params=pi_params[name1]), get_pi_controller(params=pi_params[name2])
    PI1.output_labels = ['u1']
    PI1.input_labels = ['e1']
    PI1.name = f'PI_1'

    PI2.output_labels = ['u2']
    PI2.input_labels = ['e2']
    PI2.name = f'PI_2'

    # rename for new closed loop
    G1.input_labels = ['u1']
    G1.output_labels = ['y1']
    G1.name = 'G1'
    G2.input_labels = ['u2']
    G2.output_labels = ['y2']
    G2.name = 'G2'

    # create new total output
    y_total = ct.summing_junction(['y1', 'y2'], 'y', name='y_total')  # create new total output

    # closed_loop_agg = ct.interconnect([T1_des, T2_des, error1, error2, PI1, PI2, G1, G2, y_total], inputs=['yref'], outputs=['y', 'y1', 'y2', 'ref_y1', 'ref_y2', 'u1', 'u2'],
    #                                   connections=[
    #                                     #   ['y_total.y', 'T_des1.yref'],
    #                                     #   ['y_total.y', 'T_des2.yref'],
    #                                     ['err1.ref_y1', 'T_des1.ref_y1'],
    #                                     ['err2.ref_y2', 'T_des2.ref_y2'],
    #                                     ['PI_1.e1', 'err1.e1'],
    #                                     ['PI_2.e2', 'err2.e2'],
    #                                     ['G1.u1', 'PI_1.u1'],
    #                                     ['G2.u2', 'PI_2.u2'],
    #                                     ['y_total.y1', 'G1.y1'],
    #                                     ['y_total.y2', 'G2.y2'],
    #                                     ['err1.y1', 'G1.y1'],
    #                                     ['err2.y2', 'G2.y2']
    #                                   ])
    closed_loop_agg = ct.interconnect([T1_des, T2_des, error1, error2, PI1, PI2, G1, G2, y_total], inputs=['yref'], outputs=['y', 'y1', 'y2',  'ref_y1', 'ref_y2', 'u1', 'u2'])
    name_agg = f'{name1} + {name2}'

    return closed_loop_agg, name_agg
