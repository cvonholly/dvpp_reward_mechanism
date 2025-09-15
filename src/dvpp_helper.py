import control as ct
import numpy as np
import matplotlib.pyplot as plt

from src.get_controllers import get_pi_controller


def get_DVPP(IO_dict,
            pi_params,
            Gs_diff,
            vref, min_hard_constrains,
            title='', tlim=(0, 60),
            scales_hard_constrains=np.array([]),
            tol=1e-3,
            save_path='pics/rewards',
            print_total_energy=False,
            get_peak_power=False,
            save_plots=True,
            tau_c=0.001,
            STATIC_PF=False,
            min_service_rating=0.1,
            n_points=1000,
            price=1):
    """
    IO_dict: dict of IO systems with entries: 
        {(name): (ct.tf(...), device_type, rating)} 
            where ct.tf(...) is the transfer function of the device,
            where device_type is 'lpf', 'bpf' or 'hpf'
            where rating is the power rating in MW
    pi_params: dict of pi parameters (kp, ki)
    Gs_diff: first order transfer function for algorithm, if empty use IO_dict

    input_service_max: maximum service curve a.k.a. reference
    curve_service_min: minimum service curve a.k.a. hard constraint to be upheld
    scales_hard_constrains: values for scaling hard constraints
    T_MAX: time horizon of the service in seconds
    save_path: for the plots
    price: price of the service in EUR/MW
    """
    # scale reference and hard constraints to service rating minus tollerance
    lpf_devices = {k: v[0] for k, v in IO_dict.items() if v[1] == 'lpf'}
    bpf_devices = {k: v[0] for k, v in IO_dict.items() if v[1] == 'bpf'}
    hpf_devices = {k: v[0] for k, v in IO_dict.items() if v[1] == 'hpf'}

    mks = {}
    Gs_sum = ct.tf([0], [1])   # set sum to zero transfer function
    if len(lpf_devices)==0 and len(bpf_devices)!=0:   
        # make one bpf device to an lpf device
        bpf_name, bpf_g = list(bpf_devices.items())[0]
        lpf_devices[bpf_name] = bpf_g
        # also add to IO_dict
        IO_dict[bpf_name] = (bpf_g, 'lpf', IO_dict[bpf_name][2])
        del bpf_devices[bpf_name]
    if len(lpf_devices)==0 and len(hpf_devices)!=0:
        # make one hpf device to an lpf device
        hpf_name, hpf_g = list(hpf_devices.items())[0]
        lpf_devices[hpf_name] = hpf_g
        # also add to IO_dict
        IO_dict[hpf_name] = (hpf_g, 'lpf', IO_dict[hpf_name][2])
        del hpf_devices[hpf_name]
    if len(lpf_devices)==1 and len(hpf_devices)==0:
        # make one bpf device to an hpf device
        bpf_name, bpf_g = list(bpf_devices.items())[0]
        hpf_devices[bpf_name] = bpf_g
        # also add to IO_dict
        IO_dict[bpf_name] = (bpf_g, 'hpf', IO_dict[bpf_name][2])
        del bpf_devices[bpf_name]

    # print devicees and type
    print(f'LPF devices: {lpf_devices}, BPF devices: {bpf_devices}, HPF devices: {hpf_devices}')
    
    # set devices and rating
    n_devices = len(lpf_devices) + len(bpf_devices) + len(hpf_devices)
    sum_service_rating = sum([v[2] for v in IO_dict.values() if v[1] == 'lpf'])

    if not STATIC_PF:
        theta_i = 1 / len(lpf_devices)
        for name, g in lpf_devices.items():
            if name in Gs_diff: g = Gs_diff[name]  # take other tranfer function if specified
            theta_i = IO_dict[name][2] / sum_service_rating
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
    
    # next, simulate devices individually
    names = [k for k in lpf_devices.keys()] + [k for k in bpf_devices.keys()] + [k for k in hpf_devices.keys()]
    name_agg = ' + '.join(names)
    all_devices = [G for G in lpf_devices.values()] + [G for G in bpf_devices.values()] + [G for G in hpf_devices.values()]
    # closed_loops = {}
    responses = {}

    t = np.linspace(tlim[0], tlim[1], n_points)
    x0 = [0, 0]
    service_rating = max(sum_service_rating, min_service_rating / scales_hard_constrains[0])   # 0.1 MW is min service rating
    vref = service_rating * vref   # sacel by rating
    for k, name, G in zip(range(n_devices), names, all_devices):
        # each plant has their own desired transfer function, namely mks[name] from y -> y_i
        T_des = mks[name]
        T_des.name = f'T_des_{name}'
        T_des.input_labels = ['yref']
        T_des.output_labels = [f'ref_y{k+1}']
        error = ct.summing_junction([f'ref_y{k+1}', f'-y{k+1}'], f'e{k+1}', name=f'err_{name}')  # error signal

        PI = get_pi_controller(params=pi_params[name])  # get PI controller
        PI.input_labels = [f'e{k+1}']
        PI.output_labels = [f'u{k+1}']        
        PI.name = f'PI_{name}'
        # set plant inputs and outputs
        G.input_labels = [f'u{k+1}']
        G.output_labels = [f'y{k+1}']
        G.name = f'G_{name}'

        closed_loop = ct.interconnect([T_des, error, PI, G], inputs=['yref'], outputs=[f'y{k+1}', f'ref_y{k+1}', f'u{k+1}'], name=f'closed_loop_{name}')
                                    #   connections=[
                                    #     [f'err_{name}.ref_y{k+1}', f'T_des_{name}.ref_y{k+1}'],
                                    #     [f'PI_{name}.e{k+1}', f'err_{name}.e{k+1}'],
                                    #     [f'G_{name}.u{k+1}', f'PI_{name}.u{k+1}'],
                                    #     [f'err_{name}.y{k+1}', f'G_{name}.y{k+1}']])

        response_i = ct.input_output_response(closed_loop, t, vref, x0,
                                        solve_ivp_method='LSODA')
        responses[name] = response_i
    
    # get total system ouput
    plant_output = np.sum([responses[n].outputs[0] for n in names], axis=0) if responses[names[0]].outputs.ndim > 1 else np.sum([responses[n].outputs for n in names], axis=0)

    # check if unit fulfills test
    reward = -3 * service_rating * price * scales_hard_constrains[0]  # penalty if not fulfilling requirements
    new_hard_constraints = service_rating * min_hard_constrains * scales_hard_constrains[0]
    for scale in scales_hard_constrains:
        diff = tol + plant_output - service_rating * min_hard_constrains * scale
        fulfill_requirements = np.all(diff >= 0)
        if fulfill_requirements:
            reward = service_rating * scale * price
            new_hard_constraints = service_rating * min_hard_constrains * scale
        else:
            # failed the test
            break

    energy_dict = {}  # dictionary for total energy
    peak_powerd_dict = {}   # dictionary for peak power

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, vref, 'b--', label='Reference', linewidth=2)
    
    # plot star at hard constraints
    plt.plot(t, new_hard_constraints, '*',  color='red', markersize=1, label='Min Hard Constraint')
    plt.fill_between(t, 0 , new_hard_constraints, color='red', alpha=0.1)

    plt.subplot(2, 1, 1)
    # first, plot total ouput
    plt.plot(t, plant_output, linewidth=2, label=f'{name_agg} total output')

    # plot every device output and input
    for name in names:
        plt.subplot(2, 1, 1)
        plt.plot(t, responses[name].outputs[0], linewidth=1.5, label=f'{name} at {IO_dict[name][2]:.1f}')
        # plot reference in same color but dimmer
        color = plt.gca().lines[-1].get_color()
        plt.plot(t, responses[name].outputs[1], '--', linewidth=1.5, label=f'{name} reference', color=color, alpha=0.5)

        plt.subplot(2, 1, 2)
        plt.plot(t, responses[name].outputs[-1], '--', label=f'{name} PI output', color=color)

        if print_total_energy:
            energy = np.trapz(responses[name].outputs[0], x=t)
            print(f'Total Energy for {name}: {energy:.2f}')
            energy_dict[name] = round(float(energy), 2)
        if get_peak_power:
            peak_power = np.max(responses[name].outputs[0])
            print(f'Peak Power for {name}: {peak_power:.2f}')
            peak_powerd_dict[name] = round(float(peak_power), 2)

    print('========================================')

    plt.subplot(2, 1, 1)
    plt.legend(loc='upper right')
    final_title = title + f', Reward: {reward:.2f}€, at {service_rating:.1f}MW'
    plt.title(final_title)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Output')
    plt.xlim(tlim)

    plt.subplot(2, 1, 2)
    plt.xlabel('Time [s]')
    plt.ylabel('U')
    plt.xlim(tlim)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    if save_plots:
        plt.savefig(f'{save_path}/{title.replace(" ", "_").replace(".", "_")}.png')

    return reward, energy_dict, peak_powerd_dict



def DVPP_2_devices(lpf_devices, bpf_devices, hpf_devices,
                   pi_params, tau_c=0,
                   STATIC_PF=False,
                   Gs_diff={}):
    """
    devices: dict with name -> transfer function    
    """
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

# do same for 3 devices
def DVPP_3_devices(lpf_devices, bpf_devices, hpf_devices,
                   pi_params, tau_c=0,
                   STATIC_PF=False,
                   Gs_diff={}):
    """
    assume 3 devices are connected in parallel
    """
    mks = {}
    Gs_sum = ct.tf([0], [1])
    if not STATIC_PF:
        # construct ADPFs
        # assume constant and same capacity, thus theta_i is:
        theta_i = 1 / len(lpf_devices)
        for name, g in lpf_devices.items():
            if name in Gs_diff: g = Gs_diff[name]
            mks[name] = theta_i * g  # Define steady-state ADPFs as LPFs
            Gs_sum += theta_i * g
        for name, g in bpf_devices.items():
            if name in Gs_diff: g = Gs_diff[name]
            mks[name] = g * (g - Gs_sum)   # Fix intermediate ADPFs as BPFs
            Gs_sum += theta_i * g
        for name, g in hpf_devices.items():
            mks[name] = ct.tf([1], [tau_c, 1]) - Gs_sum   # Fix fastest device’s ADPF as HPF
    else:
        # every device is treated equally
        theta_i = len(lpf_devices) + len(bpf_devices) + len(hpf_devices)
        for name, g in lpf_devices.items():
            mks[name] = ct.tf([1 / theta_i], [1])
        for name, g in bpf_devices.items():
            mks[name] = ct.tf([1 / theta_i], [1])
        for name, g in hpf_devices.items():
            mks[name] = ct.tf([1 / theta_i], [1])

    # next, build the system assuming 3 devices
    name1, name2, name3 = [k for k in lpf_devices.keys()] + [k for k in bpf_devices.keys()] + [k for k in hpf_devices.keys()]
    all_devices = [G for G in lpf_devices.values()] + [G for G in bpf_devices.values()] + [G for G in hpf_devices.values()]
    G1, G2, G3 = all_devices[0], all_devices[1], all_devices[2]
    # rename for new closed loop
    G1.output_labels = ['y1']
    G1.name = name1
    G1.input_labels = ['u1']
    G2.output_labels = ['y2']
    G2.name = name2
    G2.input_labels = ['u2']
    G3.output_labels = ['y3']
    G3.name = name3
    G3.input_labels = ['u3']
    # get PI controller
    PI1, PI2, PI3 = get_pi_controller(params=pi_params[name1]), get_pi_controller(params=pi_params[name2]), get_pi_controller(params=pi_params[name3])  # get PI controller
    PI1.output_labels = ['u1']
    PI1.input_labels = ['e1']
    PI1.name = f'PI_{name1}'
    PI2.output_labels = ['u2']
    PI2.input_labels = ['e2']
    PI2.name = f'PI_{name2}'
    PI3.output_labels = ['u3']
    PI3.input_labels = ['e3']
    PI3.name = f'PI_{name3}'
    # get error signals
    error1 = ct.summing_junction(['ref_y1', '-y1'], 'e1')  # error signal
    error2 = ct.summing_junction(['ref_y2', '-y2'], 'e2')  # error signal
    error3 = ct.summing_junction(['ref_y3', '-y3'], 'e3')  # error signal
    # each plant has their own desired transfer function, namely mks[name] from y -> y_i
    T1_des = mks[name1]
    T1_des.input_labels = ['yref']
    T1_des.output_labels = ['ref_y1']
    T2_des = mks[name2]
    T2_des.input_labels = ['yref']
    T2_des.output_labels = ['ref_y2']
    T3_des = mks[name3]
    T3_des.input_labels = ['yref']
    T3_des.output_labels = ['ref_y3']

    y_total = ct.summing_junction(['y1', 'y2', 'y3'], 'y')  # create new total output

    closed_loop_agg = ct.interconnect([PI1, PI2, PI3, G1, G2, G3, y_total, error1, error2, error3, T1_des, T2_des, T3_des], inputs=['yref'], outputs=['y','y1', 'y2', 'y3', 'ref_y1', 'ref_y2', 'ref_y3', 'u1', 'u2', 'u3'])
    name_agg = f'{name1} + {name2} + {name3}'

    return closed_loop_agg, name_agg

# do same for 4 devices
def DVPP_4_devices(lpf_devices, bpf_devices, hpf_devices,
                   pi_params, tau_c=0,
                   STATIC_PF=False,
                   Gs_diff={}):
    """
    assume 4 devices are connected in parallel
    """
    # todo
    mks = {}
    Gs_sum = ct.tf([0], [1])
    if not STATIC_PF:
        # assume constant and same capacity, thus theta_i is:
        theta_i = 1 / len(lpf_devices)
        for name, g in lpf_devices.items():
            if name in Gs_diff: g = Gs_diff[name]
            mks[name] = theta_i * g  # Define steady-state ADPFs as LPFs
            Gs_sum += theta_i * g
        for name, g in bpf_devices.items():
            if name in Gs_diff: g = Gs_diff[name]
            mks[name] = g * (g - Gs_sum)   # Fix intermediate ADPFs as BPFs
            Gs_sum += theta_i * g
        for name, g in hpf_devices.items():
            mks[name] = ct.tf([1], [tau_c, 1]) - Gs_sum   # Fix fastest device’s ADPF as HPF
    else:
        # every device is treated equally
        theta_i = len(lpf_devices) + len(bpf_devices) + len(hpf_devices)
        for name, g in lpf_devices.items():
            mks[name] = 1 / theta_i
        for name, g in bpf_devices.items():
            mks[name] = 1 / theta_i
        for name, g in hpf_devices.items():
            mks[name] = 1 / theta_i

    # next, build the system assuming 4 devices
    # name1, name2, name3, name4 = list(mks.keys())[:4]
    # G1, G2, G3, G4 = Gs[name1], Gs[name2], Gs[name3], Gs[name4]
    name1, name2, name3, name4 = [k for k in lpf_devices.keys()] + [k for k in bpf_devices.keys()] + [k for k in hpf_devices.keys()]
    all_devices = [G for G in lpf_devices.values()] + [G for G in bpf_devices.values()] + [G for G in hpf_devices.values()]
    G1, G2, G3, G4 = all_devices[0], all_devices[1], all_devices[2], all_devices[3]
    # rename for new closed loop
    G1.output_labels = ['y1']
    G1.name = name1
    G1.input_labels = ['u1']
    G2.output_labels = ['y2']
    G2.name = name2
    G2.input_labels = ['u2']
    G3.output_labels = ['y3']
    G3.name = name3
    G3.input_labels = ['u3']
    G4.output_labels = ['y4']
    G4.name = name4
    G4.input_labels = ['u4']
    # get PI controller
    PI1, PI2, PI3, PI4 = get_pi_controller(params=pi_params[name1]), get_pi_controller(params=pi_params[name2]), get_pi_controller(params=pi_params[name3]), get_pi_controller(params=pi_params[name4])  # get PI controller
    PI1.output_labels = ['u1']
    PI1.input_labels = ['e1']
    PI1.name = f'PI_{name1}'
    PI2.output_labels = ['u2']
    PI2.input_labels = ['e2']
    PI2.name = f'PI_{name2}'
    PI3.output_labels = ['u3']
    PI3.input_labels = ['e3']
    PI3.name = f'PI_{name3}'
    PI4.output_labels = ['u4']
    PI4.input_labels = ['e4']
    PI4.name = f'PI_{name4}'
    # get error signals
    error1 = ct.summing_junction(['ref_y1', '-y1'], 'e1')  # error signal
    error2 = ct.summing_junction(['ref_y2', '-y2'], 'e2')  # error signal
    error3 = ct.summing_junction(['ref_y3', '-y3'], 'e3')  # error signal
    error4 = ct.summing_junction(['y4ref', '-y4'], 'e4')  # error signal
    # each plant has their own desired transfer function, namely mks[name] from y -> y_i
    T1_des = mks[name1]
    T1_des.input_labels = ['yref']
    T1_des.output_labels = ['ref_y1']
    T2_des = mks[name2]
    T2_des.input_labels = ['yref']
    T2_des.output_labels = ['ref_y2']
    T3_des = mks[name3]
    T3_des.input_labels = ['yref']
    T3_des.output_labels = ['ref_y3']
    T4_des = mks[name4]
    T4_des.input_labels = ['yref']
    T4_des.output_labels = ['y4ref']

    y_total = ct.summing_junction(['y1', 'y2', 'y3', 'y4'], 'y')  # create new total output

    closed_loop_agg = ct.interconnect([PI1, PI2, PI3, PI4, G1, G2, G3, G4, y_total, error1, error2, error3, error4, T1_des, T2_des, T3_des, T4_des], inputs=['yref'], outputs=['y','y1', 'y2', 'y3', 'y4', 'u1', 'u2', 'u3', 'u4'])
    name_agg = f'{name1} + {name2} + {name3} + {name4}'

    return closed_loop_agg, name_agg
