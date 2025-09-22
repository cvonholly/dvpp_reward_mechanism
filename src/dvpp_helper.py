import control as ct
import numpy as np
import matplotlib.pyplot as plt

from src.get_controllers import get_pi_controller

from src.get_device_systems import get_adaptive_dc_sys, get_adaptive_saturation_sys, get_ADPF, get_static_pf_varying_ref


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
            price=1,
            save_pics=True,
            adaptive_func={}):
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
    if sum([v[2] for v in IO_dict.values() if v[1] == 'lpf']) < 1e-3:
        # LPF device has zero capacity, make hpf device to lpf
        hpf_name, hpf_g = list(hpf_devices.items())[0]
        lpf_devices[hpf_name] = hpf_g
        # also add to IO_dict
        IO_dict[hpf_name] = (hpf_g, 'lpf', IO_dict[hpf_name][2])
        del hpf_devices[hpf_name]

    # print devices and type
    print(f'LPF devices: {lpf_devices.keys()}, BPF devices: {bpf_devices.keys()}, HPF devices: {hpf_devices.keys()}')

    # set devices and rating
    n_devices = len(lpf_devices) + len(bpf_devices) + len(hpf_devices)
    sum_service_rating = sum([v[2] for v in IO_dict.values() if v[1] == 'lpf'])

    if not STATIC_PF and adaptive_func=={}:
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
    elif not STATIC_PF and adaptive_func!={}:
        mks = get_ADPF(lpf_devices, bpf_devices, hpf_devices, 
                       IO_dict, sum_service_rating, Gs_diff, adaptive_func, T_END=tlim[1])
    elif STATIC_PF and adaptive_func!={}:
        mks = get_static_pf_varying_ref(IO_dict, adaptive_func)
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
    vref = service_rating * vref   # scale by rating

    for k, name, G in zip(range(n_devices), names, all_devices):
        print('Running for device:', name)
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

        # # adaptive function part
        # if adaptive_func!={}:
        #     if name in lpf_devices:
        #         T_adaptive = get_adaptive_dc_sys(params={'gain': adaptive_func[name]})
        #         sat_adaptive = get_adaptive_saturation_sys(params={'gain': adaptive_func[name], 'saturation_limits': IO_dict[name][2]})
        #         T_des.output_labels = [f'adaptive_ref_y{k+1}']
        #         T_adaptive.input_labels = [f'adaptive_ref_y{k+1}']
        #         T_adaptive.output_labels = [f'ref_y{k+1}']

        #         PI.output_labels = [f'adaptive_u{k+1}']
        #         sat_adaptive.input_labels = [f'adaptive_u{k+1}']
        #         sat_adaptive.output_labels = [f'u{k+1}']

        #         closed_loop = ct.interconnect([T_des, T_adaptive, error, PI, sat_adaptive, G], inputs=['yref'], outputs=[f'y{k+1}', f'ref_y{k+1}', f'u{k+1}'], name=f'closed_loop_{name}')
        #     if name in hpf_devices:
        #         # adjust to track full reference
        #         T_adaptive = get_adaptive_dc_sys(params={'gain': adaptive_func[name]})
        #         T_des.output_labels = [f'adaptive_ref_y{k+1}']
        #         T_adaptive.input_labels = [f'adaptive_ref_y{k+1}']
        #         T_adaptive.output_labels = [f'ref_y{k+1}']

        response_i = ct.input_output_response(closed_loop, t, vref, x0,
                                        solve_ivp_method='LSODA')
        responses[name] = response_i
    
    # get total system ouput
    plant_output = np.sum([responses[n].outputs[0] for n in names], axis=0) if responses[names[0]].outputs.ndim > 1 else np.sum([responses[n].outputs for n in names], axis=0)

    # check if unit fulfills test
    final_rating = service_rating * scales_hard_constrains[0]
    reward = -3 * price * final_rating  # penalty if not fulfilling requirements
    new_hard_constraints = final_rating * min_hard_constrains
    for scale in scales_hard_constrains:
        diff = tol + plant_output - scale * service_rating * min_hard_constrains
        fulfill_requirements = np.all(diff >= 0)
        if fulfill_requirements:
            reward = service_rating * scale * price
            final_rating = service_rating * scale
            new_hard_constraints = final_rating * min_hard_constrains
        else:
            # failed the test
            break

    if not save_pics:  # do not plot
        return reward, {}, {}

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
    final_title = title + f', Reward: {reward:.2f}€, at {final_rating:.2f}MW'
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

