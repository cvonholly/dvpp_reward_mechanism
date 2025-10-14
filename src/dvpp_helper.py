import control as ct
import numpy as np
import matplotlib.pyplot as plt

from src.get_controllers import get_pi_controller, get_pi_controller_adaptive

from src.get_device_systems import get_ADPF #, get_static_pf_varying_ref


def get_DVPP(IO_dict,
            pi_params,
            dpfs,
            vref, min_hard_constrains,
            title='', tlim=(0, 60),
            scales_rating=np.array([]),
            tol=1e-3,
            save_path='pics/rewards',
            print_total_energy=False,
            get_peak_power=False,
            save_plots=True,
            tau_c=1e-4,
            pf_name=False,
            min_service_rating=0.1,
            n_points=1000,
            price=1,
            save_pics=True,
            adaptive_func={},
            sum_service_rating=1.0,
            total_dc_gain=1.0):
    """
    IO_dict: dict of IO systems with entries: 
        {(name): (ct.tf(...), device_type, rating)} 
            where ct.tf(...) is the transfer function of the device,
            where device_type is 'lpf', 'bpf' or 'hpf'
            where rating is the power rating in MW
    pi_params: dict of pi parameters (kp, ki)
    dpfs: dynamic participation factors for each device

    input_service_max: maximum service curve a.k.a. reference
    curve_service_min: minimum service curve a.k.a. hard constraint to be upheld
    scales_rating: values for scaling the reference
    T_MAX: time horizon of the service in seconds
    save_path: for the plots
    price: price of the service in EUR/MW
    """
    # scale reference and hard constraints to service rating minus tollerance
    lpf_devices = {k: v[0] for k, v in IO_dict.items() if v[1] == 'lpf'}
    bpf_devices = {k: v[0] for k, v in IO_dict.items() if v[1] == 'bpf'}
    hpf_devices = {k: v[0] for k, v in IO_dict.items() if v[1] == 'hpf'}

    # sum(mks) = 1 has to be fullfilled
    # this means: we always need at least one HPF device
    mks = {}
    Gs_sum = ct.tf([0], [1])   # set sum to zero transfer function
    if sum([v[2] for v in IO_dict.values() if v[1] == 'lpf']) < 1e-3:
        # LPF devices have near zero capacity
        # ensure there is at least one "HPF" devices such that sum(mks)=1 can be fulfilled
        if len(bpf_devices)>0:
            bpf_name, bpf_g = list(bpf_devices.items())[0]
            hpf_devices[bpf_name] = bpf_g
            # also add to IO_dict
            IO_dict[bpf_name] = (bpf_g, 'hpf', IO_dict[bpf_name][2])
            del bpf_devices[bpf_name]
    elif len(lpf_devices)==1 and len(hpf_devices)==0:
        # make one bpf device to an hpf device
        bpf_name, bpf_g = list(bpf_devices.items())[0]
        hpf_devices[bpf_name] = bpf_g
        # also add to IO_dict
        IO_dict[bpf_name] = (bpf_g, 'hpf', IO_dict[bpf_name][2])
        del bpf_devices[bpf_name]
    elif len(lpf_devices)>1 and len(hpf_devices)==0:
        # make one lpf device to an hpf device
        lpf_name, lpf_g = list(lpf_devices.items())[-1]
        hpf_devices[lpf_name] = lpf_g
        # also add to IO_dict
        IO_dict[lpf_name] = (lpf_g, 'hpf', IO_dict[lpf_name][2])
        del lpf_devices[lpf_name]
    elif len(lpf_devices)==0 and len(bpf_devices)!=0:   
        # make one bpf device to an lpf device (hpf device exists)
        bpf_name, bpf_g = list(bpf_devices.items())[0]
        lpf_devices[bpf_name] = bpf_g
        # also add to IO_dict
        IO_dict[bpf_name] = (bpf_g, 'lpf', IO_dict[bpf_name][2])
        del bpf_devices[bpf_name]

    # print devices and type
    print(f'LPF devices: {lpf_devices.keys()}, BPF devices: {bpf_devices.keys()}, HPF devices: {hpf_devices.keys()}')

    # set devices and rating
    n_devices = len(lpf_devices) + len(bpf_devices) + len(hpf_devices)

    # Dynamic Participation Factors
    if pf_name=='DPF':
        for name, g in lpf_devices.items():
            g = IO_dict[name][2] / total_dc_gain * dpfs[name]  # compute dynamic participation factor
            mks[name] = g  # Define steady-state DPFs as LPFs
            Gs_sum += g
        for name, g in bpf_devices.items():
            g = dpfs[name]  # compute dynamic participation factor
            mks[name] = g * (g - Gs_sum)   # Fix intermediate DPFs as BPFs
            Gs_sum += g * (g - Gs_sum)
        for name, g in hpf_devices.items():
            mks[name] = ct.tf([1], [tau_c, 1]) - Gs_sum   # Fix fastest device’s DPFs as HPF
    # Adaptive Dynamic Participation Factors
    elif pf_name=='ADPF':
        mks = get_ADPF(lpf_devices, bpf_devices, hpf_devices, 
                       IO_dict, total_dc_gain, dpfs, adaptive_func, T_END=tlim[1], tau_c=tau_c)
    # elif STATIC_PF and adaptive_func!={}:
    #     mks = get_static_pf_varying_ref(IO_dict, adaptive_func)
    # todo add option for dynamic but not adaptive participation factors
    else:
        # static participation factors
        # every device is treated equally
        theta_i = 1 / (len(lpf_devices) + len(bpf_devices) + len(hpf_devices))
        for name, g in lpf_devices.items():
            mks[name] = ct.tf([theta_i], [tau_c, 1]) 
        for name, g in bpf_devices.items():
            mks[name] = ct.tf([theta_i], [tau_c, 1]) 
        for name, g in hpf_devices.items():
            mks[name] = ct.tf([theta_i], [tau_c, 1])  
    
    # next, simulate devices individually
    names = [k for k in lpf_devices.keys()] + [k for k in bpf_devices.keys()] + [k for k in hpf_devices.keys()]
    name_agg = ' + '.join(names)
    all_devices = [G for G in lpf_devices.values()] + [G for G in bpf_devices.values()] + [G for G in hpf_devices.values()]
    # closed_loops = {}
    responses = {}

    t = np.linspace(tlim[0], tlim[1], n_points)
    x0 = [0, 0]
    vref = sum_service_rating * vref   # scale by rating

    for k, name, G in zip(range(n_devices), names, all_devices):
        print('Running for device:', name)
        # each plant has their own desired transfer function, namely mks[name] from y -> y_i
        T_des = mks[name]
        T_des.name = f'T_des_{name}'
        T_des.input_labels = ['yref']
        T_des.output_labels = [f'ref_y{k+1}']
        error = ct.summing_junction([f'ref_y{k+1}', f'-y{k+1}'], f'e{k+1}', name=f'err_{name}')  # error signal
        if name=='SC' and 'Hydro' in IO_dict.keys():
            # idea: take hydro error and add it to SC error
            # this way, the SC can help the Hydro to reach the reference faster
            # Hydro was already simulated, so we can use its error signal
            hydro_error = responses['Hydro'].outputs[-1].tolist()
            error = ct.summing_junction([f'ref_y{k+1}', 'hydro_error', f'-y{k+1}'], f'e{k+1}', name=f'err_{name}')  # error signal

        # set PI controller and saturation block, if time-varying (adaptive) use adaptive version
        if name in adaptive_func:
            pi_params[name]['adaptive_func'] = adaptive_func[name]
            PI = get_pi_controller_adaptive(params=pi_params[name])
        else:
            PI = get_pi_controller(params=pi_params[name])  # get PI controller
        PI.input_labels = [f'e{k+1}']
        PI.output_labels = [f'u{k+1}']        
        PI.name = f'PI_{name}'
        # set plant inputs and outputs
        G.input_labels = [f'u{k+1}']
        G.output_labels = [f'y{k+1}']
        G.name = f'G_{name}'

        closed_loop = ct.interconnect([T_des, error, PI, G], inputs=['yref'], outputs=[f'y{k+1}', f'ref_y{k+1}', f'u{k+1}', f'e{k+1}'], name=f'closed_loop_{name}')
        if name=='SC' and 'Hydro' in IO_dict.keys():
            closed_loop = ct.interconnect([T_des, error, PI, G], inputs=['yref', 'hydro_error'], outputs=[f'y{k+1}', f'ref_y{k+1}', f'u{k+1}', f'e{k+1}'], name=f'closed_loop_{name}')                                    #   connections=[
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
        if name=='SC' and 'Hydro' in IO_dict.keys():
            response_i = ct.input_output_response(closed_loop, t, [vref, hydro_error], x0,
                                                  solve_ivp_method='LSODA')
        else:
            response_i = ct.input_output_response(closed_loop, t, vref, x0,
                                        solve_ivp_method='LSODA')
        responses[name] = response_i
    
    # get total system ouput
    plant_output = np.sum([responses[n].outputs[0] for n in names], axis=0) if responses[names[0]].outputs.ndim > 1 else np.sum([responses[n].outputs for n in names], axis=0)

    # check if unit fulfills test
    final_rating = sum_service_rating * scales_rating[0]
    reward = -3 * price * final_rating  # penalty if not fulfilling requirements
    new_hard_constraints = final_rating * min_hard_constrains
    for scale in scales_rating:
        diff = tol + plant_output - scale * sum_service_rating * min_hard_constrains
        fulfill_requirements = np.all(diff >= 0)
        if fulfill_requirements:
            reward = sum_service_rating * scale * price
            final_rating = sum_service_rating * scale
            new_hard_constraints = final_rating * min_hard_constrains
        else:
            # failed the test
            break

    # check if minimum rating is reached
    if final_rating < min_service_rating:
        reward = -3 * price * final_rating  # penalty if not fulfilling requirements
    
    # in this case, we have a minimum rating to fulfill
    if min_service_rating > 0.1:
        if final_rating < min_service_rating:
            reward = -3 * price * min_service_rating  # penalty if not fulfilling requirements
        else:
            reward = final_rating * price

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
        plt.plot(t, responses[name].outputs[0], linewidth=1.5, label=f'{name} at {IO_dict[name][2]:.2f}')
        # plot reference in same color but dimmer
        color = plt.gca().lines[-1].get_color()
        plt.plot(t, responses[name].outputs[1], '--', linewidth=1.5, label=f'{name} reference', color=color, alpha=0.5)

        # plot PI output
        plt.subplot(2, 1, 2)
        plt.plot(t, responses[name].outputs[2], '--', label=f'{name} PI output', color=color)

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
        # save response to file


    return reward, energy_dict, peak_powerd_dict

