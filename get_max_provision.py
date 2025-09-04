import numpy as np
import matplotlib.pyplot as plt
import control as ct


def get_reward_value(closed_loop: ct.TransferFunction, vref, name: str, title='', tlim=(0, 60),
                         min_hard_constrains=np.array([]),
                         scales_hard_constrains=np.array([]),
                         tol=5e-2,
                         print_total_energy=False,
                         get_peak_power=False):
    """
    gets the maximum the power plant can output and outputs the result


    Gs: list of interconnected system
    vref: reference to follow
    names: list of names for each system
    title: title of plot
    tlim: time limits for the plot (1000 points fixed)
    scales_hard_constrains: values for scaling hard constraints
    min_hard_constrains: plot hard constraints of reference as np.ndarray
    tol: tolerance for checking system performance
    print_total_energy: print total energy of the system
    """
    t = np.linspace(tlim[0], tlim[1], 1000)   
    # Initial conditions: [PI integrator state, system state]
    x0 = [0, 0]

    energy_dict = {}  # dictionary for total energy
    peak_powerd_dict = {}   # dictionary for peak power
    # Simulate the closed-loop response
    response = ct.input_output_response(closed_loop, t, vref, x0,
                                        solve_ivp_method='LSODA')
    plant_output = response.outputs[0] if response.outputs.ndim > 1 else response.outputs
    
    # check if unit fulfills test
    reward = 0
    final_scale = 0
    new_hard_constraints = min_hard_constrains
    diff = tol + plant_output - new_hard_constraints
    fulfill_requirements = np.all(diff >= 0)
    if fulfill_requirements:
        reward, final_scale = 1, 1
        # check if also scales can be met
        for scale in scales_hard_constrains:
            diff = tol + plant_output - min_hard_constrains * scale
            fulfill_requirements = np.all(diff >= 0)
            if fulfill_requirements:
                reward = scale * 1
                new_hard_constraints = min_hard_constrains * scale
                final_scale = scale
            else:
                # failed the test
                break

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, vref, 'b--', label='Reference', linewidth=2)
    
    # plot star at hard constraints
    plt.plot(t, new_hard_constraints, '*',  color='red', markersize=1, label=f'Min Hard Constr at {final_scale}x')
    plt.fill_between(t, 0 , new_hard_constraints, color='red', alpha=0.1)

    # Extract outputs and states properly
    UNIT_NAMES, k = name.strip(' ').split('+'), -1
    labels = response.output_labels

    plt.subplot(2, 1, 1)
    for l, out in zip(labels, response.outputs):
        if l.startswith('y'):
            lab = f'{name} {l}' if k==-1 else f'{UNIT_NAMES[k]} {l}'
            plt.plot(t, out, linewidth=1.5, label=lab)
            if print_total_energy:
                energy = np.trapz(out, x=t)
                print(f'Total Energy for {lab}: {energy:.2f}')
                energy_dict[lab] = round(float(energy), 2)
            if get_peak_power:
                peak_power = np.max(out)
                print(f'Peak Power for {lab}: {peak_power:.2f}')
                peak_powerd_dict[lab] = round(float(peak_power), 2)
            k += 1

    # second plot
    plt.subplot(2, 1, 2)
    
    k = 0
    for l, out in zip(labels, response.outputs):
        if l.startswith('u'):
            lab = f'{name} {l}' if len(UNIT_NAMES) == 1 else f'{UNIT_NAMES[k]} PI {l}'
            plt.plot(t, out, '--', label=lab)
            k += 1

    print('========================================')

    plt.subplot(2, 1, 1)
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Output')
    plt.xlim(tlim)

    plt.subplot(2, 1, 2)
    plt.xlabel('Time [s]')
    plt.ylabel('U')
    plt.xlim(tlim)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('pics/rewards/{}.png'.format(title.replace(' ', '_')))

    return reward, energy_dict, peak_powerd_dict
