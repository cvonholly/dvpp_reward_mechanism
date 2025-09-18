import numpy as np
import matplotlib.pyplot as plt
import control as ct


def plot_reward_value(closed_loop: ct.TransferFunction, vref, min_hard_constrains,
                         name: str, title='', tlim=(0, 60),
                         service_rating=1,
                         scales_hard_constrains=np.array([]),
                         tol=1e-3,
                         save_path='pics/rewards',
                         print_total_energy=False,
                         get_peak_power=False,
                         save_plots=True,
                         min_service_rating=.1,
                         price=1):
    """
    gets the maximum the power plant can output and outputs the result


    closed_loop: interconnected system
    vref: reference to follow as nd.array
    min_hard_constrains: hard constraints of reference as np.ndarray
    name: name for the system

    title: title of plot
    tlim: time limits for the plot (1000 points fixed)
    service_rating: rating of the service in MW (default 1 MW)
    scales_hard_constrains: values for scaling hard constraints
    tol: tolerance for checking system performance
    print_total_energy: print total energy of the system
    """
    t = np.linspace(tlim[0], tlim[1], 1000)   
    # Initial conditions: [PI integrator state, system state]
    x0 = [0, 0]

    energy_dict = {}  # dictionary for total energy
    peak_powerd_dict = {}   # dictionary for peak power

    # scale reference and hard constraints to service rating minus tollerance
    service_rating = max(service_rating, min_service_rating / scales_hard_constrains[0])   # 0.1 MW is min service rating
    vref = service_rating * vref   # scale by rating

    # Simulate the closed-loop response
    response = ct.input_output_response(closed_loop, t, vref, x0,
                                        solve_ivp_method='LSODA')
    plant_output = response.outputs[0] if response.outputs.ndim > 1 else response.outputs
    
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

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, vref, 'b--', label='Reference', linewidth=2)
    
    # plot star at hard constraints
    plt.plot(t, new_hard_constraints, '*',  color='red', markersize=1, label='Min Hard Constraint')
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
        elif l.startswith('ref'):
            lab = f'{l}'
            plt.plot(t, out, '--', linewidth=1.5, label=lab)
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
    plt.legend(loc='upper right')
    plt.title(title + f', Reward: {reward:.2f}€, at {final_rating:.2f}MW')
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
        plt.savefig(f'{save_path}/{title.replace(" ", "_")}.png')

    return reward, energy_dict, peak_powerd_dict
