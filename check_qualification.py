import matplotlib.pyplot as plt
import numpy as np
import control as ct
import itertools
import math as mt


def check_control_system(Gs: list, vref, names: list, title='', tlim=(0, 60),
                         print_total_energy=False,
                         plot_hard_constrains=np.array([]),
                         tol=1e-1):
    """
    Check the performance of the control system.
    Gs: list of interconnected system
    vref: reference to follow
    names: list of names for each system
    title: title of plot
    tlim: time limits for the plot (1000 points fixed)
    print_total_energy: print total energy of the system
    plot_hard_constrains: plot hard constraints of reference as np.ndarray
    tol: tolerance for checking system performance
    """
    t = np.linspace(tlim[0], tlim[1], 1000)   
    # Initial conditions: [PI integrator state, system state]
    x0 = [0, 0]

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, vref, 'b--', label='Reference', linewidth=2)
    
    # plot star at hard constraints
    if plot_hard_constrains.size != 0:
        plt.plot(t, plot_hard_constrains, '*',  color='red', markersize=1, label='Min Hard Constraint')
        plt.fill_between(t, 0 , plot_hard_constrains, color='red', alpha=0.1)

    j = 0
    name = names[j]
    qual_dict = {}  # dictionary for satisfying requirements
    energy_dict = {}  # dictionary for total energy

    for closed_loop in Gs:
        name = names[j]
        # Simulate the closed-loop response
        response = ct.input_output_response(closed_loop, t, vref, x0,
                                            solve_ivp_method='LSODA')
    
        # Extract outputs and states properly
        UNIT_NAMES, k = name.strip(' ').split('+'), -1
        labels = response.output_labels
        plant_output = response.outputs[0] if response.outputs.ndim > 1 else response.outputs

        plt.subplot(2, 1, 1)
        for l, out in zip(labels, response.outputs):
            if l.startswith('y'):
                lab = f'{name} {l}' if k==-1 else f'{UNIT_NAMES[k]} {l}'
                plt.plot(t, out, linewidth=1.5, label=lab)
                if print_total_energy:
                    energy = np.trapz(out, x=t)
                    print(f'Total Energy for {lab}: {energy:.2f}')
                    energy_dict[lab] = round(float(energy), 2)
                k += 1

        # second plot
        plt.subplot(2, 1, 2)
        
        k = 0
        for l, out in zip(labels, response.outputs):
            if l.startswith('u'):
                lab = f'{name} {l}' if len(UNIT_NAMES) == 1 else f'{UNIT_NAMES[k]} PI {l}'
                plt.plot(t, out, '--', label=lab)
                k += 1

        # check if unit fulfills test
        if plot_hard_constrains.size != 0:
            diff = tol + plant_output - plot_hard_constrains

            fulfill_requirements = np.all(diff >= 0)

            print(f"{name} qualification requirements: {fulfill_requirements}")
            qual_dict[name] = fulfill_requirements

        j += 1
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

    plt.savefig('pics/{}.png'.format(title.replace(' ', '_')))
    # plt.show(block=False)
    # plt.pause(0.1)

    if print_total_energy:
        # also return total energy of devices delivered
        return qual_dict, energy_dict

    return qual_dict

def DVPP_2_devices(lpf_devices, bpf_devices, hpf_devices,
                   Gs, PIs, tau_c=0):
    mks = {}
    Gs_sum = ct.tf([0], [1])
    # assume static and same capacity, thus theta_i is:
    theta_i = 1 / len(lpf_devices)
    for name, g in lpf_devices.items():
        mks[name] = theta_i * g  # Define steady-state ADPFs as LPFs
        Gs_sum += theta_i * g
    for name, g in bpf_devices.items():
        mks[name] = g * (g - Gs_sum)   # Fix intermediate ADPFs as BPFs
        Gs_sum += theta_i * g
    for name, g in hpf_devices.items():
        mks[name] = ct.tf([1], [tau_c, 1]) - Gs_sum   # Fix fastest device’s ADPF as HPF
    # return mks, Gs_sum

    # next, build the system assuming 2 devices
    name1, name2 = list(mks.keys())[:2]
    G1, G2 = Gs[name1], Gs[name2]
    # rename for new closed loop
    G1.output_labels = ['y1']
    G1.name = name1
    G2.output_labels = ['y2']
    G2.name = name2
    # error = ct.summing_junction(['yref', '-y'], 'e')  # error signal
    error1 = ct.summing_junction(['y1ref', '-y1'], 'e1')  # error signal
    error2 = ct.summing_junction(['y2ref', '-y2'], 'e2')  # error signal
    # each plant has their own desired transfer function, namely mks[name] from y -> y_i
    T1_des = ct.tf(mks[name1].num, mks[name1].den, inputs=['yref'], outputs=['y1ref'])
    T2_des = ct.tf(mks[name2].num, mks[name2].den, inputs=['yref'], outputs=['y2ref'])

    y_total = ct.summing_junction(['y1', 'y2'], 'y')  # create new total output

    PI1, PI2 = PIs[name1], PIs[name2]  # get PI controller
    PI1.output_labels = ['u1']
    PI1.input_labels = ['e1']
    PI1.name = f'PI_{name1}'
    PI2.output_labels = ['u2']
    PI2.input_labels = ['e2']
    PI2.name = f'PI_{name2}'
    G1.input_labels = ['u1']
    G2.input_labels = ['u2']

    closed_loop_agg = ct.interconnect([PI1, PI2, G1, G2, y_total, error1, error2, T1_des, T2_des], inputs=['yref'], outputs=['y', 'y1', 'y2', 'u1', 'u2'])
    name_agg = f'{name1} + {name2}'

    return closed_loop_agg, name_agg

# do same for 3 devices
def DVPP_3_devices(lpf_devices, bpf_devices, hpf_devices,
                   Gs, PIs, tau_c=0):
    """
    assume 3 devices are connected in parallel
    """
    mks = {}
    Gs_sum = ct.tf([0], [1])
    # assume static and same capacity, thus theta_i is:
    theta_i = 1 / len(lpf_devices)
    for name, g in lpf_devices.items():
        mks[name] = theta_i * g  # Define steady-state ADPFs as LPFs
        Gs_sum += theta_i * g
    for name, g in bpf_devices.items():
        mks[name] = g * (g - Gs_sum)   # Fix intermediate ADPFs as BPFs
        Gs_sum += theta_i * g
    for name, g in hpf_devices.items():
        mks[name] = ct.tf([1], [tau_c, 1]) - Gs_sum   # Fix fastest device’s ADPF as HPF
    # return mks, Gs_sum

    # next, build the system assuming 3 devices
    name1, name2, name3 = list(mks.keys())[:3]
    G1, G2, G3 = Gs[name1], Gs[name2], Gs[name3]
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
    PI1, PI2, PI3 = PIs[name1], PIs[name2], PIs[name3]  # get PI controller
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
    error1 = ct.summing_junction(['y1ref', '-y1'], 'e1')  # error signal
    error2 = ct.summing_junction(['y2ref', '-y2'], 'e2')  # error signal
    error3 = ct.summing_junction(['y3ref', '-y3'], 'e3')  # error signal
    # each plant has their own desired transfer function, namely mks[name] from y -> y_i
    T1_des = ct.tf(mks[name1].num, mks[name1].den, inputs=['yref'], outputs=['y1ref'])
    T2_des = ct.tf(mks[name2].num, mks[name2].den, inputs=['yref'], outputs=['y2ref'])
    T3_des = ct.tf(mks[name3].num, mks[name3].den, inputs=['yref'], outputs=['y3ref'])

    y_total = ct.summing_junction(['y1', 'y2', 'y3'], 'y')  # create new total output

    closed_loop_agg = ct.interconnect([PI1, PI2, PI3, G1, G2, G3, y_total, error1, error2, error3, T1_des, T2_des, T3_des], inputs=['yref'], outputs=['y','y1', 'y2', 'y3', 'u1', 'u2', 'u3'])
    name_agg = f'{name1} + {name2} + {name3}'

    return closed_loop_agg, name_agg

# shapely value of coalition

def get_shapely_value(v: dict, players: list) -> dict:
    """
    Shapley value calculation for a coalition game.

    v: Characteristic function of the game, must be defined for EVERY coalition
        {dict -> float}
    players: list of players
    """
    shapley_values = {p: 0 for p in players}  # empty coaltion: 0 value
    n = len(players)
    # iterate over all coalitions
    for c_size in range(1, len(players) + 1):  # iterate over coalition sizes
        for subset in itertools.combinations(players, c_size):
            k = len(subset)
            for p in subset:  # iterate over player in subset
                marginal_contribution = v.get(subset, 0) - v.get(tuple(c for c in subset if c != p), 0)
                shapley_values[p] += marginal_contribution * mt.factorial(k - 1) * mt.factorial(n - k) / mt.factorial(n)

    return shapley_values