import matplotlib.pyplot as plt
import numpy as np
import control as ct
import itertools
import math as mt
import sympy as sp

def sympy_to_tf(T_sym, s=sp.symbols('s')):
    """
    Convert a SymPy rational function T(s) into a python-control TransferFunction.
    """
    num, den = sp.fraction(sp.together(T_sym))   # separate numerator and denominator
    num_poly = sp.Poly(sp.expand(num), s)
    den_poly = sp.Poly(sp.expand(den), s)

    # coefficients in descending powers of s
    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]

    return ct.tf(num_coeffs, den_coeffs)


def check_control_system(Gs: list, vref, names: list, title='', tlim=(0, 60),
                         print_total_energy=False,
                         plot_hard_constrains=np.array([]),
                         tol=1e-1,
                         get_peak_power=False):
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
    peak_powerd_dict = {}   # dictionary for peak power

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

    return qual_dict, energy_dict, peak_powerd_dict

def DVPP_2_devices(lpf_devices, bpf_devices, hpf_devices,
                   Gs, PIs, tau_c=0,
                   STATIC_PF=False):
    mks = {}
    Gs_sum = ct.tf([0], [1])
    if len(lpf_devices)==0:
        # make one hpf device to an lpf device
        hpf_name, hpf_g = list(hpf_devices.items())[0]
        lpf_devices[hpf_name] = hpf_g
        del hpf_devices[hpf_name]
    # assume constant and same capacity, thus theta_i is:
    if not STATIC_PF:
        theta_i = 1 / len(lpf_devices)
        for name, g in lpf_devices.items():
            mks[name] = theta_i * g  # Define steady-state ADPFs as LPFs
            Gs_sum += theta_i * g
        for name, g in bpf_devices.items():
            mks[name] = g * (g - Gs_sum)   # Fix intermediate ADPFs as BPFs
            Gs_sum += theta_i * g
        for name, g in hpf_devices.items():
            mks[name] = ct.tf([1], [tau_c, 1]) - Gs_sum   # Fix fastest device’s ADPF as HPF
    else:
        # every device is treated equally
        theta_i = len(lpf_devices) + len(bpf_devices) + len(hpf_devices)
        for name, g in lpf_devices.items():
            mks[name] = ct.tf([1], [theta_i]) 
        for name, g in bpf_devices.items():
            mks[name] = ct.tf([1], [theta_i]) 
        for name, g in hpf_devices.items():
            mks[name] = ct.tf([1], [theta_i])  

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
    T1_des = mks[name1]
    T2_des = mks[name2]
    T1_des.input_labels = ['yref']
    T1_des.output_labels = ['y1ref']
    T2_des.input_labels = ['yref']
    T2_des.output_labels = ['y2ref']

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
                   Gs, PIs, tau_c=0,
                   STATIC_PF=False):
    """
    assume 3 devices are connected in parallel
    """
    mks = {}
    Gs_sum = ct.tf([0], [1])
    if not STATIC_PF:
        # assume constant and same capacity, thus theta_i is:
        theta_i = 1 / len(lpf_devices)
        for name, g in lpf_devices.items():
            mks[name] = theta_i * g  # Define steady-state ADPFs as LPFs
            Gs_sum += theta_i * g
        for name, g in bpf_devices.items():
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
    T1_des = mks[name1]
    T2_des = mks[name2]
    T1_des.input_labels = ['yref']
    T1_des.output_labels = ['y1ref']
    T2_des.input_labels = ['yref']
    T2_des.output_labels = ['y2ref']
    T3_des = mks[name3]
    T3_des.input_labels = ['yref']
    T3_des.output_labels = ['y3ref']

    y_total = ct.summing_junction(['y1', 'y2', 'y3'], 'y')  # create new total output

    closed_loop_agg = ct.interconnect([PI1, PI2, PI3, G1, G2, G3, y_total, error1, error2, error3, T1_des, T2_des, T3_des], inputs=['yref'], outputs=['y','y1', 'y2', 'y3', 'u1', 'u2', 'u3'])
    name_agg = f'{name1} + {name2} + {name3}'

    return closed_loop_agg, name_agg

# do same for 4 devices
def DVPP_4_devices(lpf_devices, bpf_devices, hpf_devices,
                   Gs, PIs, tau_c=0,
                   STATIC_PF=False):
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
            mks[name] = theta_i * g  # Define steady-state ADPFs as LPFs
            Gs_sum += theta_i * g
        for name, g in bpf_devices.items():
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
    name1, name2, name3, name4 = list(mks.keys())[:4]
    G1, G2, G3, G4 = Gs[name1], Gs[name2], Gs[name3], Gs[name4]
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
    PI1, PI2, PI3, PI4 = PIs[name1], PIs[name2], PIs[name3], PIs[name4]  # get PI controller
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
    error1 = ct.summing_junction(['y1ref', '-y1'], 'e1')  # error signal
    error2 = ct.summing_junction(['y2ref', '-y2'], 'e2')  # error signal
    error3 = ct.summing_junction(['y3ref', '-y3'], 'e3')  # error signal
    error4 = ct.summing_junction(['y4ref', '-y4'], 'e4')  # error signal
    # each plant has their own desired transfer function, namely mks[name] from y -> y_i
    T1_des = mks[name1]
    T1_des.input_labels = ['yref']
    T1_des.output_labels = ['y1ref']
    T2_des = mks[name2]
    T2_des.input_labels = ['yref']
    T2_des.output_labels = ['y2ref']
    T3_des = mks[name3]
    T3_des.input_labels = ['yref']
    T3_des.output_labels = ['y3ref']
    T4_des = mks[name4]
    T4_des.input_labels = ['yref']
    T4_des.output_labels = ['y4ref']

    y_total = ct.summing_junction(['y1', 'y2', 'y3', 'y4'], 'y')  # create new total output

    closed_loop_agg = ct.interconnect([PI1, PI2, PI3, PI4, G1, G2, G3, G4, y_total, error1, error2, error3, error4, T1_des, T2_des, T3_des, T4_des], inputs=['yref'], outputs=['y','y1', 'y2', 'y3', 'y4', 'u1', 'u2', 'u3', 'u4'])
    name_agg = f'{name1} + {name2} + {name3} + {name4}'

    return closed_loop_agg, name_agg

# shapely value of coalition

def get_shapely_value(v: dict, players: list) -> dict:
    """
    Shapley value calculation for a coalition game.

    v: Characteristic function of the game, must be defined for EVERY coalition
        {tuple -> float}
    players: list of players
    """
    # initialize with zero value
    shapley_values = {p: 0 for p in players}
    v[()] = 0  # empty coalition: zero value 
    n = len(players)
    # iterate over all coalitions
    for c_size in range(1, len(players) + 1):  # iterate over coalition sizes
        for subset in itertools.combinations(players, c_size):  # get all coalitions of size c_size
            k = len(subset)
            for p in subset:  # iterate over player in coalition and change their shapely value
                subset_wo_p = tuple(c for c in subset if c != p)
                if subset_wo_p not in v:
                    print(f"ERROR: Coalition {subset_wo_p} not in characteristic function.")
                    return False
                marginal_contribution = v.get(subset) - v.get(subset_wo_p)
                shapley_values[p] += marginal_contribution * mt.factorial(k - 1) * mt.factorial(n - k) / mt.factorial(n)
    
    return shapley_values