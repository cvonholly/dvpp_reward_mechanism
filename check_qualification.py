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

def create_curve(params, t_max, n_points=1000):
    """
    create a piecewise linear curve based on the given parameters
    """
    t = np.linspace(0, t_max, n_points)
    v = np.zeros_like(t)
    for (t_start, t_end), (v1, v2) in params.items():
        mask = (t >= t_start) & (t <= t_end)   # include both ends to make final points non-zero
        v[mask] = np.linspace(v1, v2, np.sum(mask))
    return t, v


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
