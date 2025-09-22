import control as ct
import numpy as np
import sympy as sp

def get_pi_params():
    """
    returns dictionary with pi controller parameters for each device

    PV, Wind, BESS, Hydro, SC   
    """
    pi_params = {}
    pi_params['PV'] = {"kp": 11.9, "ki": 157.9}
    pi_params['Wind'] = {"kp": 11.9, "ki": 118}
    pi_params['BESS'] = {"kp": 12, "ki": 2370}
    pi_params['Hydro'] = {"kp": -0.0796, "ki": -0.09788}
    pi_params['SC'] = {"kp": 12, "ki": 2370}
    return pi_params



def pi_update(t, x, u, params={}):
    # Get the controller parameters that we need
    kp = params.get('kp')
    ki = params.get('ki', 0.1)
    kaw = params.get('kaw', 15 * ki)  # anti-windup gain
    lower, upper = params.get('saturation_limits', (-np.inf, np.inf))

    err = u[0]   # error signal
    z = x[0]     # integrated error

    # Compute unsaturated output
    u_a = kp * err + ki * z

    # Compute anti-windup compensation (scale by ki to account for structure)
    u_aw = kaw/ki * (u_a - np.clip(u_a, lower, upper)) if ki != 0 else 0

    # State is the integrated error, minus anti-windup compensation
    return err - u_aw

def pi_output(t, x, u, params={}):
    # Get the controller parameters that we need
    kp = params.get('kp')
    ki = params.get('ki')
    lower, upper = params.get('saturation_limits', (-np.inf, np.inf))

    # Assign variables for inputs and states (for readability)
    err = u[0]  # error signal
    z = x[0]    # integrated error

    # PI controller with saturation
    return np.clip(kp * err + ki * z, lower, upper) # account for saturation

def get_pi_controller(params):
    """
    create PI controller with anti-windup

    params: dict with 'kp', 'ki', 'saturation_limits' (tuple), optional: 'kaw' (anti-windup gain)
    """
    return ct.NonlinearIOSystem(
        pi_update, pi_output, name='control',
        inputs=['e'], outputs=['u'], states=2,
        params=params)

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