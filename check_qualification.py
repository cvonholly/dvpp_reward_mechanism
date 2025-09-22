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
