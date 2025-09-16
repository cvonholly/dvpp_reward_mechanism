# shapely value of coalition

import itertools
import math as mt
import scipy.optimize as opt
from itertools import combinations

def get_shapely_value(values: dict, players: list) -> dict:
    """
    Shapley value calculation for a coalition game.

    v: Characteristic function of the game, must be defined for EVERY coalition
        {tuple -> float}
    players: list of players
    """
    # initialize with zero value
    shapley_values = {p: 0 for p in players}
    # convert v to use sets as keys
    v = {frozenset({}): 0}  # empty set: zero value
    for k, vals in values.items():
        v[frozenset(k)] = vals
    n = len(players)
    # iterate over all coalitions
    for c_size in range(1, len(players) + 1):  # iterate over coalition sizes
        for subset in combinations(players, c_size):  # get all coalitions of size c_size
            k = len(subset)
            for p in subset:  # iterate over player in coalition and change their shapely value
                subset = frozenset(subset)
                subset_wo_p = frozenset(c for c in subset if c != p)
                if subset_wo_p not in v:
                    raise ValueError(f"Coalition {subset_wo_p} not in characteristic function.")
                if subset not in v:
                    raise ValueError(f"Coalition {subset} not in characteristic function.")
                marginal_contribution = v.get(subset) - v.get(subset_wo_p)
                shapley_values[p] += marginal_contribution * mt.factorial(k - 1) * mt.factorial(n - k) / mt.factorial(n)
    
    return shapley_values

def get_nash_bargaining_solution(v: dict, players: list, bargaining_power={}) -> dict:
    """
    Nash bargaining solution for a coalition game.

    v: Characteristic function of the game, must be defined for EVERY coalition
        {tuple -> float}
    players: list of players
    bargaining_power (optional): bargaining power of each player
        {player -> float}
    """
    # start by extracting status quo point: value of single players
    status_quo = {p: v.get((p), 0) for p in players}
    # add constraint: sum of values must be less than value of grand coalition
    grand_coalition = tuple(players)
    total = v.get(grand_coalition, 0)
    # construct scipy optimization problem
    if not bargaining_power:
        bargaining_power = {p: 1 for p in players}
    def objective(x):
        prod = 1
        for i, p in enumerate(players):
            prod *= (x[i] - status_quo[p]) ** bargaining_power[p]
        return -prod  # negative for minimization

    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - total}]
    constraints += [{'type': 'ineq', 'fun': lambda x: x[i] - status_quo[p]} for i, p in enumerate(players)]
    result = opt.minimize(objective, x0=list(status_quo.values()), constraints=constraints)
    if not result.success:
        print("ERROR: Optimization failed.")
        raise ValueError("Optimization failed.")
    nash_values = {p: result.x[i] for i, p in enumerate(players)}
    return nash_values