# shapely value of coalition

import math as mt
import scipy.optimize as opt
from itertools import combinations, chain
import numpy as np
from scipy.optimize import linprog
import pandas as pd


def get_banzhaf_value(v: dict, players: list, normalized=True) -> dict:
    """
    get Banzhaf values for each player

    v: Characteristic function of the game, must be defined for EVERY coalition
        {frozenset -> float}
    players: list of players
    """
    factor = 1 / (2 ** (len(players) - 1))
    banzhaf_values = {p: 0 for p in players}
    for S in powerset(players):
        if S:  # skip 
            for p in S:
                subset_wo_p = frozenset(c for c in S if c != p)
                marginal_contribution = v.get(S) - v.get(subset_wo_p)
                banzhaf_values[p] += marginal_contribution
    banzhaf_values = {k: v * factor for k, v in banzhaf_values.items()}
    if normalized:
        total_banzhaf = sum(banzhaf_values.values())
        reward = v[frozenset(players)]
        if total_banzhaf > 0:
            banzhaf_values = {k: v * (reward / total_banzhaf) for k, v in banzhaf_values.items()}
    return banzhaf_values

def powerset(iterable):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    x = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return [frozenset(comb) for comb in x]

def is_convex_game(v: dict, players: list, tol=5e-2) -> bool:
    """
    Check if a coalition game is convex.

    critertion: v(S ∪ T) + v(S ∩ T) ≥ v(S) + v(T) for all coalitions S, T

    v: Characteristic function of the game, must be defined for EVERY coalition
        {frozenset -> float}
    """
    for S in powerset(players):
        for T in powerset(players):
            if S and T:  # skip empty sets
                union = S.union(T)
                intersection = S.intersection(T)
                if v[union] + v[intersection] < v[S] + v[T] - tol:
                    print(f"Game is not convex: {S}, {T}")
                    return False
    return True


def core_nonempty(v, players):
    """
    Check if the core of a TU cooperative game is non-empty.

    Parameters:
    - v: dict mapping coalitions (as frozensets of players) -> value
    - players: list of players

    Returns:
    - True if core is non-empty, False otherwise
    """
    N = len(players)
    player_to_index = {p: i for i, p in enumerate(players)}

    # Efficiency: sum x_i = v(N)
    A_eq = [[1] * N]
    b_eq = [v[frozenset(players)]]

    # Inequalities: sum_{i in S} x_i >= v(S)
    A_ub, b_ub = [], []
    for r in range(1, N):  # skip empty and grand coalition
        for S in combinations(players, r):
            row = [0] * N
            for i in S:
                row[player_to_index[i]] = -1  # negate for <= constraint
            A_ub.append(row)
            b_ub.append(-v[frozenset(S)])

    # Objective: irrelevant (feasibility), minimize 0
    c = [0] * N

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")

    if res.success:
        print('Core is non-empty, one feasible allocation:', res.x)
        return True  #, res.x
    else:
        return False   #, None



def get_loo(v: dict, players: list, normalized=True) -> dict:
    """
    get Leave-One-Out values for each player

    v: Characteristic function of the game, must be defined for EVERY coalition
        {frozenset -> float}
    players: list of players
    """
    loo_values = {}
    grand_coalition = frozenset(players)
    for p in players:
        subset_wo_p = frozenset(c for c in players if c != p)
        diff = v.get(grand_coalition) - v.get(subset_wo_p)
        if diff < 0: diff=0
        loo_values[p] = diff
    if normalized:
        total_loo = sum(loo_values.values())
        reward = v[grand_coalition]
        if total_loo > 0:
            loo_values = {k: v * (reward / total_loo) for k, v in loo_values.items()}
    return loo_values

def get_banzhaf_value(v: dict, players: list, normalized=True) -> dict:
    """
    get Banzhaf values for each player

    v: Characteristic function of the game, must be defined for EVERY coalition
        {frozenset -> float}
    players: list of players
    """
    factor = 1 / (2 ** (len(players) - 1))
    banzhaf_values = {p: 0 for p in players}
    for S in powerset(players):
        if S:  # skip 
            for p in S:
                subset_wo_p = frozenset(c for c in S if c != p)
                marginal_contribution = v.get(S) - v.get(subset_wo_p)
                banzhaf_values[p] += marginal_contribution
    banzhaf_values = {k: v * factor for k, v in banzhaf_values.items()}
    if normalized:
        total_banzhaf = sum(banzhaf_values.values())
        reward = v[frozenset(players)]
        if total_banzhaf > 0:
            banzhaf_values = {k: v * (reward / total_banzhaf) for k, v in banzhaf_values.items()}
    return banzhaf_values

def get_shapley_value(values: dict, players: list,
                      key_type='tuple') -> dict:
    """
    Shapley value calculation for a coalition game.

    v: Characteristic function of the game, must be defined for EVERY coalition
        {tuple/frozenset -> float}
    players: list of players
    """
    # initialize with zero value
    shapley_values = {p: 0 for p in players}
    # convert v to use sets as keys
    v = {frozenset({}): 0}  # empty set: zero value
    if key_type != 'frozenset':
        for k, vals in values.items():
            v[frozenset(k)] = vals
    else:
        v = values
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
        {frozenset -> float}
    players: list of players
    bargaining_power (optional): bargaining power of each player
        {player -> float}
    """
    # start by extracting status quo point: value of single players
    status_quo = {p: v.get(frozenset({p}), 0) for p in players}
    # add constraint: sum of values must be less than value of grand coalition
    grand_coalition = frozenset(players)
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