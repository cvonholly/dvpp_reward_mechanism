# shapely value of coalition

import itertools
import math as mt

def get_shapely_value(v: dict, players: list) -> dict:
    """
    Shapley value calculation for a coalition game.

    v: Characteristic function of the game, must be defined for EVERY coalition
        {tuple -> float}
    players: list of players
    """
    # initialize with zero value
    shapley_values = {p: 0 for p in players}
    v[frozenset({})] = 0  # empty set: zero value
    # convert v to use sets as keys
    v = {frozenset(k): val for k, val in v.items()}
    n = len(players)
    # iterate over all coalitions
    for c_size in range(1, len(players) + 1):  # iterate over coalition sizes
        for subset in itertools.combinations(players, c_size):  # get all coalitions of size c_size
            k = len(subset)
            for p in subset:  # iterate over player in coalition and change their shapely value
                subset = frozenset(subset)
                subset_wo_p = frozenset(c for c in subset if c != p)
                if subset_wo_p not in v:
                    print(f"ERROR: Coalition {subset_wo_p} not in characteristic function.")
                    raise ValueError(f"Coalition {subset_wo_p} not in characteristic function.")
                if subset not in v:
                    print(f"ERROR: Coalition {subset} not in characteristic function.")
                    raise ValueError(f"Coalition {subset} not in characteristic function.")
                marginal_contribution = v.get(subset) - v.get(subset_wo_p)
                shapley_values[p] += marginal_contribution * mt.factorial(k - 1) * mt.factorial(n - k) / mt.factorial(n)
    
    return shapley_values