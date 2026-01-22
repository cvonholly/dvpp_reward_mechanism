# shapely value of coalition

import math as mt
import scipy.optimize as opt
from itertools import combinations, chain
import numpy as np
from scipy.optimize import linprog
import pandas as pd

from src.nucleolus import get_nucleolus
from src.least_core_nucleolus import get_least_core_nucleolus


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

def powerset_tuple(iterable,
             exclude_empty=True) -> list:
    "Subsequences of the iterable from shortest to longest."
    s = list(iterable)
    x = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    if not exclude_empty:
        x = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return [tuple(comb) for comb in x]

def powerset(iterable,
             exclude_empty=False) -> list:
    "frozenset Subsets of the iterable from shortest to longest."
    # powerset([1,2,3]) → {}, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}
    s = list(iterable)
    x = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    if exclude_empty:
        x = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    return [frozenset(comb) for comb in x]

def is_convex_game(v: dict, players: list, tol=5e-3,
                   print_warnings=False) -> bool:
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
                    if print_warnings: print(f"Game is not convex: {S}, {T}")
                    return False
    return True

def game_is_superadditive(v: dict, players: list, tol=5e-4,
                          print_warnings=False) -> bool:
    """
    check if game is superadditive: v(S ∪ T) ≥ v(S) + v(T) for all disjoint coalitions S, T
    """
    for S in powerset(players, exclude_empty=True):
        for T in powerset(players, exclude_empty=True):
            if S.isdisjoint(T):
                union = S.union(T)
                if v[union] < v[S] + v[T] - tol:
                    if print_warnings: print(f"Game is not superadditive: {S}, {T}")
                    return False
    return True

def make_forecasted_realized_superadditive(v: dict, 
                                           v_realized: dict,
                                           players: list,
                                           print_warnings=False) -> dict:
    """
    make forecasted game superadditive and aply same formulation for realized game
    this represnts coalition S bidding the optimal (partitions) bids and the getting the realized value

    v: Characteristic function of the game, must be defined for EVERY coalition
        {frozenset -> float}
    v_realized: Characteristic function of the realized game, must be defined for EVERY coalition
        {frozenset -> float}
    players: list of players
    """
    v_superadd = v.copy()
    for S in powerset(players, exclude_empty=True):
        for T in powerset(players, exclude_empty=True):
            if S.isdisjoint(T):
                union = S.union(T)
                if v_superadd[union] < v_superadd[S] + v_superadd[T]:
                    if print_warnings: print(f"Making game superadditive for: {S}, {T}")
                    v_superadd[union] = v_superadd[S] + v_superadd[T]
                    v_realized[union] = v_realized[S] + v_realized[T]
    return v_superadd, v_realized

def is_superadditive_game(v: dict, players: list, tol=5e-4,
                          print_warnings=False) -> bool:
    """
    check if game is superadditive: v(S ∪ T) ≥ v(S) + v(T) for all disjoint coalitions S, T
    """
    for S in powerset(players, exclude_empty=True):
        for T in powerset(players, exclude_empty=True):
            if S.isdisjoint(T):
                union = S.union(T)
                if v[union] < v[S] + v[T] - tol:
                    if print_warnings: print(f"Game is not superadditive: {S}, {T}")
                    return False
    return True

def make_game_superadditive(v: dict, players: list, print_warnings=False) -> dict:
    """
    make game superadditive by adjusting values of coalitions

    v: Characteristic function of the game, must be defined for EVERY coalition
        {frozenset -> float}
    players: list of players
    """
    v_superadd = v.copy()
    for S in powerset(players, exclude_empty=True):
        for T in powerset(players, exclude_empty=True):
            if S.isdisjoint(T):
                union = S.union(T)
                if v_superadd[union] < v_superadd[S] + v_superadd[T]:
                    v_superadd[union] = v_superadd[S] + v_superadd[T]
    return v_superadd

def core_nonempty(v, players,
                  return_allocation=False):
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
        # print('Core is non-empty, one feasible allocation:', res.x)
        if return_allocation:
            return True, res.x
        else:
            return True
    else:
        if return_allocation:
            return False, None
        else:
            return False



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
    status_quo = {p: v.get(frozenset({p})) for p in players}
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


def solve_optimal_partition(v):
    """
    Finds the partition of players that maximizes the sum of values.
    
    Args:
        v (dict): A dictionary where keys are frozensets (coalitions) 
                  and values are numbers (worth of the coalition).
    
    Returns:
        tuple: (max_value, list_of_coalitions, i.e., [frozenset, ...])
    """
    
    # 1. Identify all unique players
    all_players = set()
    for coalition in v.keys():
        all_players.update(coalition)
    
    # Sort players to ensure consistent bitmask mapping
    players_list = sorted(list(all_players))
    n = len(players_list)
    
    # Map player names to bit indices (0 to n-1)
    player_to_bit = {player: i for i, player in enumerate(players_list)}
    
    # 2. Convert input v (frozensets) to an array indexed by bitmask
    # 1 << n is 2^n
    limit = 1 << n
    
    # worth[mask] stores the raw v(S) value
    worth = [0.0] * limit
    
    # Iterate over the input dictionary and populate the worth array
    for coalition, val in v.items():
        mask = 0
        for p in coalition:
            mask |= (1 << player_to_bit[p])
        worth[mask] = val
        
    # 3. Dynamic Programming
    # dp[mask] stores the Maximum Possible Value for the subset defined by mask
    # splits[mask] stores the submask that led to the optimal split (for reconstruction)
    dp = [0.0] * limit
    best_split = [0] * limit 
    
    # Initialize DP with the base coalition values
    for i in range(limit):
        dp[i] = worth[i]

    # Iterate through all masks from 1 to 2^n - 1
    for mask in range(1, limit):
        # We want to check if splitting 'mask' into 'submask' and 'mask^submask'
        # yields a higher value than the current known value for 'mask'.
        
        # Iterate over all submasks of the current mask.
        # This standard bitwise trick iterates submasks efficiently.
        submask = (mask - 1) & mask
        while submask > 0:
            current_val = dp[submask] + dp[mask ^ submask]
            
            if current_val > dp[mask]:
                dp[mask] = current_val
                best_split[mask] = submask # Record that we split here
            
            submask = (submask - 1) & mask

    # 4. Reconstruct the Optimal Partition
    optimal_coalitions = []
    
    # Helper recursive function to retrieve parts
    def reconstruct(mask):
        if mask == 0:
            return
        
        # If best_split is 0, it means the optimal way for this mask 
        # is to stay together (the raw coalition value was highest)
        # OR we didn't find a split better than the original value.
        # However, we must ensure we don't infinitely recurse if best_split[mask] is 0 
        # but the mask itself is not empty.
        split = best_split[mask]
        
        if split == 0:
            # Reconstruct player names from mask
            coalition = set()
            for i in range(n):
                if (mask >> i) & 1:
                    coalition.add(players_list[i])
            optimal_coalitions.append(frozenset(coalition))
        else:
            # Recursively solve for the two halves
            reconstruct(split)
            reconstruct(mask ^ split)

    reconstruct(limit - 1) # Start with the full set of players (Grand Coalition)

    return dp[limit - 1], optimal_coalitions

def evaluate_full_game(df_forecasted: pd.DataFrame,
                       df_realized: pd.DataFrame,
                       MAKE_games_superadditive=False,
                       print_warnings=False,
                       MAKE_REALIZED_SUPERADDITIVE=False) -> pd.DataFrame:
    # create output df
    index = pd.MultiIndex.from_product([df_forecasted.index.get_level_values(1), ['Forecasted', 'Realized'], ['Value', 'Reward']],)
    # Method can be: ['Shapley', 'Nucleolus', 'Sub-Game']
    player_cols = [c for c in df_forecasted.columns if len(c)==1]  # devices cols
    players = [c[0] for c in df_forecasted.columns if len(c)==1]  # devices
    columns = players + ['Method', 'Method-Realized']
    df = pd.DataFrame(pd.NA, index=index, columns=columns)  # all coalition
    
    for idx, row in df_forecasted.iterrows():
        realized_row = df_realized.loc[idx]
        idx = idx[1]  # adjust index
        df.loc[(idx, 'Forecasted', 'Value'), players] = row[player_cols].values  # add row to output
        v = {frozenset(k): val for k, val in row.items()}  # create value function
        v[frozenset()] = 0
        # get realized values
        df.loc[(idx, 'Realized', 'Value'), players] = realized_row[player_cols].values  # add realized row to output
        v_realized = {frozenset(k): val for k, val in realized_row.items()}  # create value function
        v_realized[frozenset()] = 0
        # make games superadditive because we want to bid optimally
        if MAKE_games_superadditive:
            v, v_realized = make_forecasted_realized_superadditive(v, v_realized, players, print_warnings=print_warnings)
        if MAKE_REALIZED_SUPERADDITIVE:
            v_realized = make_game_superadditive(v_realized, players, print_warnings=print_warnings)
        # check if convex
        if is_convex_game(v, players):
            df.loc[idx, 'Method'] = 'Shapley'
            shapley = get_shapley_value(v, players)
            df.loc[(idx, 'Forecasted', 'Reward'), list(shapley.keys())] = list(shapley.values())
            shapley = get_shapley_value(v_realized, players)
            df.loc[(idx, 'Realized', 'Reward'), list(shapley.keys())] = list(shapley.values())
        # elif check if core non-empty or superadditive
        elif game_is_superadditive(v, players) or core_nonempty(v, players):
            df.loc[idx, 'Method'] = 'Nucleolus'
            # get nucleolus
            nucleolus = get_nucleolus(v, players)
            df.loc[(idx, 'Forecasted', 'Reward'), list(nucleolus.keys())] = list(nucleolus.values())
            # get least core nucleolus as the game may be empty core / not superadditive
            nucleolus = get_least_core_nucleolus(v_realized, players)
            df.loc[(idx, 'Realized', 'Reward'), list(nucleolus.keys())] = list(nucleolus.values())
        else:
            df.loc[idx, 'Method'] = 'Sub-Game'
            # form sub-games
            max_value, coalitions = solve_optimal_partition(v)
            for coalition in coalitions:
                # iterate over coalitions and compute optimal value
                sub_game_value = v[coalition]
                if len(coalition) == 1:
                    # if single device, just assign its value
                    df.loc[(idx, 'Forecasted', 'Reward'), tuple(coalition)[0]] = sub_game_value
                    df.loc[(idx, 'Realized', 'Reward'), tuple(coalition)[0]] = v_realized[coalition]
                else:
                    v_subgame = {k: val for k, val in v.items() if k.issubset(coalition)}
                    v_subgame[frozenset()] = 0
                    if is_convex_game(v_subgame, list(coalition)):
                        shapley_sub = get_shapley_value(v_subgame, list(coalition))
                        for p, val in shapley_sub.items():
                            df.loc[(idx, 'Forecasted', 'Reward'), p] = val
                        shapley_sub = get_shapley_value({k: val for k, val in v_realized.items() if k.issubset(coalition)}, list(coalition))
                        for p, val in shapley_sub.items(): 
                            df.loc[(idx, 'Realized', 'Reward'), p] = val
                    else:
                        nucleolus_sub = get_nucleolus(v_subgame, list(coalition))
                        for p, val in nucleolus_sub.items():
                            df.loc[(idx, 'Forecasted', 'Reward'), p] = val
                        nucleolus_sub = get_least_core_nucleolus({k: val for k, val in v_realized.items() if k.issubset(coalition)}, list(coalition))
                        for p, val in nucleolus_sub.items(): 
                            df.loc[(idx, 'Realized', 'Reward'), p] = val
        # also calculated realized game Shapley / Nucleolus / Sub-Game
        if is_convex_game(v_realized, players):
            df.loc[idx, 'Method-Realized'] = 'Shapley'
        elif game_is_superadditive(v_realized, players) or core_nonempty(v_realized, players):
            df.loc[idx, 'Method-Realized'] = 'Nucleolus'
        else:
            df.loc[idx, 'Method-Realized'] = 'Sub-Game'
    return df