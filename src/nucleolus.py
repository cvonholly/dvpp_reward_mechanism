from bisect import bisect
import gurobipy as gp
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np

def powerset(iterable,
             exclude_empty=False) -> list:
    "frozenset Subsets of the iterable from shortest to longest."
    # powerset([1,2,3]) → {}, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}
    s = list(iterable)
    x = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return [frozenset(comb) for comb in x]

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def toleq(a:float, b:float, tol:float=1e-6) -> bool:
    return abs(a-b)<=tol

def nucleolusFullDetail(N, charFun, verbose: bool=False):
    """
    charFun: value function for each coalition S ⊆ N
    N: list of players
    """
    n = len(N)
    #  Data PreProcess
    subsets = [S for i in range(n-1) for S in itertools.combinations(N, i+1)]  # subsets excluding empty set and grand coalition
    frozen = {}
    nucl = {pp: 0 for pp in N}
    thetas = []
    # The main loop
    while True:
        completed = True
        ## Given the currently frozen coalitions, identifies the minimum possible maximal excess
        M1 = gp.Model("Nucleolus-Phase1", env=env)
        x = M1.addVars(N, lb=-gp.GRB.INFINITY, name="x")
        excess = M1.addVar(obj=1, lb=-gp.GRB.INFINITY, name="MaxExc")
        for pp in N: # Only interested in imputations
            x[pp].lb = charFun([pp])
        for i, S in enumerate(subsets):
            if S in frozen:
                M1.addConstr(charFun(S) - gp.quicksum(x[pp] for pp in S) == frozen[S], 
                                name="exc-"+str(i))
            else:
                completed = False
                M1.addConstr(charFun(S) - gp.quicksum(x[pp] for pp in S) <= excess, 
                                name="exc-"+str(i)+"F")
        M1.addConstr(x.sum('*') == charFun(N), name="TotalVal")
        ## If every coalition is frozen, then nothing left to optimize. Otherwise, go ahead.
        if not completed:
            M1.optimize()
            theta = M1.ObjVal
            thetas.append(theta)
            for pp in N:
                nucl[pp] = x[pp].X
        else:
            break
        #  Prospective elements in the set Sigma (Notation as per Maschler's book)
        selec = (toleq(
                        theta, 
                        (charFun(S) - gp.quicksum(x[pp] for pp in S)).getValue()
                        ) for S in subsets
                )
        # Solve another model to check which of those prospective members are definitely in Sigma
        M2 = gp.Model("Nucleolus-Phase2", env=env)
        M2.params.LogToConsole=False
        x2 = M2.addVars(N, lb=-gp.GRB.INFINITY, name="x")
        for pp in N: # Only interested in imputations
            x2[pp].lb = charFun([pp])
        for i,S in enumerate(subsets):
            if S in frozen:
                M2.addConstr(charFun(S) - gp.quicksum(x2[pp] for pp in S) == frozen[S], 
                name="exc-"+str(i))
            else:
                # Observe, here the RHS is the objective value obtained earlier
                M2.addConstr(charFun(S) - gp.quicksum(x2[pp] for pp in S) <= theta, 
                    name="exc-"+str(i))
        M2.addConstr(x2.sum('*') == charFun(N))

        for SS in itertools.compress(subsets, selec):
            # https://stackoverflow.com/questions/8312829/how-to-remove-item-from-a-python-list-in-a-loop
            # Answer by Gurney Alex - https://stackoverflow.com/a/8313120
            M2.setObjective(charFun(SS) - gp.quicksum(x2[pp] for pp in SS))
            M2.optimize()
            if toleq(theta, M2.ObjVal):
                frozen[SS] = theta
        if verbose:
            print(frozen)
            print('----')

    return nucl, frozen, thetas

def get_nucleolus(v, players):
    """
    v: v = {frozenset(k): value, ...} where k is a tuple of players in the coalition
    players: list of players

    returns: nucleolus as a dict {player: value, ...}
    """
    # convert v to charFun
    def charFun(S):
        return v.get(frozenset(S))
    
    nucl, frozen, thetas = nucleolusFullDetail(players, charFun)
    return nucl

def get_cdf_sections(outcomes, probabilities=None):
    """
    Constructs the CDF from outcomes and optional probabilities.
    
    Parameters:
    - outcomes (list): List of outcomes.
    - probabilities (list, optional): Corresponding probabilities. If None, assumes uniform distribution.
    
    Returns:
    - List of tuples: [(outcome1, cdf1), (outcome2, cdf2), ...] sorted by outcome.
    """
    from collections import Counter

    # If probabilities not provided, compute uniform or frequency-based
    if probabilities is None:
        total = len(outcomes)
        outcome_counts = Counter(outcomes)
        probabilities = [outcome_counts[x] / total for x in outcomes]
    
    # Pair outcomes with probabilities
    data = list(zip(outcomes, probabilities))

    # Aggregate probabilities by outcome
    outcome_prob = {}
    for outcome, prob in data:
        outcome_prob[outcome] = outcome_prob.get(outcome, 0) + prob

    # Sort outcomes
    sorted_outcomes = sorted(outcome_prob.items())

    # Compute cumulative probabilities
    cdf = []
    cumulative = 0.0
    for outcome, prob in sorted_outcomes:
        cumulative += prob
        cdf.append((outcome, cumulative))

    return cdf

def get_cdf_function(outcomes, probabilities=None):
    """
    Returns a CDF function that can be evaluated at any x.
    """
    from collections import Counter
    import bisect

    # Compute probabilities if not given
    if probabilities is None:
        total = len(outcomes)
        outcome_counts = Counter(outcomes)
        probabilities = [outcome_counts[x] / total for x in outcomes]

    # Aggregate probabilities by outcome
    outcome_prob = {}
    for o, p in zip(outcomes, probabilities):
        outcome_prob[o] = outcome_prob.get(o, 0) + p

    # Sort outcomes and build cumulative probabilities
    sorted_outcomes = sorted(outcome_prob.keys())
    probs = [outcome_prob[o] for o in sorted_outcomes]

    cumulative = []
    total = 0.0
    for p in probs:
        total += p
        cumulative.append(total)

    # Return a function
    def cdf(x):
        idx = bisect.bisect_right(sorted_outcomes, x)
        if idx == 0:
            return 0.0
        else:
            return cumulative[idx - 1]

    return cdf

def get_inverse_cdf_function(cdf_func, lower_bound, upper_bound, num_points=1000):
    """
    Constructs an inverse CDF function using numerical methods.
    
    Parameters:
    - cdf_func: Function that computes the CDF.
    - lower_bound: Lower bound of the outcome space.
    - upper_bound: Upper bound of the outcome space.
    - num_points: Number of points to sample for constructing the inverse CDF.
    
    Returns:
    - Function that computes the inverse CDF.
    """
    xs = np.linspace(lower_bound, upper_bound, num_points)
    cdf_values = [cdf_func(x) for x in xs]

    def inv_cdf(p):
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be in [0, 1]")
        idx = bisect.bisect_left(cdf_values, p)
        if idx == 0:
            return xs[0]
        elif idx == len(cdf_values):
            return xs[-1]
        else:
            return xs[idx]

    return inv_cdf

def get_bankruptcy_game(vs_priors: list, v_realized: dict, players):
    """
    compute bankruptcy game from prior value functions and realized value function
    """
    # compute ex-ante rewards
    first_stage_outcomes, ex_ante_rewards, players_ourtcomes, v_prior = [], [], {p: [] for p in players}, {S: 0 for S in powerset(players)}
    K = len(vs_priors)
    for v in vs_priors:
        reward_ex_ante = get_nucleolus(v, players)
        first_stage_outcomes.append(reward_ex_ante)
        ex_ante_rewards.append(sum(reward_ex_ante.values()))
        for p in players:
            players_ourtcomes[p].append(reward_ex_ante[p])
        for k, v in vs.items():
            v_prior[k] += v / K  # average over K runs

    # compute realized reward
    rewards_ex_post = get_nucleolus(v_realized, players)
    total_realized = sum(rewards_ex_post.values())

    # create solution
    if total_realized < np.mean(ex_ante_rewards) and total_realized > 0:
        # create bankruptcy game
        grand_coalition = frozenset(players)
        claims = {p: sum(players_ourtcomes[p]) / K for p in players}
        v_bankruptcy = {k: 0 for k in powerset(players)}
        v_bankruptcy[grand_coalition] = v_realized[grand_coalition]
        for k, v in v_prior.items():
            if k != grand_coalition:
                v_bankruptcy[k] = max(0, total_realized - sum(claims[p] for p in k))
        rewards_bankruptcy = get_nucleolus(v_bankruptcy, players)
        return rewards_bankruptcy
    elif total_realized < 0:
        # now we have a penalty which we have to distribute
        # we use value function here as thetas not available
        # todo
        return {}
    else:
        return rewards_ex_post


def get_closed_loop_nucleolus(vs_priors: list, v_realized: dict, players):
    """
    compute closed-loop nucleolus from prior value functions and realized value function
    """
    # compute ex-ante rewards
    first_stage_outcomes, ex_ante_rewards, players_ourtcomes, v_prior = [], [], {p: [] for p in players}, {S: 0 for S in powerset(players)}
    K = len(vs_priors)
    for v in vs_priors:
        reward_ex_ante = get_nucleolus(v, players)
        first_stage_outcomes.append(reward_ex_ante)
        ex_ante_rewards.append(sum(reward_ex_ante.values()))
        for p in players:
            players_ourtcomes[p].append(reward_ex_ante[p])
        for k, v in vs.items():
            v_prior[k] += v / K  # average over K runs

    # compute realized reward
    rewards_ex_post = get_nucleolus(v_realized, players)
    total_realized = sum(rewards_ex_post.values())

    # create solution
    if total_realized < np.mean(ex_ante_rewards) and total_realized > 0:
        # create bankruptcy game
        grand_coalition = frozenset(players)
        claims = {p: sum(players_ourtcomes[p]) / K for p in players}
        v_closed_loop = {k: 0 for k in powerset(players)}
        v_closed_loop[grand_coalition] = v_realized[grand_coalition]
        for S, v in v_prior.items():
            if S != grand_coalition:
                v_closed_loop[S] = min(v_prior[S], v_realized[S])
        rewards_closed_loop = get_nucleolus(v_closed_loop, players)
        return rewards_closed_loop
    elif total_realized < 0:
        # now we have a penalty which we have to distribute
        # we use value function here as thetas not available
        # todo
        return {}
    else:
        return rewards_ex_post

if __name__ == "__main__":
    Ns = ['A', 'B', 'C']
    # first stage solutions (probabilistic for k scenarios)
    first_stage_outcomes = []
    total_rewards = []
    players_ourtcomes = {p: [] for p in Ns}
    vs_first_stage = {S: 0 for S in powerset(Ns)}
    K = 8

    all_vs = []

    for k in range(K):
        # add some randomness
        vs = {k: 0 for k in powerset(Ns)}
        vs[frozenset(['A', 'B'])] = 40 + random.uniform(-5, 5)
        vs[frozenset(['A', 'C'])] = 50 + random.uniform(-5, 5)
        vs[frozenset(['A', 'B', 'C'])] = 100 + random.uniform(-10, 10)

        all_vs.append(vs)

        rewards_first_stage = get_nucleolus(vs, Ns)

        first_stage_outcomes.append(rewards_first_stage)
        total_rewards.append(sum(rewards_first_stage.values()))
        for p in Ns:
            players_ourtcomes[p].append(rewards_first_stage[p])
        for k, v in vs.items():
            vs_first_stage[k] += v / K  # average over K runs
    
    # print("First stage nucleolus:", first_stage_outcomes)
    # compute expected reward for each player
    expected_rewards = {p: sum(players_ourtcomes[p]) / K for p in Ns}
    print("Expected rewards from first stage nucleolus:", expected_rewards)

    # now get second stage reward, smaller then first stage
    vs = {k: 0 for k in powerset(Ns)}
    vs[frozenset(['A', 'B'])] = 40 + random.uniform(-5, 5)
    vs[frozenset(['A', 'C'])] = 50 + random.uniform(-5, 5)
    vs[frozenset(['A', 'B', 'C'])] = 100 + random.uniform(-10, 10)

    results_second_stage = get_nucleolus(vs, Ns)
    print("Second stage nucleolus (if negative -> sanction):", results_second_stage)

    # create penalty game
    grand_coalition = frozenset(Ns)
    v_penalty = {k: 0 for k in powerset(Ns)}
    v_penalty[grand_coalition] = vs[grand_coalition]
    for k, v in vs_first_stage.items():
        if k != grand_coalition:
            v_penalty[k] = min(vs_first_stage[k], vs[k])
    rewards_penalty = get_nucleolus(v_penalty, Ns)
    print("Penalty nucleolus:", rewards_penalty)

    # # convert each players reward to cdf
    players_cdfs = {p: get_cdf_sections(players_ourtcomes[p]) for p in Ns}

    # construct cdf's for two stage for each coalition
    # coalition_cdfs = {}
    value_function_Fx_star = {frozenset({}): 0}  # probabilistic value function for each coalition
    for S in powerset(Ns):
        if len(S)>0:
            def this_cdf(x_inner, S=S):
                print("Evaluating CDF for coalition:", S, "at x* =", x_inner)
                # all_coalition_vals = [v[S] for v in all_vs]
                # coalition allocation is the sum of players allocation in S
                all_coalition_allocation = [sum(x[p] for p in S) for x in first_stage_outcomes]
                print("All coalition renumeration for S:", S, "->", all_coalition_allocation)
                count = sum(1 for alloc in all_coalition_allocation if alloc <= x_inner)
                return count / K
            # sum over expected rewards of players in S
            x_star = sum(expected_rewards[p] for p in S)
            # construct value function
            value_function_Fx_star[S] = this_cdf(x_star)

    print("value function for x*:", value_function_Fx_star)
    
    # compute nucleolus for the two-stage game
    rewards_two_stage = get_nucleolus(value_function_Fx_star, Ns)
    print("Two-stage nucleolus:", rewards_two_stage)

    # # plot cdfs    
    # # for p, cdf in players_cdfs.items():
    #     outcomes, cum_probs = zip(*cdf)
    #     plt.step(outcomes, cum_probs, where='post', label=p)
    # plt.xlabel('Total Reward')
    # plt.ylabel('Cumulative Probability')
    # plt.title('CDF of Total Rewards from First Stage Nucleolus')
    # plt.grid()
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.show()