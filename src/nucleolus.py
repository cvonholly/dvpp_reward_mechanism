# %%
import gurobipy as gp
import itertools
import random
import matplotlib.pyplot as plt

from game_theory_helpers import powerset

env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def toleq(a:float, b:float, tol:float=1e-6) -> bool:
    return abs(a-b)<=tol

def nucleolusFullDetail(N, charFun, verbose: bool=False):
    """
    charFun
    """
    n = len(N)
    #  Data PreProcess
    subsets = [S for i in range(n-1) for S in itertools.combinations(N, i+1)]  # subsets excluding empty set and grand coalition
    frozen = {}
    nucl = {pp:0 for pp in N}
    thetas = []
    # The main loop
    while True:
        completed = True
        ## Given the currently frozen coalitions, identifies the minimum possible maximal excess
        M1 = gp.Model("Nucleolus-Phase1", env=env)
        M1.params.LogToConsole=False
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

# def get_two_stage_nucleol


if __name__ == "__main__":
    Ns = ['A', 'B', 'C']
    # first stage solutions (probabilistic for k scenarios)
    first_stage_outcomes = []
    total_rewards = []
    players_ourtcomes = {p: [] for p in Ns}
    for k in range(8):
        # add some randomness
        vs = {k: 0 for k in powerset(Ns)}
        vs[frozenset(['A', 'B'])] = 40 + random.uniform(-5, 5)
        vs[frozenset(['A', 'C'])] = 50 + random.uniform(-5, 5)
        vs[frozenset(['A', 'B', 'C'])] = 100 + random.uniform(-10, 10)

        v_first_stage = get_nucleolus(vs, Ns)

        first_stage_outcomes.append(v_first_stage)
        total_rewards.append(sum(v_first_stage.values()))
        for p in Ns:
            players_ourtcomes[p].append(v_first_stage[p])
    
    print("First stage nucleolus:", first_stage_outcomes)

    # convert each players reward to cdf
    players_cdfs = {p: get_cdf_sections(players_ourtcomes[p]) for p in Ns}
    # print("Players CDFs:", players_cdfs)    

    for p, cdf in players_cdfs.items():
        outcomes, cum_probs = zip(*cdf)
        plt.step(outcomes, cum_probs, where='post', label=p)
    plt.xlabel('Total Reward')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Total Rewards from First Stage Nucleolus')
    plt.grid()
    plt.ylim(0, 1)
    plt.legend()
    plt.show()