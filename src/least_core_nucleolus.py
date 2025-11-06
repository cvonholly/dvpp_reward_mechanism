from bisect import bisect
import gurobipy as gp
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np

from game_theory_helpers import powerset

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
    MAX_ITER, ii = 1e3, 0
    while ii < MAX_ITER:
        completed = True
        ## Given the currently frozen coalitions, identifies the minimum possible maximal excess
        M1 = gp.Model("Nucleolus-Phase1", env=env)
        x = M1.addVars(N, lb=-gp.GRB.INFINITY, name="x")
        excess = M1.addVar(obj=1, lb=-gp.GRB.INFINITY, name="MaxExc")
        # for pp in N: # Only interested in imputations
            # x[pp].lb = charFun([pp])
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
            print(f"first stage nucleolus values and excess: {nucl} and {thetas}")
        else:
            break
        #  Prospective elements in the set Sigma (Notation as per Maschler's book)
        selec = (toleq(
                        theta, 
                        (charFun(S) - gp.quicksum(x[pp] for pp in S)).getValue()
                        ) for S in subsets
                )
        # print(f"selec: {list(selec)}")
        # Solve another model to check which of those prospective members are definitely in Sigma
        M2 = gp.Model("Nucleolus-Phase2", env=env)
        M2.params.LogToConsole=False
        x2 = M2.addVars(N, lb=-gp.GRB.INFINITY, name="x")
        # for pp in N: # least core
        #     x2[pp].lb = charFun([pp])  
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
        ii += 1

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
    
    return nucleolusFullDetail(players, charFun)

def nucleolusLeastCore(N, charFun, verbose: bool=False):
    """
    Computes the nucleolus of the least core (works for non-superadditive games)

    charFun: value function for each coalition S ⊆ N
    N: list of players
    """
    n = len(N)
    # Data PreProcess
    subsets = [S for i in range(n-1) for S in itertools.combinations(N, i+1)]  # subsets excluding empty set and grand coalition
    
    # STEP 0: Find the minimum epsilon (least core threshold)
    M0 = gp.Model("LeastCore-Epsilon", env=env)
    M0.params.LogToConsole = False
    x_eps = M0.addVars(N, lb=-gp.GRB.INFINITY, name="x")
    epsilon = M0.addVar(obj=1, lb=-gp.GRB.INFINITY, name="epsilon")
    
    # Constraints for all proper coalitions (including singletons)
    for i, S in enumerate(subsets):
        M0.addConstr(gp.quicksum(x_eps[pp] for pp in S) >= charFun(S) - epsilon, 
                     name="coal-"+str(i))
    
    # Individual players also need constraints
    for pp in N:
        M0.addConstr(x_eps[pp] >= charFun([pp]) - epsilon, name="ind-"+str(pp))
    
    # Grand coalition constraint (equality with epsilon)
    M0.addConstr(x_eps.sum('*') == charFun(N) + epsilon, name="TotalVal")
    
    M0.optimize()
    min_epsilon = M0.ObjVal
    
    if verbose:
        print(f"Minimum epsilon for least core: {min_epsilon}")
    
    # STEP 1-N: Compute nucleolus within the least core
    frozen = {}
    nucl = {pp: 0 for pp in N}
    thetas = []
    
    # The main loop
    while True:
        completed = True
        ## Given the currently frozen coalitions, identifies the minimum possible maximal excess
        M1 = gp.Model("Nucleolus-Phase1", env=env)
        M1.params.LogToConsole = False
        x = M1.addVars(N, lb=-gp.GRB.INFINITY, name="x")
        excess = M1.addVar(obj=1, lb=-gp.GRB.INFINITY, name="MaxExc")
        
        # NO individual rationality constraints here (this is the key change)
        # Instead, coalitions including singletons are handled uniformly below
        
        # Add constraints for singleton coalitions
        for pp in N:
            S = (pp,)
            if S in frozen:
                M1.addConstr(charFun([pp]) - x[pp] == frozen[S], name="exc-single-"+str(pp))
            else:
                completed = False
                M1.addConstr(charFun([pp]) - x[pp] <= excess, name="exc-single-"+str(pp)+"F")
        
        # Add constraints for all other proper coalitions
        for i, S in enumerate(subsets):
            if S in frozen:
                M1.addConstr(charFun(S) - gp.quicksum(x[pp] for pp in S) == frozen[S], 
                                name="exc-"+str(i))
            else:
                completed = False
                M1.addConstr(charFun(S) - gp.quicksum(x[pp] for pp in S) <= excess, 
                                name="exc-"+str(i)+"F")
        
        # Grand coalition constraint uses epsilon
        M1.addConstr(x.sum('*') == charFun(N) + min_epsilon, name="TotalVal")
        
        ## If every coalition is frozen, then nothing left to optimize. Otherwise, go ahead.
        if not completed:
            M1.optimize()
            theta = M1.ObjVal
            thetas.append(theta)
            for pp in N:
                nucl[pp] = x[pp].X
        else:
            break
        
        # Prospective elements in the set Sigma (including singletons)
        # Check singletons
        selec_singletons = []
        for pp in N:
            S = (pp,)
            if S not in frozen:
                val = (charFun([pp]) - x[pp]).getValue()
                selec_singletons.append((S, toleq(theta, val)))
        
        # Check other coalitions
        selec_others = []
        for S in subsets:
            if S not in frozen:
                val = (charFun(S) - gp.quicksum(x[pp] for pp in S)).getValue()
                selec_others.append((S, toleq(theta, val)))
        
        # Solve another model to check which of those prospective members are definitely in Sigma
        M2 = gp.Model("Nucleolus-Phase2", env=env)
        M2.params.LogToConsole = False
        x2 = M2.addVars(N, lb=-gp.GRB.INFINITY, name="x")
        
        # Singleton constraints
        for pp in N:
            S = (pp,)
            if S in frozen:
                M2.addConstr(charFun([pp]) - x2[pp] == frozen[S], name="exc-single-"+str(pp))
            else:
                M2.addConstr(charFun([pp]) - x2[pp] <= theta, name="exc-single-"+str(pp))
        
        # Other coalition constraints
        for i, S in enumerate(subsets):
            if S in frozen:
                M2.addConstr(charFun(S) - gp.quicksum(x2[pp] for pp in S) == frozen[S], 
                name="exc-"+str(i))
            else:
                M2.addConstr(charFun(S) - gp.quicksum(x2[pp] for pp in S) <= theta, 
                    name="exc-"+str(i))
        
        M2.addConstr(x2.sum('*') == charFun(N) + min_epsilon)
        
        # Check singletons
        for S, is_candidate in selec_singletons:
            if is_candidate:
                pp = S[0]
                M2.setObjective(charFun([pp]) - x2[pp])
                M2.optimize()
                if toleq(theta, M2.ObjVal):
                    frozen[S] = theta
        
        # Check other coalitions
        for S, is_candidate in selec_others:
            if is_candidate:
                M2.setObjective(charFun(S) - gp.quicksum(x2[pp] for pp in S))
                M2.optimize()
                if toleq(theta, M2.ObjVal):
                    frozen[S] = theta
        
        if verbose:
            print(frozen)
            print('----')

    return nucl, frozen, thetas, min_epsilon


def get_least_core_nucleolus(v, players):
    """
    v: v = {frozenset(k): value, ...} where k is a tuple of players in the coalition
    players: list of players

    returns: nucleolus as a dict {player: value, ...}
    """
    # convert v to charFun
    def charFun(S):
        return v.get(frozenset(S))
    
    return nucleolusLeastCore(players, charFun)

def create_bankruptcy_game(players, claims, estate):
    """
    Create a bankruptcy game characteristic function.

    players: list of players
    claims: dict mapping player -> claim amount
    estate: total available estate

    returns: characteristic function v as a dict {frozenset(k): value, ...}
    """
    n = len(players)
    v = {}
    for S in powerset(players):
        if len(S) == 0:
            v[S] = 0
        else:
            total_claims = sum(claims[p] for p in S)
            v[S] = max(0, estate - sum(claims[p] for p in players if p not in S))
    return v

if __name__ == "__main__":
    # construct a simple game
    players = [1,2,3]
    v = {
        frozenset([]): 0,
        frozenset([1]): 0,
        frozenset([2]): 0,
        frozenset([3]): 0,
        frozenset([1,2]): 20,
        frozenset([1,3]): -30,
        frozenset([2,3]): -20,
        frozenset([1,2,3]): -100
    }
    # v = create_bankruptcy_game(players, claims={1:300, 2:200, 3:100}, estate=200)
    print(v)
    nucl, frozen, thetas = get_nucleolus(v, players)
    print("Nucleolus computation")
    print("Nucleolus: ", nucl)

    # nucl, frozen, thetas, min_epsilon = get_least_core_nucleolus(v, players)
    # print("Least Core Nucleolus computation")
    # print("Nucleolus: ", nucl)
    # print("Frozen: ", frozen)
    # print("Thetas: ", thetas)
    # print("Min Epsilon: ", min_epsilon)
