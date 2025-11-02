import gurobipy as gp
import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, FrozenSet

def powerset(iterable):
    """Generate all subsets of iterable"""
    s = list(iterable)
    return [frozenset(combo) for r in range(len(s)+1) 
            for combo in itertools.combinations(s, r)]

def toleq(a: float, b: float, tol: float = 1e-6) -> bool:
    """Check if two floats are equal within tolerance"""
    return abs(a - b) <= tol

class ProbabilisticCharacteristicFunction:
    """Represents a probabilistic characteristic function F_S(x)"""
    
    def __init__(self, outcomes: List[float], probabilities: List[float] = None):
        """
        Parameters:
        - outcomes: List of possible values for the coalition
        - probabilities: Corresponding probabilities (uniform if None)
        """
        self.outcomes = sorted(outcomes)
        
        if probabilities is None:
            self.probabilities = [1.0 / len(outcomes)] * len(outcomes)
        else:
            # Normalize probabilities
            total = sum(probabilities)
            self.probabilities = [p / total for p in probabilities]
        
        # Build CDF
        self.cdf_points = []
        cumulative = 0.0
        for outcome, prob in zip(self.outcomes, self.probabilities):
            cumulative += prob
            self.cdf_points.append((outcome, cumulative))
    
    def F(self, x: float) -> float:
        """Evaluate CDF at point x: P(outcome <= x)"""
        if x < self.outcomes[0]:
            return 0.0
        if x >= self.outcomes[-1]:
            return 1.0
        
        # Binary search for the right interval
        for i, (outcome, cdf_val) in enumerate(self.cdf_points):
            if x < outcome:
                if i == 0:
                    return 0.0
                return self.cdf_points[i-1][1]
            elif toleq(x, outcome):
                return cdf_val
        
        return self.cdf_points[-1][1]
    
    def inverse_F(self, p: float, num_points: int = 10000) -> float:
        """Find x such that F(x) = p (approximate)"""
        if p <= 0:
            return self.outcomes[0]
        if p >= 1:
            return self.outcomes[-1]
        
        # Linear interpolation between CDF points
        for i in range(len(self.cdf_points)):
            outcome, cdf_val = self.cdf_points[i]
            if toleq(p, cdf_val) or p < cdf_val:
                if i == 0:
                    return outcome
                prev_outcome, prev_cdf = self.cdf_points[i-1]
                # Linear interpolation
                if toleq(cdf_val, prev_cdf):
                    return outcome
                alpha = (p - prev_cdf) / (cdf_val - prev_cdf)
                return prev_outcome + alpha * (outcome - prev_outcome)
        
        return self.outcomes[-1]

def solve_two_stage_nucleolus(
    N: List[str],
    F_dict: Dict[FrozenSet, ProbabilisticCharacteristicFunction],
    x_star: Dict[str, float],
    RV_N: float,
    verbose: bool = False
) -> Tuple[Dict[str, float], List[float], Dict]:
    """
    Solve the two-stage nucleolus problem.
    
    Parameters:
    - N: List of players
    - F_dict: Dictionary mapping coalitions to their probabilistic characteristic functions
    - x_star: Prior payoff vector (dict mapping players to their prior payoffs)
    - RV_N: Realized value of the grand coalition
    - verbose: Print intermediate results
    
    Returns:
    - nucleolus: Two-stage nucleolus payoff vector
    - thetas: List of optimal excess values at each iteration
    - frozen: Dictionary of frozen coalitions and their excess values
    """
    n = len(N)
    
    # All coalitions except empty set and grand coalition
    subsets = [S for S in powerset(N) if len(S) > 0 and len(S) < n]
    
    # Compute x_star(S) for each coalition
    x_star_S = {}
    for S in powerset(N):
        x_star_S[S] = sum(x_star[p] for p in S)
    
    frozen = {}
    nucl = {p: 0 for p in N}
    thetas = []
    
    # Create Gurobi environment
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    iteration = 0
    while True:
        iteration += 1
        completed = True
        
        if verbose:
            print(f"\n=== Iteration {iteration} ===")
            print(f"Frozen coalitions: {len(frozen)}/{len(subsets)}")
        
        # Phase 1: Minimize maximum probabilistic excess
        M1 = gp.Model("TwoStage-Phase1", env=env)
        M1.params.LogToConsole = False
        
        # Decision variables: y_i for each player
        y = M1.addVars(N, lb=0, name="y")
        
        # Maximum excess variable
        max_excess = M1.addVar(obj=1, lb=-gp.GRB.INFINITY, name="MaxExcess")
        
        # Constraints for each coalition
        for S in subsets:
            y_S_expr = gp.quicksum(y[p] for p in S)
            
            if S in frozen:
                # Frozen coalition: fix the excess
                M1.addConstr(
                    F_dict[S].F(x_star_S[S]) - F_dict[S].F(y_S_expr) == frozen[S],
                    name=f"frozen_{tuple(sorted(S))}"
                )
            else:
                completed = False
                # Non-frozen: constrain by max_excess
                # We need to handle F_S(x*(S)) - F_S(y(S)) <= max_excess
                # This is challenging because F_S(y(S)) depends on y(S)
                # We'll use a piecewise linear approximation
                
                # For simplicity, we discretize the possible values of y(S)
                # and add constraints for each discrete point
                y_S_min = 0
                y_S_max = RV_N
                n_points = 50
                
                # Add auxiliary variable for y(S)
                y_S = M1.addVar(lb=y_S_min, ub=y_S_max, name=f"y_{tuple(sorted(S))}")
                M1.addConstr(y_S == y_S_expr)
                
                # Piecewise linear approximation of -F_S(y(S))
                y_points = np.linspace(y_S_min, y_S_max, n_points)
                F_values = [F_dict[S].F(yp) for yp in y_points]
                
                M1.addGenConstrPWL(
                    y_S, 
                    M1.addVar(lb=-1, ub=1, name=f"F_{tuple(sorted(S))}"),
                    y_points.tolist(),
                    F_values,
                    name=f"pwl_{tuple(sorted(S))}"
                )
                
                # Excess constraint (approximate)
                excess_val = F_dict[S].F(x_star_S[S])
                M1.addConstr(
                    excess_val - F_dict[S].F(y_S_expr) <= max_excess,
                    name=f"exc_{tuple(sorted(S))}"
                )
        
        # Grand coalition constraint
        M1.addConstr(gp.quicksum(y[p] for p in N) == RV_N, name="grand_coalition")
        
        if not completed:
            M1.optimize()
            
            if M1.status != gp.GRB.OPTIMAL:
                if verbose:
                    print(f"Warning: Phase 1 optimization status: {M1.status}")
                break
            
            theta = M1.ObjVal
            thetas.append(theta)
            
            for p in N:
                nucl[p] = y[p].X
            
            if verbose:
                print(f"Theta: {theta:.6f}")
                print(f"Current solution: {nucl}")
        else:
            if verbose:
                print("All coalitions frozen - algorithm complete")
            break
        
        # Phase 2: Identify coalitions to freeze
        # Check which coalitions achieve the maximum excess
        coalitions_to_check = []
        for S in subsets:
            if S not in frozen:
                y_S_val = sum(nucl[p] for p in S)
                excess_S = F_dict[S].F(x_star_S[S]) - F_dict[S].F(y_S_val)
                if toleq(excess_S, theta):
                    coalitions_to_check.append(S)
        
        if verbose:
            print(f"Coalitions achieving maximum excess: {len(coalitions_to_check)}")
        
        # For each coalition achieving max excess, check if it must be frozen
        for S in coalitions_to_check:
            # Create Phase 2 model
            M2 = gp.Model("TwoStage-Phase2", env=env)
            M2.params.LogToConsole = False
            
            y2 = M2.addVars(N, lb=0, name="y2")
            
            # All previous constraints
            for S2 in subsets:
                y2_S2_expr = gp.quicksum(y2[p] for p in S2)
                
                if S2 in frozen:
                    M2.addConstr(
                        F_dict[S2].F(x_star_S[S2]) - F_dict[S2].F(y2_S2_expr) == frozen[S2]
                    )
                else:
                    # Use theta as upper bound
                    M2.addConstr(
                        F_dict[S2].F(x_star_S[S2]) - F_dict[S2].F(y2_S2_expr) <= theta
                    )
            
            M2.addConstr(gp.quicksum(y2[p] for p in N) == RV_N)
            
            # Minimize excess for coalition S
            y2_S_expr = gp.quicksum(y2[p] for p in S)
            M2.setObjective(
                F_dict[S].F(x_star_S[S]) - F_dict[S].F(y2_S_expr),
                gp.GRB.MINIMIZE
            )
            
            M2.optimize()
            
            if M2.status == gp.GRB.OPTIMAL and toleq(M2.ObjVal, theta):
                frozen[S] = theta
                if verbose:
                    print(f"Freezing coalition {tuple(sorted(S))} at excess {theta:.6f}")
        
        if verbose:
            print(f"Total frozen: {len(frozen)}/{len(subsets)}")
    
    return nucl, thetas, frozen


def example_1_from_paper():
    """
    Solve Example 1 from the paper:
    Two-person game with specific distribution functions
    """
    print("="*60)
    print("Example 1 from Charnes-Granot (1977) Paper")
    print("="*60)
    
    N = ['1', '2']
    
    # Define F_1(x) as in the paper
    def F1_func(x):
        if x < 0:
            return 0
        elif x < 1:
            return 0.5
        elif x < 4:
            return 0.75
        elif x < 5:
            return 0.75 + (x - 4) / 2
        else:
            return 1.0
    
    # Define F_2(x) as in the paper
    def F2_func(x):
        if x < 0:
            return 0
        elif x < 3:
            return 1/3
        elif x < 4:
            return 2/3
        elif x < 4 + 1/3:
            return 2/3 + (x - 4)
        else:
            return 1.0
    
    # Create discrete approximations
    x_vals = np.linspace(0, 6, 1000)
    
    # Player 1 outcomes and probabilities
    outcomes_1 = [0, 1, 4, 5]
    probs_1 = [0.5, 0.25, 0, 0.25]  # Approximate from the step function
    
    # Player 2 outcomes and probabilities  
    outcomes_2 = [0, 3, 4, 4.333]
    probs_2 = [1/3, 1/3, 0, 1/3]
    
    # Create probabilistic characteristic functions
    F_1 = ProbabilisticCharacteristicFunction(outcomes_1, probs_1)
    F_2 = ProbabilisticCharacteristicFunction(outcomes_2, probs_2)
    
    # Grand coalition - for simplicity, assume deterministic
    F_12 = ProbabilisticCharacteristicFunction([5], [1.0])
    
    F_dict = {
        frozenset(): ProbabilisticCharacteristicFunction([0], [1.0]),
        frozenset(['1']): F_1,
        frozenset(['2']): F_2,
        frozenset(['1', '2']): F_12
    }
    
    # Prior payoffs from the paper
    x_star = {'1': 2, '2': 4}
    
    # Different realized values to test
    realized_values = [3, 4, 5]
    
    for RV_N in realized_values:
        print(f"\n--- Realized value RV(N) = {RV_N} ---")
        
        nucleolus, thetas, frozen = solve_two_stage_nucleolus(
            N, F_dict, x_star, RV_N, verbose=True
        )
        
        print(f"\nTwo-Stage Nucleolus: {nucleolus}")
        print(f"Excess values through iterations: {thetas}")
        
        # Verify the solution
        y1, y2 = nucleolus['1'], nucleolus['2']
        print(f"\nVerification:")
        print(f"  y(1) = {y1:.4f}, y(2) = {y2:.4f}")
        print(f"  y(1) + y(2) = {y1 + y2:.4f} (should be {RV_N})")
        print(f"  F_1(x*(1)) - F_1(y(1)) = {F_1.F(x_star['1']) - F_1.F(y1):.4f}")
        print(f"  F_2(x*(2)) - F_2(y(2)) = {F_2.F(x_star['2']) - F_2.F(y2):.4f}")


if __name__ == "__main__":
    example_1_from_paper()