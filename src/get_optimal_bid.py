import numpy as np
from scipy.optimize import minimize_scalar

# # Example data
# p = np.array([0.1, 0.2, 0.3, 0.4])   # probabilities for each k
# b_k = np.array([0.34, 0.7, 1.0, 1.5]) # threshold values for comparison
# pi = 100                             # historic average price

def get_optimal_bid(bids, ps,
                    max_bid_value=1e3):
    """
    runs optimizaion problem to yield the optimal bid value

    bids: array of bid values (b_k)
    ps: array of probabilities (p_k)
    max_bid_value: maximum bid value to consider
    """
    # Expected reward function
    def expected_reward(b, p, b_k):
        # reward depends on whether b_k >= b
        r_k = np.where(b_k >= b, b, -3 * b)
        return np.sum(p * r_k)

    # Objective for optimization (maximize reward)
    def objective(b):
        return -expected_reward(b, ps, bids)
    
    # check if bids and ps are valid
    if len(bids) != len(ps):
        print("Length of bids and probabilities must be the same.")
        return 0  # default bid
    if not np.isclose(np.sum(ps), 1):
        print("Probabilities must sum to 1.")
        return 0  # default bid
    if max(bids) > max_bid_value or np.mean(bids) <= 0:
        print("Bids are: ", bids, " submitting 0 bid (not participating).")
        return 0  # default bid

    # Optimize within a reasonable range
    res = minimize_scalar(objective, bounds=(0, max(bids) * 1.5), method='bounded')

    b_opt = res.x
    # reward_opt = -res.fun

    print(f"Optimal bid b*: {b_opt:.4f}")
    return b_opt
    # print(f"Maximum expected reward: {reward_opt:.4f}")