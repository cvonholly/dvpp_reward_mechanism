import numpy as np
from scipy.optimize import minimize_scalar

# # Example data
# p = np.array([0.1, 0.2, 0.3, 0.4])   # probabilities for each k
# b_k = np.array([0.34, 0.7, 1.0, 1.5]) # threshold values for comparison
# pi = 100                             # historic average price

def get_optimal_bid(bids, ps):
    """
    runs optimizaion problem to yield the optimal bid value
    """
    # Expected reward function
    def expected_reward(b, p, b_k):
        # reward depends on whether b_k >= b
        r_k = np.where(b_k >= b, b, -3 * b)
        return np.sum(p * r_k)

    # Objective for optimization (maximize reward)
    def objective(b):
        return -expected_reward(b, ps, bids)

    # Optimize within a reasonable range
    res = minimize_scalar(objective, bounds=(0, max(bids) * 1.5), method='bounded')

    b_opt = res.x
    # reward_opt = -res.fun

    print(f"Optimal bid b*: {b_opt:.4f}")
    return b_opt
    # print(f"Maximum expected reward: {reward_opt:.4f}")