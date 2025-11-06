import numpy as np


def get_optimal_bid(bids, ps,
                    max_bid_value=1e3,
                    return_reward=False):
    """
    Runs optimization problem to yield the optimal bid value
    by checking only the discrete values present in the bids array.

    bids: array of competitor bid values (b_k)
    ps: array of probabilities (p_k)
    max_bid_value: maximum bid value to consider
    return_reward: if True, returns (b_opt, reward, gamma)
    """
    
    bids = np.asarray(bids)
    ps = np.asarray(ps)

    # Expected reward function (unchanged)
    def expected_reward(b, p, b_k):
        r_k = np.where(b_k >= b, b, -3 * b)
        return np.sum(p * r_k)

    # check if bids and ps are valid (unchanged)
    if len(bids) != len(ps):
        print("Length of bids and probabilities must be the same.")
        if return_reward:
            return 0, 0, 1  # default bid and reward
        return 0  # default bid
    if not np.isclose(np.sum(ps), 1):
        print("Probabilities must sum to 1.")
        if return_reward:
            return 0, 0, 1  # default bid and reward
        return 0  # default bid
    if max(bids) > max_bid_value or np.mean(bids) <= 0:
        print(f"Bids are invalid (max bid: {max(bids):.3f}, mean bid: {np.mean(bids):.3f}), submitting 0 bid (not participating).")
        if return_reward:
            return 0, 0, 1  # default bid and reward
        return 0  # default bid

    # --- Optimization ---
    
    # Create a set of candidate bids to check.
    candidate_bids = np.unique(bids)
    if 0 not in candidate_bids:
        candidate_bids = np.append(candidate_bids, 0)

    # Calculate the expected reward for *each* candidate bid
    rewards = [expected_reward(b, ps, bids) for b in candidate_bids]
    
    # Find the index of the highest reward
    best_index = np.argmax(rewards)
    
    # Get the optimal bid and reward from our discrete set
    b_opt = candidate_bids[best_index]
    reward = rewards[best_index]

    # --- START: NEW CALCULATION ---
    
    # Calculate gamma: sum of probabilities where b_k >= b_opt
    gamma = np.sum(ps[bids >= b_opt])
    
    # --- END: NEW CALCULATION ---

    if return_reward:
        # --- MODIFIED RETURN ---
        return b_opt, reward, gamma

    return b_opt


if __name__ == "__main__":
    # --- Example Usage ---
    # Competitor bids and their probabilities
    example_bids = np.arange(1, 11) #np.array([10, 20, 30, 40, 50])
    N = len(example_bids)
    example_ps = np.repeat(1/N, N) #np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    # This will now return one of the values from [0, 10, 20, 30, 40, 50]
    opt_bid, opt_reward = get_optimal_bid(example_bids, example_ps, return_reward=True)

    print(f"Optimal Bid: {opt_bid}")
    print(f"Expected Reward: {opt_reward}")