import math


def collision_rate(choices_log):
    """
    Fraction of timesteps where at least two agents chose the same arm.

    In dark pool terms: the fraction of rounds in which multiple institutions
    simultaneously targeted the same venue. High collision rate means agents are
    herding — which is fine under linear_share but costly under zero_on_collision.
    """
    if not choices_log:
        return 0.0
    n_collisions = 0
    for choices in choices_log:
        arm_counts = {}
        for arm in choices:
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
        if any(v > 1 for v in arm_counts.values()):
            n_collisions += 1
    return n_collisions / len(choices_log)


def market_welfare(rewards_log):
    """
    Total reward realized by all agents at each timestep.

    In dark pool terms: the total execution quality captured by the market as a
    whole. Under linear_share, welfare is preserved (split, not destroyed).
    Under zero_on_collision, every collision destroys welfare entirely.
    Under winner_takes_all, welfare is preserved in expectation but concentrated.

    Returns a list of per-step welfare values (length = T).
    """
    return [sum(step) for step in rewards_log]


def arm_selection_entropy(choices_log, n_arms, agent_idx):
    """
    Shannon entropy of arm selection for a given agent over its entire history.

      H = 0          → agent always picks the same arm (full exploitation / herding)
      H = log(n_arms) → agent picks each arm equally often (maximum diversification)

    In dark pool terms: measures how much the institution is spreading its
    order flow across venues (venue fragmentation). High entropy ≈ the agent
    has learned to act like a real SOR that diversifies to avoid adverse selection.
    """
    choices = [step[agent_idx] for step in choices_log]
    counts = {}
    for arm in choices:
        counts[arm] = counts.get(arm, 0) + 1
    total = len(choices)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log(p) for p in probs)


def rolling_arm_entropy(choices_log, n_arms, agent_idx, window=100):
    """
    Shannon entropy of arm selection computed over a rolling window of `window`
    timesteps. Tracks how diversification evolves over the course of learning.

    Key observable: under zero_on_collision, entropy should rise quickly as
    agents burn themselves on collisions and learn to spread. Under linear_share,
    entropy should remain low — herding on the best arm is still the rational
    strategy because collisions don't destroy value.

    Returns a list of entropy values, one per timestep (length = T).
    """
    entropies = []
    for t in range(len(choices_log)):
        start = max(0, t - window + 1)
        window_choices = [choices_log[s][agent_idx] for s in range(start, t + 1)]
        counts = {}
        for arm in window_choices:
            counts[arm] = counts.get(arm, 0) + 1
        total = len(window_choices)
        probs = [c / total for c in counts.values()]
        h = -sum(p * math.log(p) for p in probs) if len(probs) > 1 else 0.0
        entropies.append(h)
    return entropies


def welfare_loss_from_collisions(choices_log, rewards_log, oracle_arm_mean, n_agents):
    """
    Estimated welfare destroyed by collisions under zero_on_collision.

    When agents collide and all get 0, the 'lost' welfare is approximately
    n_colliding_agents * oracle_arm_mean for that step. This gives a measure
    of the social cost of uncoordinated herding.

    oracle_arm_mean: the mean of the best arm (upper bound on per-step welfare).
    Returns total estimated welfare loss over the run.
    """
    total_loss = 0.0
    for t, choices in enumerate(choices_log):
        arm_counts = {}
        for arm in choices:
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
        realized = sum(rewards_log[t])
        # Number of agents who collided and got nothing
        colliders = sum(count for count in arm_counts.values() if count > 1)
        # Approximate what they would have gotten with perfect coordination
        counterfactual = colliders * oracle_arm_mean
        total_loss += max(0.0, counterfactual - realized)
    return total_loss
