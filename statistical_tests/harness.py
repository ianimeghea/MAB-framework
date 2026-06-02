"""
Simulation harness for statistical tests.

Runs the MAB simulation for a given configuration and returns per-seed
metric vectors suitable for hypothesis testing.
"""

import random
import math
import numpy as np

from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.arm import Arm
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.metrics import (
    collision_rate,
    market_welfare,
    arm_selection_entropy,
)
from multi_agent_bandits.core.reward_sharing import (
    linear_share,
    zero_on_collision,
    winner_takes_all,
)
from multi_agent_bandits.strategies.ucb_baseline import UCB_BaselineAgent
from multi_agent_bandits.strategies.thompson_sampling import ThompsonSamplingAgent
from multi_agent_bandits.strategies.sliding_window_ucb import SlidingWindowUCBAgent
from multi_agent_bandits.strategies.kl_ucb import KLUCBAgent
from multi_agent_bandits.strategies.epsilon_greedy import EpsilonGreedyAgent
from multi_agent_bandits.strategies.random import RandomAgent
from multi_agent_bandits.strategies.static_baselines import (
    FixedArmAgent,
    RoundRobinAgent,
)


# ── Arm configurations ──────────────────────────────────────────────────────

ARMS_WIDE = [
    (1.0, 1.0),
    (2.0, 1.2),
    (3.0, 1.0),
    (4.0, 1.5),
    (5.0, 1.0),
]

ARMS_CALIBRATED = [
    (2.430, 0.450),   # JPM-X
    (2.840, 0.710),   # Level ATS
    (3.200, 0.840),   # Sigma X2
    (3.800, 1.190),   # Intelligent Cross
    (5.000, 1.480),   # UBS ATS
]

POLICY_MAP = {
    "linear_share": linear_share,
    "zero_on_collision": zero_on_collision,
    "winner_takes_all": winner_takes_all,
}


def make_agents(strategy_name, n_agents, n_arms, best_arm_idx=None):
    """Factory for creating agent lists by strategy name."""
    builders = {
        "UCB": lambda: UCB_BaselineAgent(n_arms),
        "TS": lambda: ThompsonSamplingAgent(n_arms),
        "SW-UCB": lambda: SlidingWindowUCBAgent(n_arms, window=200),
        "KL-UCB": lambda: KLUCBAgent(n_arms),
        "EG(0.05)": lambda: EpsilonGreedyAgent(n_arms, epsilon=0.05),
        "EG(0.20)": lambda: EpsilonGreedyAgent(n_arms, epsilon=0.20),
        "Greedy": lambda: EpsilonGreedyAgent(n_arms, epsilon=0.0),
        "Random": lambda: RandomAgent(n_arms),
        "Fixed(best)": lambda: FixedArmAgent(n_arms, arm_idx=best_arm_idx or n_arms - 1),
        "Fixed(mid)": lambda: FixedArmAgent(n_arms, arm_idx=n_arms // 2),
        "Fixed(worst)": lambda: FixedArmAgent(n_arms, arm_idx=0),
        "RoundRobin": lambda: RoundRobinAgent(n_arms),
    }
    return [builders[strategy_name]() for _ in range(n_agents)]


def run_single_seed(seed, arms_config, policy_fn, strategy_name, n_agents, steps):
    """Run one replication. Returns choices_log, rewards_log, n_arms."""
    random.seed(seed)
    np.random.seed(seed)
    arms = [Arm(mean=m, sd=sd) for m, sd in arms_config]
    n_arms = len(arms)
    env = Environment(n_agents=n_agents, arms=arms, collision_policy=policy_fn)
    agents = make_agents(strategy_name, n_agents, n_arms)
    runner = ExperimentRunner(env, agents, timestep_limit=steps)
    runner.print_experiment_info = lambda: None  # suppress output
    choices_log, rewards_log = runner.run()
    return choices_log, rewards_log, n_arms


def run_single_seed_regime(seed, arms_config, policy_fn, strategy_name,
                           n_agents, steps_per_session, n_sessions):
    """Run one replication with regime shifts (arm permutation between sessions).

    Arms means are randomly permuted at each session boundary.
    Agent state persists across sessions.
    Returns choices_log, rewards_log, n_arms, session_boundaries.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    random.seed(seed)

    base_means = [m for m, _ in arms_config]
    sds = [sd for _, sd in arms_config]
    n_arms = len(arms_config)

    agents = make_agents(strategy_name, n_agents, n_arms)

    all_choices = []
    all_rewards = []
    boundaries = []

    for session in range(n_sessions):
        permuted_means = base_means.copy()
        rng.shuffle(permuted_means)
        arms = [Arm(mean=m, sd=sds[i]) for i, m in enumerate(permuted_means)]
        env = Environment(n_agents=n_agents, arms=arms, collision_policy=policy_fn)
        boundaries.append(len(all_choices))

        for t in range(steps_per_session):
            choices, rewards = env.step(agents)
            all_choices.append(choices)
            all_rewards.append(rewards)

    return all_choices, all_rewards, n_arms, boundaries, base_means


def compute_seed_metrics(choices_log, rewards_log, n_arms, n_agents, steps,
                         oracle_mean):
    """Compute all metrics for a single seed. Returns dict of scalar metrics."""
    welf = market_welfare(rewards_log)
    avg_welfare = sum(welf) / steps
    crate = collision_rate(choices_log)

    entropies = [
        arm_selection_entropy(choices_log, n_arms, agent_idx=i)
        for i in range(n_agents)
    ]
    avg_entropy = sum(entropies) / n_agents

    per_agent_total = [
        sum(rewards_log[t][i] for t in range(steps))
        for i in range(n_agents)
    ]
    total_reward = sum(per_agent_total)

    cumulative_regret = sum(oracle_mean - rewards_log[t][0] for t in range(steps))

    return {
        "avg_welfare": avg_welfare,
        "collision_rate": crate,
        "avg_entropy": avg_entropy,
        "total_reward": total_reward,
        "cumulative_regret": cumulative_regret,
        "per_agent_total": per_agent_total,
    }


def phase_collision_rate(choices_log, start, end):
    """Collision rate over a slice of timesteps."""
    subset = choices_log[start:end]
    if not subset:
        return 0.0
    n_collisions = sum(
        1 for choices in subset
        if len(set(choices)) < len(choices)
    )
    return n_collisions / len(subset)


def late_phase_arm_distribution(choices_log, n_agents, n_arms, start):
    """Per-agent arm selection frequencies in the late phase."""
    late = choices_log[start:]
    dists = []
    for i in range(n_agents):
        counts = [0] * n_arms
        for step in late:
            counts[step[i]] += 1
        total = sum(counts)
        dists.append([c / total for c in counts])
    return dists
