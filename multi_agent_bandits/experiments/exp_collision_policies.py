"""
RQ1: How does the pool's collision mechanism affect total market welfare
     and individual agent performance?

Experimental setup
------------------
- 5 arms (dark pools) with heterogeneous execution quality (mean fill improvement).
- 3 agents (institutional investors / smart order routers) running UCB.
- Collision policy is the only variable across scenarios.
- Results are averaged over multiple random seeds.

The three collision policies and their dark pool interpretations
---------------------------------------------------------------
linear_share     → Pro-rata allocation (e.g., IEX, some crossing networks)
                   Multiple institutions hit the same pool → each gets a
                   proportional fill. Welfare is split, not destroyed.
                   Agents face no penalty for herding → expect low entropy,
                   high collision rate, welfare preserved.

zero_on_collision → Toxic pool / information-leakage detection
                   Pool detects simultaneous large orders and cancels all
                   fills. Models adverse selection: pool protects its liquidity
                   providers from crowded, predictable flow.
                   Strong penalty for herding → expect agents to learn to
                   diversify, collision rate should fall over time, but
                   early-phase welfare loss can be severe.

winner_takes_all  → Price-time priority / latency competition
                   One randomly-chosen agent captures the full fill; others
                   get nothing. Expected welfare per step equals linear_share
                   in expectation, but individual outcomes are a lottery.
                   Models pools where speed is the tie-breaker.

Key metrics
-----------
- Market welfare: total reward realized per step across all agents
- Collision rate: fraction of steps with any overlap
- Per-agent cumulative reward: individual performance
- Final arm entropy: how diversified is each agent's arm selection?
"""

import random
import math

from multi_agent_bandits.core.environment import Environment
from multi_agent_bandits.core.experiment_runner import ExperimentRunner
from multi_agent_bandits.core.arm import Arm
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


# 5 dark pools with increasing execution quality.
# Mean = expected fill improvement (e.g. price improvement in bps).
# SD  = liquidity uncertainty (pool depth variability).
ARMS_CONFIG = [
    (1.0, 1.0),   # Pool 0: low quality, stable
    (2.0, 1.2),   # Pool 1
    (3.0, 1.0),   # Pool 2: mid quality
    (4.0, 1.5),   # Pool 3: good quality, higher uncertainty
    (5.0, 1.0),   # Pool 4: best quality, stable (oracle arm)
]

ORACLE_MEAN = max(m for m, _ in ARMS_CONFIG)


def _run_seed(policy_fn, n_agents, steps, seed):
    """Run one replication and return raw logs."""
    random.seed(seed)
    arms = [Arm(mean=m, sd=sd) for m, sd in ARMS_CONFIG]
    env = Environment(n_agents=n_agents, arms=arms, collision_policy=policy_fn)
    agents = [UCB_BaselineAgent(env.n_arms) for _ in range(n_agents)]
    runner = ExperimentRunner(env, agents, timestep_limit=steps)
    choices_log, rewards_log = runner.run()
    return choices_log, rewards_log, env.n_arms


def _aggregate(policy_fn, n_agents, steps, seeds):
    """Average key metrics across seeds."""
    all_avg_welfare = []
    all_crate = []
    all_entropies = []
    all_per_agent = []

    n_arms = len(ARMS_CONFIG)

    for seed in seeds:
        choices_log, rewards_log, _ = _run_seed(policy_fn, n_agents, steps, seed)

        welf = market_welfare(rewards_log)
        all_avg_welfare.append(sum(welf) / steps)
        all_crate.append(collision_rate(choices_log))

        entropies = [
            arm_selection_entropy(choices_log, n_arms, agent_idx=i)
            for i in range(n_agents)
        ]
        all_entropies.append(entropies)

        per_agent = [
            sum(rewards_log[t][i] for t in range(steps))
            for i in range(n_agents)
        ]
        all_per_agent.append(per_agent)

    n_seeds = len(seeds)

    return {
        "avg_welfare_per_step": sum(all_avg_welfare) / n_seeds,
        "collision_rate": sum(all_crate) / n_seeds,
        "avg_entropy": [
            sum(all_entropies[s][i] for s in range(n_seeds)) / n_seeds
            for i in range(n_agents)
        ],
        "avg_total_per_agent": [
            sum(all_per_agent[s][i] for s in range(n_seeds)) / n_seeds
            for i in range(n_agents)
        ],
    }


def main(steps=2000, n_agents=3, seeds=None, save_dir=None, **_):
    """
    Run RQ1 across all three collision policies and print a comparative summary.

    Parameters
    ----------
    steps     : timesteps per run
    n_agents  : number of competing institutions
    seeds     : list of int seeds (default: 5 seeds for stable averages)
    """
    if seeds is None:
        seeds = [42, 123, 7, 99, 555]

    policies = [
        ("linear_share",      linear_share),
        ("zero_on_collision", zero_on_collision),
        ("winner_takes_all",  winner_takes_all),
    ]

    n_arms = len(ARMS_CONFIG)
    max_entropy = math.log(n_arms)

    print(f"\n{'='*65}")
    print("RQ1: Collision Policy and Market Efficiency")
    print(f"{'='*65}")
    print(f"Arms: {n_arms} dark pools | Agents: {n_agents} | Steps: {steps} | Seeds: {len(seeds)}")
    print(f"Strategy: UCB | Oracle arm mean: {ORACLE_MEAN:.1f}")
    print(f"Max possible entropy (uniform over arms): {max_entropy:.3f}")

    results = {}

    for policy_name, policy_fn in policies:
        print(f"\n{'─'*65}")
        print(f"Policy: {policy_name}")
        r = _aggregate(policy_fn, n_agents, steps, seeds)
        results[policy_name] = r

        avg_ent = sum(r["avg_entropy"]) / n_agents

        print(f"  Market welfare (avg total reward/step): {r['avg_welfare_per_step']:.3f}")
        print(f"  Collision rate:                         {r['collision_rate']:.1%}")
        print(f"  Per-agent cumulative reward (avg):")
        for i, total in enumerate(r["avg_total_per_agent"]):
            print(f"    Agent {i}: {total:.1f}")
        print(f"  Avg arm entropy per agent: {avg_ent:.3f}  (max = {max_entropy:.3f})")

    # ── Cross-policy summary table ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Cross-policy summary")
    print(f"{'='*65}")
    header = f"{'Policy':<22}  {'Welfare/step':>13}  {'Collision%':>11}  {'Entropy':>9}  {'Entropy%':>9}"
    print(header)
    print(f"{'─'*65}")
    for policy_name, r in results.items():
        avg_ent = sum(r["avg_entropy"]) / n_agents
        pct = avg_ent / max_entropy * 100
        print(
            f"{policy_name:<22}  {r['avg_welfare_per_step']:>13.3f}"
            f"  {r['collision_rate']:>10.1%}  {avg_ent:>9.3f}  {pct:>8.1f}%"
        )

    # ── Interpretation guide ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Reading the results")
    print(f"{'='*65}")
    print(
        "Welfare/step: how much execution quality the market realised on average.\n"
        "  linear_share ≈ winner_takes_all in expectation (welfare preserved).\n"
        "  zero_on_collision will be lower — collisions destroy value entirely.\n"
        "\n"
        "Collision rate: how often agents chose the same pool.\n"
        "  Should fall over time under zero_on_collision as agents learn to spread.\n"
        "  May stay high under linear_share — herding is rational when it's costless.\n"
        "\n"
        "Entropy: how diversified each agent's venue selection became.\n"
        "  Low entropy (≈0) = agent concentrated on one pool (optimal under linear_share).\n"
        "  High entropy (→ log(n_arms)) = agent spreading across venues.\n"
        "  Under zero_on_collision, emergent diversification without communication\n"
        "  is the key phenomenon linking this to RQ2."
    )

    return results
