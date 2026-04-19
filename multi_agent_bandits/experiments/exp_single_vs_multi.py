"""
RQ2 (foundation): Does emergent diversification appear without communication?

This experiment asks whether learning agents spontaneously learn to spread
across venues simply as a consequence of the collision structure — no explicit
coordination, no shared information.

The core mechanism
------------------
Under zero_on_collision: when two agents pick the same pool and both get
zero reward, each agent's estimated value for that arm is pulled down toward
zero. Over time, the best-arm estimate is "poisoned" by collisions, and the
second-best arm starts to look relatively more attractive. If this happens
symmetrically, one agent ends up anchoring on the best pool and the other
drifts to the second-best — a Nash-like separation emerges from pure learning.

Under linear_share: collisions only split the reward, not destroy it. The
best arm's estimate stays high even under collision. Both agents keep learning
that it's the best arm, and herding persists — there's no pressure to diversify.

What to look for
----------------
- Rolling entropy rising over time → diversification is emerging.
- Collision rate falling over time → agents are learning to avoid each other.
- Per-agent rewards converging to different levels → specialisation.
- Compare early-phase (first 25% of steps) vs late-phase (last 25%): if
  late-phase collision rate is substantially lower than early-phase under
  zero_on_collision but not under linear_share, emergent diversification
  is confirmed.
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
    rolling_arm_entropy,
)
from multi_agent_bandits.core.reward_sharing import linear_share, zero_on_collision
from multi_agent_bandits.strategies.ucb_baseline import UCB_BaselineAgent


ARMS_CONFIG = [
    (1.0, 1.0),
    (2.0, 1.2),
    (3.0, 1.0),
    (4.0, 1.5),
    (5.0, 1.0),
]


def _phase_collision_rate(choices_log, start, end):
    """Collision rate restricted to timesteps [start, end)."""
    subset = choices_log[start:end]
    if not subset:
        return 0.0
    n_collisions = 0
    for choices in subset:
        arm_counts = {}
        for arm in choices:
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
        if any(v > 1 for v in arm_counts.values()):
            n_collisions += 1
    return n_collisions / len(subset)


def _run_seed(policy_fn, n_agents, steps, seed):
    random.seed(seed)
    arms = [Arm(mean=m, sd=sd) for m, sd in ARMS_CONFIG]
    env = Environment(n_agents=n_agents, arms=arms, collision_policy=policy_fn)
    agents = [UCB_BaselineAgent(env.n_arms) for _ in range(n_agents)]
    runner = ExperimentRunner(env, agents, timestep_limit=steps)
    choices_log, rewards_log = runner.run()
    return choices_log, rewards_log, env.n_arms


def _analyse(choices_log, rewards_log, n_agents, n_arms, steps, window=200):
    early_end = steps // 4
    late_start = 3 * steps // 4

    early_crate = _phase_collision_rate(choices_log, 0, early_end)
    late_crate = _phase_collision_rate(choices_log, late_start, steps)

    final_entropies = [
        arm_selection_entropy(choices_log, n_arms, agent_idx=i)
        for i in range(n_agents)
    ]

    # Per-agent arm distribution in the final quarter (revealed preference)
    final_choices = choices_log[late_start:]
    arm_dist = []
    for i in range(n_agents):
        counts = [0] * n_arms
        for step in final_choices:
            counts[step[i]] += 1
        total = sum(counts)
        arm_dist.append([c / total for c in counts])

    return {
        "early_collision_rate": early_crate,
        "late_collision_rate": late_crate,
        "collision_drop": early_crate - late_crate,
        "final_entropy": final_entropies,
        "arm_distribution_late": arm_dist,
    }


def _avg_analysis(policy_fn, n_agents, steps, seeds):
    n_arms = len(ARMS_CONFIG)
    analyses = []
    for seed in seeds:
        choices_log, rewards_log, _ = _run_seed(policy_fn, n_agents, steps, seed)
        analyses.append(_analyse(choices_log, rewards_log, n_agents, n_arms, steps))

    n_seeds = len(seeds)

    def avg(key):
        return sum(a[key] for a in analyses) / n_seeds

    avg_arm_dist = [
        [
            sum(analyses[s]["arm_distribution_late"][i][arm] for s in range(n_seeds)) / n_seeds
            for arm in range(n_arms)
        ]
        for i in range(n_agents)
    ]

    return {
        "early_collision_rate": avg("early_collision_rate"),
        "late_collision_rate": avg("late_collision_rate"),
        "collision_drop": avg("collision_drop"),
        "final_entropy": [
            sum(analyses[s]["final_entropy"][i] for s in range(n_seeds)) / n_seeds
            for i in range(n_agents)
        ],
        "arm_distribution_late": avg_arm_dist,
    }


def main(steps=3000, n_agents=2, seeds=None, **_):
    """
    Compare emergent diversification under zero_on_collision vs linear_share.

    Uses 2 agents to make the specialisation signal clean: with 2 agents and
    5 arms, perfect specialisation means each agent anchors on a different arm
    (no overlap). With 3+ agents the picture is richer but noisier.

    Parameters
    ----------
    steps    : timesteps per run (longer = clearer late-phase signal)
    n_agents : number of competing agents (default 2 for clear separation)
    seeds    : list of int seeds
    """
    if seeds is None:
        seeds = [42, 123, 7, 99, 555]

    n_arms = len(ARMS_CONFIG)
    max_entropy = math.log(n_arms)

    scenarios = [
        ("zero_on_collision", zero_on_collision,
         "Agents penalised for herding → diversification expected"),
        ("linear_share", linear_share,
         "Herding is costless → diversification not expected"),
    ]

    print(f"\n{'='*65}")
    print("RQ2 Foundation: Emergent Diversification Without Communication")
    print(f"{'='*65}")
    print(f"Arms: {n_arms} | Agents: {n_agents} | Steps: {steps} | Seeds: {len(seeds)}")
    print(f"Strategy: UCB | Max entropy: {max_entropy:.3f}")
    print(
        "\nDiversification signal: if late-phase collision rate << early-phase,\n"
        "agents learned to avoid each other purely from bandit feedback."
    )

    all_results = {}

    for policy_name, policy_fn, description in scenarios:
        print(f"\n{'─'*65}")
        print(f"Policy: {policy_name}")
        print(f"  {description}")

        r = _avg_analysis(policy_fn, n_agents, steps, seeds)
        all_results[policy_name] = r

        avg_ent = sum(r["final_entropy"]) / n_agents
        drop_pct = r["collision_drop"] / max(r["early_collision_rate"], 1e-9) * 100

        print(f"\n  Collision rate — early phase (first 25%): {r['early_collision_rate']:.1%}")
        print(f"  Collision rate — late  phase (last  25%): {r['late_collision_rate']:.1%}")
        print(f"  Drop:                                      {r['collision_drop']:+.1%}  ({drop_pct:.0f}% relative reduction)")
        print(f"  Final entropy per agent: {avg_ent:.3f}  (max = {max_entropy:.3f})")
        print(f"\n  Late-phase venue distribution per agent:")
        for i, dist in enumerate(r["arm_distribution_late"]):
            bar = " ".join(f"{p:.2f}" for p in dist)
            top_arm = dist.index(max(dist))
            print(f"    Agent {i}: [{bar}]  → top pool: {top_arm}")

    # ── Comparison ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Did diversification emerge?")
    print(f"{'='*65}")
    for policy_name, r in all_results.items():
        # Specialisation: agents anchoring on DIFFERENT top arms is the core signal.
        # Herding = all agents converge on the same arm (top arm identical).
        top_arms = [dist.index(max(dist)) for dist in r["arm_distribution_late"]]
        all_same = len(set(top_arms)) == 1
        late_crate = r["late_collision_rate"]

        if not all_same and late_crate < 0.05:
            emerged = f"YES — agents specialised on different pools {top_arms}"
        elif not all_same:
            emerged = f"PARTIAL — different top pools {top_arms} but still colliding {late_crate:.1%}"
        else:
            emerged = f"NO  — all agents herd on pool {top_arms[0]}"
        print(f"  {policy_name:<22}: {emerged}")

    print(
        "\nNote: the zero_on_collision result is the core RQ2 finding."
        "\nNext step: add communication mechanisms (observable actions, shared"
        "\nestimates, noisy signals) and test whether they accelerate, change,"
        "\nor replace this emergent coordination."
    )

    return all_results
