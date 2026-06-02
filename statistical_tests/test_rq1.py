"""
RQ1: How does the collision mechanism affect market welfare?

Statistical design
------------------
- 3 UCB agents, 5 arms (wide gap: mu 1-5), 2000 steps, 30 seeds.
- Three conditions: linear_share, zero_on_collision, winner_takes_all.
- Metrics: avg welfare/step, collision rate, cumulative regret.
- Omnibus: Kruskal-Wallis H test across the three policies.
- Pairwise: Mann-Whitney U with Holm-Bonferroni correction (3 comparisons).
- Effect size: rank-biserial r.
- All results report mean ± 95% CI (t-based).
"""

import numpy as np
from statistical_tests.harness import (
    ARMS_WIDE, POLICY_MAP, run_single_seed, compute_seed_metrics,
)
from statistical_tests.stat_utils import (
    ci_95, kruskal_wallis, pairwise_mann_whitney,
    format_p, format_ci, describe_effect,
)


N_AGENTS = 3
STEPS = 2000
SEEDS = list(range(30))
ORACLE_MEAN = max(m for m, _ in ARMS_WIDE)
POLICIES = ["linear_share", "zero_on_collision", "winner_takes_all"]


def run():
    """Execute RQ1 experiment and return structured results."""
    print("=" * 70)
    print("RQ1: Collision Policies and Market Welfare")
    print(f"  N_agents={N_AGENTS}, K=5 arms (wide), T={STEPS}, seeds={len(SEEDS)}")
    print("=" * 70)

    # Collect per-seed metrics for each policy
    policy_data = {}
    for policy_name in POLICIES:
        policy_fn = POLICY_MAP[policy_name]
        welfare_vec, crate_vec, regret_vec = [], [], []

        for seed in SEEDS:
            cl, rl, n_arms = run_single_seed(
                seed, ARMS_WIDE, policy_fn, "UCB", N_AGENTS, STEPS
            )
            m = compute_seed_metrics(cl, rl, n_arms, N_AGENTS, STEPS, ORACLE_MEAN)
            welfare_vec.append(m["avg_welfare"])
            crate_vec.append(m["collision_rate"])
            regret_vec.append(m["cumulative_regret"])

        policy_data[policy_name] = {
            "welfare": np.array(welfare_vec),
            "collision_rate": np.array(crate_vec),
            "regret": np.array(regret_vec),
        }

    results = {"descriptive": {}, "omnibus": {}, "pairwise": {}}

    # ── Descriptive statistics ───────────────────────────────────────────
    print("\n--- Descriptive Statistics (mean ± 95% CI) ---")
    header = f"{'Policy':<22} {'Welfare/step':>16} {'Collision %':>16} {'Regret':>16}"
    print(header)
    print("-" * 72)
    for p in POLICIES:
        d = policy_data[p]
        w_m, w_ci, _ = ci_95(d["welfare"])
        c_m, c_ci, _ = ci_95(d["collision_rate"])
        r_m, r_ci, _ = ci_95(d["regret"])
        print(f"{p:<22} {format_ci(w_m, w_ci):>16} "
              f"{format_ci(c_m * 100, c_ci * 100):>16}% "
              f"{format_ci(r_m, r_ci):>16}")
        results["descriptive"][p] = {
            "welfare": {"mean": w_m, "ci": w_ci},
            "collision_rate": {"mean": c_m, "ci": c_ci},
            "regret": {"mean": r_m, "ci": r_ci},
        }

    # ── Omnibus tests ────────────────────────────────────────────────────
    print("\n--- Omnibus Tests (Kruskal-Wallis H) ---")
    for metric in ["welfare", "collision_rate", "regret"]:
        groups = [policy_data[p][metric] for p in POLICIES]
        kw = kruskal_wallis(*groups)
        results["omnibus"][metric] = kw
        print(f"  {metric:<20} H({kw['df']}) = {kw['H']:.3f}, {format_p(kw['p'])}")

    # ── Pairwise comparisons ─────────────────────────────────────────────
    print("\n--- Pairwise Comparisons (Mann-Whitney U, Holm-Bonferroni corrected) ---")
    for metric in ["welfare", "collision_rate", "regret"]:
        groups = {p: policy_data[p][metric] for p in POLICIES}
        pw = pairwise_mann_whitney(groups, metric_name=metric)
        results["pairwise"][metric] = pw
        print(f"\n  {metric}:")
        for comp in pw:
            sig_str = "*" if comp["significant"] else "ns"
            eff_str = describe_effect(comp["r_effect"])
            print(f"    {comp['label']:<40} U={comp['U']:.0f}, z={comp['z']:.2f}, "
                  f"{format_p(comp['p_adjusted'])} (adj), r={comp['r_effect']:.3f} ({eff_str}) [{sig_str}]")

    results["raw_data"] = policy_data
    return results


if __name__ == "__main__":
    run()
