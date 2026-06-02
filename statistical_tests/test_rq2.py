"""
RQ2: Do learning agents spontaneously specialise without communication?

Statistical design
------------------
- 2 UCB agents, 5 arms (wide gap: mu 1-5), 3000 steps, 30 seeds.
- Two conditions: zero_on_collision vs linear_share.
- Metrics:
    - Early-phase collision rate (first 25% of steps)
    - Late-phase collision rate (last 25% of steps)
    - Collision drop (early - late): primary measure of learned diversification
    - Specialisation index: whether agents anchor on different top arms
- Omnibus: not needed (only 2 groups).
- Hypothesis test: Mann-Whitney U on collision-drop between policies.
- Secondary: within-policy Wilcoxon signed-rank test (early vs late paired).
"""

import numpy as np
from scipy import stats as sp_stats

from statistical_tests.harness import (
    ARMS_WIDE, POLICY_MAP, run_single_seed,
    phase_collision_rate, late_phase_arm_distribution,
)
from statistical_tests.stat_utils import (
    ci_95, mann_whitney, format_p, format_ci, describe_effect,
)


N_AGENTS = 2
STEPS = 3000
SEEDS = list(range(30))
POLICIES = ["zero_on_collision", "linear_share"]


def specialisation_index(arm_dist):
    """Measure of agent specialisation: 1 if agents have distinct top arms, 0 if same."""
    top_arms = [dist.index(max(dist)) for dist in arm_dist]
    return 1.0 if len(set(top_arms)) == len(top_arms) else 0.0


def run():
    """Execute RQ2 experiment and return structured results."""
    print("=" * 70)
    print("RQ2: Emergent Diversification Without Communication")
    print(f"  N_agents={N_AGENTS}, K=5 arms (wide), T={STEPS}, seeds={len(SEEDS)}")
    print("=" * 70)

    n_arms = len(ARMS_WIDE)
    early_end = STEPS // 4
    late_start = 3 * STEPS // 4

    policy_data = {}
    for policy_name in POLICIES:
        policy_fn = POLICY_MAP[policy_name]
        early_cr_vec, late_cr_vec, drop_vec, spec_vec = [], [], [], []

        for seed in SEEDS:
            cl, rl, _ = run_single_seed(
                seed, ARMS_WIDE, policy_fn, "UCB", N_AGENTS, STEPS
            )
            early_cr = phase_collision_rate(cl, 0, early_end)
            late_cr = phase_collision_rate(cl, late_start, STEPS)
            drop = early_cr - late_cr
            arm_dist = late_phase_arm_distribution(cl, N_AGENTS, n_arms, late_start)
            spec = specialisation_index(arm_dist)

            early_cr_vec.append(early_cr)
            late_cr_vec.append(late_cr)
            drop_vec.append(drop)
            spec_vec.append(spec)

        policy_data[policy_name] = {
            "early_cr": np.array(early_cr_vec),
            "late_cr": np.array(late_cr_vec),
            "collision_drop": np.array(drop_vec),
            "specialisation": np.array(spec_vec),
        }

    results = {"descriptive": {}, "between_policy": {}, "within_policy": {}}

    # ── Descriptive statistics ───────────────────────────────────────────
    print("\n--- Descriptive Statistics (mean ± 95% CI) ---")
    header = f"{'Policy':<22} {'Early CR':>14} {'Late CR':>14} {'Drop':>14} {'Spec %':>10}"
    print(header)
    print("-" * 76)
    for p in POLICIES:
        d = policy_data[p]
        e_m, e_ci, _ = ci_95(d["early_cr"])
        l_m, l_ci, _ = ci_95(d["late_cr"])
        dr_m, dr_ci, _ = ci_95(d["collision_drop"])
        sp_m = d["specialisation"].mean() * 100
        print(f"{p:<22} {format_ci(e_m * 100, e_ci * 100):>14}% "
              f"{format_ci(l_m * 100, l_ci * 100):>14}% "
              f"{format_ci(dr_m * 100, dr_ci * 100):>14}pp "
              f"{sp_m:>9.1f}%")
        results["descriptive"][p] = {
            "early_cr": {"mean": e_m, "ci": e_ci},
            "late_cr": {"mean": l_m, "ci": l_ci},
            "collision_drop": {"mean": dr_m, "ci": dr_ci},
            "specialisation_rate": sp_m,
        }

    # ── Between-policy: Mann-Whitney on collision drop ────────────────────
    print("\n--- Between-Policy Comparison (Mann-Whitney U) ---")
    for metric in ["collision_drop", "specialisation"]:
        x = policy_data["zero_on_collision"][metric]
        y = policy_data["linear_share"][metric]
        mw = mann_whitney(x, y, alternative="two-sided")
        eff_str = describe_effect(mw["r_effect"])
        print(f"  {metric:<22} U={mw['U']:.0f}, z={mw['z']:.2f}, "
              f"{format_p(mw['p'])}, r={mw['r_effect']:.3f} ({eff_str})")
        results["between_policy"][metric] = mw

    # ── Within-policy: Wilcoxon signed-rank (early vs late, paired by seed) ──
    print("\n--- Within-Policy Paired Tests (Wilcoxon signed-rank: early vs late CR) ---")
    for p in POLICIES:
        d = policy_data[p]
        diffs = d["early_cr"] - d["late_cr"]
        non_zero = diffs[diffs != 0]
        if len(non_zero) < 2:
            print(f"  {p:<22} insufficient non-zero differences for Wilcoxon test")
            results["within_policy"][p] = {
                "note": "all differences are zero or near-zero"
            }
            continue
        w_stat, w_p = sp_stats.wilcoxon(d["early_cr"], d["late_cr"],
                                         alternative="greater")
        print(f"  {p:<22} W={w_stat:.0f}, {format_p(w_p)} "
              f"(H1: early CR > late CR)")
        results["within_policy"][p] = {
            "W": float(w_stat),
            "p": float(w_p),
        }

    results["raw_data"] = policy_data
    return results


if __name__ == "__main__":
    run()
