"""
RQ3: Does adaptive SOR outperform static allocation under regime shifts?

Statistical design
------------------
- 1 agent, 5 arms (wide gap: mu 1-5), 6 sessions x 500 steps = 3000 total.
- Arm means permuted between sessions (non-stationary).
- 7 strategies: UCB, TS, SW-UCB, KL-UCB, EG(0.05), Fixed(best), Random.
- Metric: cumulative regret over all sessions.
- Omnibus: Kruskal-Wallis across all strategies.
- Primary comparison: SW-UCB vs each other strategy (Mann-Whitney, Holm-Bonferroni).
- Secondary: all adaptive vs static pairwise.
- 30 seeds, 95% CI, rank-biserial effect size.
"""

import numpy as np

from statistical_tests.harness import (
    ARMS_WIDE, POLICY_MAP, run_single_seed_regime,
)
from statistical_tests.stat_utils import (
    ci_95, kruskal_wallis, pairwise_mann_whitney, mann_whitney,
    holm_bonferroni, format_p, format_ci, describe_effect,
)


N_AGENTS = 1
STEPS_PER_SESSION = 500
N_SESSIONS = 6
TOTAL_STEPS = STEPS_PER_SESSION * N_SESSIONS
SEEDS = list(range(30))
ORACLE_MEAN = max(m for m, _ in ARMS_WIDE)
POLICY_NAME = "linear_share"

STRATEGIES = ["UCB", "TS", "SW-UCB", "KL-UCB", "EG(0.05)", "Fixed(best)", "Random"]


def run():
    """Execute RQ3 experiment and return structured results."""
    print("=" * 70)
    print("RQ3: Non-Stationary Adaptation (Regime Shifts)")
    print(f"  N_agents={N_AGENTS}, K=5 arms (wide), "
          f"{N_SESSIONS} sessions x {STEPS_PER_SESSION} steps = {TOTAL_STEPS} total, "
          f"seeds={len(SEEDS)}")
    print("=" * 70)

    policy_fn = POLICY_MAP[POLICY_NAME]
    strategy_data = {}

    for strat in STRATEGIES:
        regret_vec = []
        for seed in SEEDS:
            cl, rl, n_arms, boundaries, base_means = run_single_seed_regime(
                seed, ARMS_WIDE, policy_fn, strat, N_AGENTS,
                STEPS_PER_SESSION, N_SESSIONS,
            )
            regret = sum(ORACLE_MEAN - rl[t][0] for t in range(len(rl)))
            regret_vec.append(regret)
        strategy_data[strat] = np.array(regret_vec)

    results = {"descriptive": {}, "omnibus": {}, "sw_ucb_vs_all": {},
               "pairwise": {}}

    # ── Descriptive statistics ───────────────────────────────────────────
    print("\n--- Descriptive Statistics (mean ± 95% CI) ---")
    header = f"{'Strategy':<14} {'Total Regret':>18} {'vs SW-UCB':>12}"
    print(header)
    print("-" * 46)

    sw_ucb_mean, _, _ = ci_95(strategy_data["SW-UCB"])
    for strat in STRATEGIES:
        m, ci, _ = ci_95(strategy_data[strat])
        if strat == "SW-UCB":
            vs = "---"
        else:
            pct = ((m - sw_ucb_mean) / sw_ucb_mean) * 100
            vs = f"+{pct:.0f}%"
        print(f"  {strat:<14} {format_ci(m, ci):>18} {vs:>12}")
        results["descriptive"][strat] = {"mean": m, "ci": ci}

    # ── Omnibus ──────────────────────────────────────────────────────────
    print("\n--- Omnibus Test (Kruskal-Wallis H) ---")
    groups = [strategy_data[s] for s in STRATEGIES]
    kw = kruskal_wallis(*groups)
    results["omnibus"] = kw
    print(f"  H({kw['df']}) = {kw['H']:.3f}, {format_p(kw['p'])}")

    # ── SW-UCB vs each other strategy ─────────────────────────────────────
    print("\n--- SW-UCB vs Each Strategy (Mann-Whitney U, Holm-Bonferroni) ---")
    raw_comparisons = []
    for strat in STRATEGIES:
        if strat == "SW-UCB":
            continue
        mw = mann_whitney(strategy_data["SW-UCB"], strategy_data[strat],
                          alternative="two-sided")
        label = f"SW-UCB vs {strat}"
        raw_comparisons.append({"label": label, **mw})

    p_list = [(r["label"], r["p"]) for r in raw_comparisons]
    corrected = holm_bonferroni(p_list)
    correction_map = {label: (adj_p, sig) for label, _, adj_p, sig in corrected}
    for r in raw_comparisons:
        adj_p, sig = correction_map[r["label"]]
        r["p_adjusted"] = adj_p
        r["significant"] = sig

    for comp in raw_comparisons:
        sig_str = "*" if comp["significant"] else "ns"
        eff_str = describe_effect(comp["r_effect"])
        print(f"  {comp['label']:<26} U={comp['U']:.0f}, z={comp['z']:.2f}, "
              f"{format_p(comp['p_adjusted'])} (adj), r={comp['r_effect']:.3f} ({eff_str}) [{sig_str}]")
    results["sw_ucb_vs_all"] = raw_comparisons

    # ── SW-UCB vs UCB specifically (the 58% claim) ────────────────────────
    sw_m, _, _ = ci_95(strategy_data["SW-UCB"])
    ucb_m, _, _ = ci_95(strategy_data["UCB"])
    reduction_pct = ((ucb_m - sw_m) / ucb_m) * 100
    print(f"\n--- Key Comparison ---")
    print(f"  SW-UCB regret: {sw_m:.1f}")
    print(f"  UCB regret:    {ucb_m:.1f}")
    print(f"  Reduction:     {reduction_pct:.1f}%")
    results["sw_ucb_vs_ucb_reduction_pct"] = reduction_pct

    # ── Full pairwise for completeness ───────────────────────────────────
    print("\n--- Full Pairwise (all strategies, Mann-Whitney U, Holm-Bonferroni) ---")
    pw = pairwise_mann_whitney(strategy_data, metric_name="regret")
    results["pairwise"] = pw
    sig_count = sum(1 for c in pw if c["significant"])
    print(f"  {len(pw)} comparisons, {sig_count} significant after correction")

    results["raw_data"] = strategy_data
    return results


if __name__ == "__main__":
    run()
