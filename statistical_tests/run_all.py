#!/usr/bin/env python3
"""
Master runner: executes all statistical tests and writes findings.md.

Usage:
    python -m statistical_tests.run_all

Output:
    statistical_tests/findings.md   -- thesis-ready results with guidance
    stdout                          -- full test output for verification
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statistical_tests import test_rq1, test_rq2, test_rq3
from statistical_tests.stat_utils import (
    format_p, format_ci, describe_effect, ci_95,
)


def write_findings(rq1, rq2, rq3, elapsed):
    """Generate findings.md from real experimental results."""

    lines = []
    def w(s=""):
        lines.append(s)

    # ── Header ───────────────────────────────────────────────────────────
    w("# Statistical Findings -- Multi-Agent Bandit Framework")
    w()
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"Runtime: {elapsed:.0f} seconds")
    w()
    w("All p-values are from non-parametric tests (Mann-Whitney U or Kruskal-Wallis).")
    w("Confidence intervals are 95% (t-based, n=30 seeds).")
    w("Multiple comparisons corrected with Holm-Bonferroni.")
    w("Effect sizes are rank-biserial r (|r| > 0.5 = large).")
    w()
    w("---")
    w()

    # ══════════════════════════════════════════════════════════════════════
    # RQ1
    # ══════════════════════════════════════════════════════════════════════
    w("## RQ1: Collision Policies and Market Welfare")
    w()
    w("**Setup:** 3 UCB agents, 5 arms (mu 1-5), 2,000 steps, 30 seeds.")
    w()

    # Descriptive table
    w("### Table 1: Cross-policy summary (30 seeds, mean ± 95% CI)")
    w()
    w("| Policy | Welfare/step | Collision % | Cumulative Regret |")
    w("|--------|-------------|-------------|-------------------|")
    for p in ["linear_share", "zero_on_collision", "winner_takes_all"]:
        d = rq1["descriptive"][p]
        w_str = format_ci(d["welfare"]["mean"], d["welfare"]["ci"])
        c_str = format_ci(d["collision_rate"]["mean"] * 100, d["collision_rate"]["ci"] * 100)
        r_str = format_ci(d["regret"]["mean"], d["regret"]["ci"])
        w(f"| {p} | {w_str} | {c_str}% | {r_str} |")
    w()

    # Omnibus
    w("### Omnibus tests (Kruskal-Wallis)")
    w()
    for metric in ["welfare", "collision_rate", "regret"]:
        kw = rq1["omnibus"][metric]
        w(f"- **{metric}**: H({kw['df']}) = {kw['H']:.3f}, {format_p(kw['p'])}")
    w()

    # Pairwise
    w("### Pairwise comparisons (Mann-Whitney U, Holm-Bonferroni adjusted)")
    w()
    w("| Comparison | Metric | U | z | p (adjusted) | r | Effect |")
    w("|-----------|--------|---|---|-------------|---|--------|")
    for metric in ["welfare", "collision_rate", "regret"]:
        for comp in rq1["pairwise"][metric]:
            sig = "\\*" if comp["significant"] else "ns"
            w(f"| {comp['label']} | {metric} | {comp['U']:.0f} | {comp['z']:.2f} | "
              f"{format_p(comp['p_adjusted'])} | {comp['r_effect']:.3f} | "
              f"{describe_effect(comp['r_effect'])} {sig} |")
    w()

    # RQ1 thesis guidance
    w("### Thesis writing guidance (RQ1)")
    w()
    w("**Section 5 (RQ1: Collision Policies and Market Welfare)**")
    w()
    w("In your **Setup** subsection, state:")
    w("> N = 3 UCB agents, K = 5 arms (wide gap: μ ∈ {1, 2, 3, 4, 5}), ")
    w("> T = 2,000 steps, 30 seeds. All results report mean ± 95% CI. ")
    w("> Mann-Whitney U tests assess pairwise significance with Holm-Bonferroni ")
    w("> correction for multiple comparisons (3 pairwise tests per metric).")
    w()
    w("In your **Results** subsection, report the numbers from Table 1 above, then write:")
    w()

    # Build the actual result sentences from real data
    kw_welfare = rq1["omnibus"]["welfare"]
    w(f"> The omnibus Kruskal-Wallis test confirms a significant effect of collision ")
    w(f"> policy on welfare (H({kw_welfare['df']}) = {kw_welfare['H']:.3f}, {format_p(kw_welfare['p'])}).")
    w()

    # Find the significant/non-significant pairs
    welfare_pairs = rq1["pairwise"]["welfare"]
    for comp in welfare_pairs:
        d1_name = comp["label"].split(" vs ")[0]
        d2_name = comp["label"].split(" vs ")[1]
        d1 = rq1["descriptive"][d1_name]["welfare"]
        d2 = rq1["descriptive"][d2_name]["welfare"]
        direction = "higher" if d1["mean"] > d2["mean"] else "lower"
        w(f"> {comp['label']}: {d1_name} welfare ({d1['mean']:.2f} ± {d1['ci']:.2f}) "
          f"is {direction} than {d2_name} ({d2['mean']:.2f} ± {d2['ci']:.2f}); "
          f"U = {comp['U']:.0f}, z = {comp['z']:.2f}, {format_p(comp['p_adjusted'])} "
          f"(Holm-Bonferroni adjusted), r = {comp['r_effect']:.3f} ({describe_effect(comp['r_effect'])}).")
        w()

    w("**Key finding for the abstract:** State the omnibus result and the specific ")
    w("comparison that is most policy-relevant (typically zero_on_collision vs linear_share).")
    w("Only claim p < 0.001 if the adjusted p-value shown above actually is < 0.001.")
    w()
    w("---")
    w()

    # ══════════════════════════════════════════════════════════════════════
    # RQ2
    # ══════════════════════════════════════════════════════════════════════
    w("## RQ2: Emergent Diversification")
    w()
    w("**Setup:** 2 UCB agents, 5 arms (mu 1-5), 3,000 steps, 30 seeds.")
    w()

    # Descriptive
    w("### Table 2: Collision rate dynamics (30 seeds, mean ± 95% CI)")
    w()
    w("| Policy | Early CR | Late CR | Drop | Specialisation % |")
    w("|--------|----------|---------|------|------------------|")
    for p in ["zero_on_collision", "linear_share"]:
        d = rq2["descriptive"][p]
        e_str = format_ci(d["early_cr"]["mean"] * 100, d["early_cr"]["ci"] * 100)
        l_str = format_ci(d["late_cr"]["mean"] * 100, d["late_cr"]["ci"] * 100)
        dr_str = format_ci(d["collision_drop"]["mean"] * 100, d["collision_drop"]["ci"] * 100)
        sp = d["specialisation_rate"]
        w(f"| {p} | {e_str}% | {l_str}% | {dr_str}pp | {sp:.1f}% |")
    w()

    # Between-policy
    w("### Between-policy comparisons (Mann-Whitney U)")
    w()
    for metric in ["collision_drop", "specialisation"]:
        mw = rq2["between_policy"][metric]
        eff_str = describe_effect(mw["r_effect"])
        w(f"- **{metric}**: U = {mw['U']:.0f}, z = {mw['z']:.2f}, "
          f"{format_p(mw['p'])}, r = {mw['r_effect']:.3f} ({eff_str})")
    w()

    # Within-policy
    w("### Within-policy paired tests (Wilcoxon signed-rank: early > late CR)")
    w()
    for p in ["zero_on_collision", "linear_share"]:
        wp = rq2["within_policy"][p]
        if "note" in wp:
            w(f"- **{p}**: {wp['note']}")
        else:
            w(f"- **{p}**: W = {wp['W']:.0f}, {format_p(wp['p'])}")
    w()

    # RQ2 thesis guidance
    w("### Thesis writing guidance (RQ2)")
    w()
    w("**Section 6 (RQ2: Emergent Diversification)**")
    w()
    w("In your **Results** subsection, lead with the collision-drop comparison:")
    w()
    mw_drop = rq2["between_policy"]["collision_drop"]
    d_zero = rq2["descriptive"]["zero_on_collision"]
    d_lin = rq2["descriptive"]["linear_share"]
    w(f"> Under zero-on-collision, the collision rate dropped by "
      f"{d_zero['collision_drop']['mean'] * 100:.1f} ± {d_zero['collision_drop']['ci'] * 100:.1f} "
      f"percentage points from early to late phase, compared to "
      f"{d_lin['collision_drop']['mean'] * 100:.1f} ± {d_lin['collision_drop']['ci'] * 100:.1f}pp "
      f"under linear share. This difference is {'statistically significant' if mw_drop['p'] < 0.05 else 'not statistically significant'} "
      f"(Mann-Whitney U = {mw_drop['U']:.0f}, z = {mw_drop['z']:.2f}, {format_p(mw_drop['p'])}, "
      f"r = {mw_drop['r_effect']:.3f}).")
    w()
    w("Then report specialisation rates and the within-policy Wilcoxon results.")
    w("The specialisation index measures whether agents anchor on *different* top arms.")
    w()
    w("**For the abstract:** Only claim 'p < 0.001' for the Nash-like separation if ")
    w("the between-policy Mann-Whitney p shown above is actually < 0.001.")
    w()
    w("---")
    w()

    # ══════════════════════════════════════════════════════════════════════
    # RQ3
    # ══════════════════════════════════════════════════════════════════════
    w("## RQ3: Non-Stationary Adaptation")
    w()
    w("**Setup:** 1 agent, 5 arms (mu 1-5), 6 sessions × 500 steps = 3,000 total, ")
    w("arm means permuted between sessions, 30 seeds.")
    w()

    # Descriptive
    w("### Table 3: Total regret under regime shifts (30 seeds, mean ± 95% CI)")
    w()
    w("| Strategy | Total Regret | vs SW-UCB |")
    w("|----------|-------------|-----------|")
    sw_m, _, _ = ci_95(rq3["raw_data"]["SW-UCB"])
    for strat in ["SW-UCB", "UCB", "TS", "KL-UCB", "EG(0.05)", "Fixed(best)", "Random"]:
        d = rq3["descriptive"][strat]
        r_str = format_ci(d["mean"], d["ci"])
        if strat == "SW-UCB":
            vs = "---"
        else:
            pct = ((d["mean"] - sw_m) / sw_m) * 100
            vs = f"+{pct:.0f}%"
        w(f"| {strat} | {r_str} | {vs} |")
    w()

    # Omnibus
    kw = rq3["omnibus"]
    w(f"**Omnibus:** Kruskal-Wallis H({kw['df']}) = {kw['H']:.3f}, {format_p(kw['p'])}")
    w()

    # SW-UCB vs each
    w("### SW-UCB vs each strategy (Mann-Whitney U, Holm-Bonferroni adjusted)")
    w()
    w("| Comparison | U | z | p (adjusted) | r | Effect |")
    w("|-----------|---|---|-------------|---|--------|")
    for comp in rq3["sw_ucb_vs_all"]:
        sig = "\\*" if comp["significant"] else "ns"
        w(f"| {comp['label']} | {comp['U']:.0f} | {comp['z']:.2f} | "
          f"{format_p(comp['p_adjusted'])} | {comp['r_effect']:.3f} | "
          f"{describe_effect(comp['r_effect'])} {sig} |")
    w()

    # Key comparison
    reduction = rq3["sw_ucb_vs_ucb_reduction_pct"]
    w(f"**Key comparison:** SW-UCB regret is {reduction:.1f}% lower than standard UCB.")
    w()

    # RQ3 thesis guidance
    w("### Thesis writing guidance (RQ3)")
    w()
    w("**Section 7 (RQ3: Non-Stationary Adaptation)**")
    w()
    w("In your **Results** subsection, lead with SW-UCB vs UCB:")
    w()
    sw_d = rq3["descriptive"]["SW-UCB"]
    ucb_d = rq3["descriptive"]["UCB"]
    # Find the sw-ucb vs ucb comparison
    sw_vs_ucb = next(c for c in rq3["sw_ucb_vs_all"] if "UCB" in c["label"] and "SW" not in c["label"].split("vs")[1])
    w(f"> SW-UCB achieves {sw_d['mean']:.0f} ± {sw_d['ci']:.0f} total regret, "
      f"{reduction:.0f}% lower than standard UCB ({ucb_d['mean']:.0f} ± {ucb_d['ci']:.0f}). "
      f"This difference is {'statistically significant' if sw_vs_ucb['significant'] else 'not statistically significant'} "
      f"(Mann-Whitney U = {sw_vs_ucb['U']:.0f}, z = {sw_vs_ucb['z']:.2f}, "
      f"{format_p(sw_vs_ucb['p_adjusted'])} Holm-Bonferroni adjusted, "
      f"r = {sw_vs_ucb['r_effect']:.3f}).")
    w()
    w("**For the abstract:** Use the actual reduction percentage from above. ")
    w(f"The data shows {reduction:.0f}% reduction. Only claim 'p < 0.001' if the ")
    w("adjusted p-value above is actually < 0.001.")
    w()
    w("---")
    w()

    # ══════════════════════════════════════════════════════════════════════
    # ABSTRACT TEMPLATE
    # ══════════════════════════════════════════════════════════════════════
    w("## Suggested Abstract (fill in from numbers above)")
    w()
    w("Use this template, replacing [...] with the actual numbers from the tables:")
    w()

    # Build the abstract from real numbers
    kw_w = rq1["omnibus"]["welfare"]
    mw_cd = rq2["between_policy"]["collision_drop"]

    w("> Institutional smart order routers face a classic multi-armed bandit problem ")
    w("> when allocating order flow across dark liquidity pools. To model this, we ")
    w("> introduce a multi-agent framework testing three collision mechanisms: pro-rata ")
    w("> splitting, toxic-flow cancellation, and winner-takes-all. Using a simulation ")
    w("> grounded in 222 weeks of FINRA ATS data for AAPL, validated across 30 ")
    w("> independent runs with non-parametric hypothesis tests, we uncover three ")
    w("> dynamics. First, the choice of collision mechanism significantly affects ")
    w(f"> welfare (Kruskal-Wallis H = {kw_w['H']:.1f}, {format_p(kw_w['p'])}). ")
    w("> Second, under toxic-flow cancellation, UCB agents spontaneously specialise ")
    w(f"> across different pools ({format_p(mw_cd['p'])} between policies, ")
    w(f"> r = {mw_cd['r_effect']:.2f}). Third, Sliding-Window UCB outperforms standard ")
    w(f"> UCB under regime shifts, cutting regret by {reduction:.0f}% ")
    w(f"> ({format_p(sw_vs_ucb['p_adjusted'])} Holm-adjusted).")
    w()
    w("**IMPORTANT:** Only include p-values that are actually significant. If any ")
    w("test above yields p > 0.05, rewrite the corresponding claim as a trend or ")
    w("descriptive finding, not a statistical claim.")
    w()
    w("---")
    w()

    # ══════════════════════════════════════════════════════════════════════
    # METHODOLOGY SECTION
    # ══════════════════════════════════════════════════════════════════════
    w("## Methodology section text (copy into thesis Section 3 or 5.1)")
    w()
    w("> All experiments use 30 independent replications with different random ")
    w("> seeds. Descriptive statistics report mean ± 95% confidence interval ")
    w("> (t-distribution, df = 29). Statistical significance is assessed with ")
    w("> non-parametric tests: Kruskal-Wallis H for omnibus comparisons across ")
    w("> three or more conditions, Mann-Whitney U for pairwise comparisons, and ")
    w("> Wilcoxon signed-rank for within-condition paired comparisons. All pairwise ")
    w("> p-values are adjusted for multiple comparisons using the Holm-Bonferroni ")
    w("> procedure, which controls the family-wise error rate without assuming ")
    w("> independence between tests. Effect sizes are reported as rank-biserial r ")
    w("> (|r| < 0.10 negligible, 0.10-0.30 small, 0.30-0.50 medium, > 0.50 large; ")
    w("> Kerby 2014). All statistical computations use scipy.stats (v1.x).")
    w()
    w("---")
    w()
    w("## Reproducibility")
    w()
    w("```bash")
    w("pip install -r requirements.txt")
    w("python -m statistical_tests.run_all")
    w("```")
    w()
    w("This regenerates all numbers in this document from scratch.")

    return "\n".join(lines)


def main():
    start = time.time()

    print("\n" + "█" * 70)
    print("  STATISTICAL TESTS -- MULTI-AGENT BANDIT FRAMEWORK")
    print("█" * 70 + "\n")

    print("Running RQ1...")
    rq1 = test_rq1.run()

    print("\n\nRunning RQ2...")
    rq2 = test_rq2.run()

    print("\n\nRunning RQ3...")
    rq3 = test_rq3.run()

    elapsed = time.time() - start

    findings = write_findings(rq1, rq2, rq3, elapsed)

    out_path = os.path.join(os.path.dirname(__file__), "findings.md")
    with open(out_path, "w") as f:
        f.write(findings)

    print(f"\n\n{'█' * 70}")
    print(f"  DONE in {elapsed:.0f}s. Results written to {out_path}")
    print(f"{'█' * 70}\n")


if __name__ == "__main__":
    main()
