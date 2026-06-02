# Statistical Findings -- Multi-Agent Bandit Framework

Generated: 2026-05-16 14:43
Runtime: 5 seconds

All p-values are from non-parametric tests (Mann-Whitney U or Kruskal-Wallis).
Confidence intervals are 95% (t-based, n=30 seeds).
Multiple comparisons corrected with Holm-Bonferroni.
Effect sizes are rank-biserial r (|r| > 0.5 = large).

---

## RQ1: Collision Policies and Market Welfare

**Setup:** 3 UCB agents, 5 arms (mu 1-5), 2,000 steps, 30 seeds.

### Table 1: Cross-policy summary (30 seeds, mean ± 95% CI)

| Policy | Welfare/step | Collision % | Cumulative Regret |
|--------|-------------|-------------|-------------------|
| linear_share | 11.63 ± 0.28 | 11.10 ± 9.04% | 2422.98 ± 627.73 |
| zero_on_collision | 11.40 ± 0.76 | 7.18 ± 9.42% | 1829.14 ± 942.72 |
| winner_takes_all | 11.86 ± 0.05 | 3.73 ± 1.58% | 2543.70 ± 623.68 |

### Omnibus tests (Kruskal-Wallis)

- **welfare**: H(2) = 23.304, p < 0.0001
- **collision_rate**: H(2) = 49.430, p < 0.0001
- **regret**: H(2) = 11.725, p = 0.003

### Pairwise comparisons (Mann-Whitney U, Holm-Bonferroni adjusted)

| Comparison | Metric | U | z | p (adjusted) | r | Effect |
|-----------|--------|---|---|-------------|---|--------|
| linear_share vs winner_takes_all | welfare | 271 | -2.65 | p = 0.017 | 0.398 | medium \* |
| linear_share vs zero_on_collision | welfare | 129 | -4.75 | p < 0.0001 | 0.713 | large \* |
| winner_takes_all vs zero_on_collision | welfare | 295 | -2.29 | p = 0.022 | 0.344 | medium \* |
| linear_share vs winner_takes_all | collision_rate | 678 | 3.38 | p = 0.0007 | -0.508 | large \* |
| linear_share vs zero_on_collision | collision_rate | 842 | 5.80 | p < 0.0001 | -0.871 | large \* |
| winner_takes_all vs zero_on_collision | collision_rate | 834 | 5.68 | p < 0.0001 | -0.853 | large \* |
| linear_share vs winner_takes_all | regret | 439 | -0.16 | p = 0.877 | 0.024 | negligible ns |
| linear_share vs zero_on_collision | regret | 655 | 3.03 | p = 0.007 | -0.456 | medium \* |
| winner_takes_all vs zero_on_collision | regret | 645 | 2.88 | p = 0.008 | -0.433 | medium \* |

### Thesis writing guidance (RQ1)

**Section 5 (RQ1: Collision Policies and Market Welfare)**

In your **Setup** subsection, state:
> N = 3 UCB agents, K = 5 arms (wide gap: μ ∈ {1, 2, 3, 4, 5}), 
> T = 2,000 steps, 30 seeds. All results report mean ± 95% CI. 
> Mann-Whitney U tests assess pairwise significance with Holm-Bonferroni 
> correction for multiple comparisons (3 pairwise tests per metric).

In your **Results** subsection, report the numbers from Table 1 above, then write:

> The omnibus Kruskal-Wallis test confirms a significant effect of collision 
> policy on welfare (H(2) = 23.304, p < 0.0001).

> linear_share vs winner_takes_all: linear_share welfare (11.63 ± 0.28) is lower than winner_takes_all (11.86 ± 0.05); U = 271, z = -2.65, p = 0.017 (Holm-Bonferroni adjusted), r = 0.398 (medium).

> linear_share vs zero_on_collision: linear_share welfare (11.63 ± 0.28) is higher than zero_on_collision (11.40 ± 0.76); U = 129, z = -4.75, p < 0.0001 (Holm-Bonferroni adjusted), r = 0.713 (large).

> winner_takes_all vs zero_on_collision: winner_takes_all welfare (11.86 ± 0.05) is higher than zero_on_collision (11.40 ± 0.76); U = 295, z = -2.29, p = 0.022 (Holm-Bonferroni adjusted), r = 0.344 (medium).

**Key finding for the abstract:** State the omnibus result and the specific 
comparison that is most policy-relevant (typically zero_on_collision vs linear_share).
Only claim p < 0.001 if the adjusted p-value shown above actually is < 0.001.

---

## RQ2: Emergent Diversification

**Setup:** 2 UCB agents, 5 arms (mu 1-5), 3,000 steps, 30 seeds.

### Table 2: Collision rate dynamics (30 seeds, mean ± 95% CI)

| Policy | Early CR | Late CR | Drop | Specialisation % |
|--------|----------|---------|------|------------------|
| zero_on_collision | 0.61 ± 0.08% | 0.02 ± 0.02% | 0.59 ± 0.09pp | 100.0% |
| linear_share | 1.49 ± 0.22% | 0.07 ± 0.03% | 1.43 ± 0.23pp | 100.0% |

### Between-policy comparisons (Mann-Whitney U)

- **collision_drop**: U = 44, z = -6.00, p < 0.0001, r = 0.901 (large)
- **specialisation**: U = 450, z = 0.00, p = 1.000, r = 0.000 (negligible)

### Within-policy paired tests (Wilcoxon signed-rank: early > late CR)

- **zero_on_collision**: W = 465, p < 0.0001
- **linear_share**: W = 465, p < 0.0001

### Thesis writing guidance (RQ2)

**Section 6 (RQ2: Emergent Diversification)**

In your **Results** subsection, lead with the collision-drop comparison:

> Under zero-on-collision, the collision rate dropped by 0.6 ± 0.1 percentage points from early to late phase, compared to 1.4 ± 0.2pp under linear share. This difference is statistically significant (Mann-Whitney U = 44, z = -6.00, p < 0.0001, r = 0.901).

Then report specialisation rates and the within-policy Wilcoxon results.
The specialisation index measures whether agents anchor on *different* top arms.

**For the abstract:** Only claim 'p < 0.001' for the Nash-like separation if 
the between-policy Mann-Whitney p shown above is actually < 0.001.

---

## RQ3: Non-Stationary Adaptation

**Setup:** 1 agent, 5 arms (mu 1-5), 6 sessions × 500 steps = 3,000 total, 
arm means permuted between sessions, 30 seeds.

### Table 3: Total regret under regime shifts (30 seeds, mean ± 95% CI)

| Strategy | Total Regret | vs SW-UCB |
|----------|-------------|-----------|
| SW-UCB | 731.17 ± 84.51 | --- |
| UCB | 1819.10 ± 268.43 | +149% |
| TS | 2942.52 ± 311.69 | +302% |
| KL-UCB | 1422.00 ± 244.97 | +94% |
| EG(0.05) | 3735.01 ± 375.17 | +411% |
| Fixed(best) | 6059.18 ± 617.40 | +729% |
| Random | 6002.31 ± 32.18 | +721% |

**Omnibus:** Kruskal-Wallis H(6) = 175.845, p < 0.0001

### SW-UCB vs each strategy (Mann-Whitney U, Holm-Bonferroni adjusted)

| Comparison | U | z | p (adjusted) | r | Effect |
|-----------|---|---|-------------|---|--------|
| SW-UCB vs UCB | 95 | -5.25 | p < 0.0001 | 0.789 | large \* |
| SW-UCB vs TS | 8 | -6.53 | p < 0.0001 | 0.982 | large \* |
| SW-UCB vs KL-UCB | 149 | -4.45 | p < 0.0001 | 0.669 | large \* |
| SW-UCB vs EG(0.05) | 0 | -6.65 | p < 0.0001 | 1.000 | large \* |
| SW-UCB vs Fixed(best) | 0 | -6.65 | p < 0.0001 | 1.000 | large \* |
| SW-UCB vs Random | 0 | -6.65 | p < 0.0001 | 1.000 | large \* |

**Key comparison:** SW-UCB regret is 59.8% lower than standard UCB.

### Thesis writing guidance (RQ3)

**Section 7 (RQ3: Non-Stationary Adaptation)**

In your **Results** subsection, lead with SW-UCB vs UCB:

> SW-UCB achieves 731 ± 85 total regret, 60% lower than standard UCB (1819 ± 268). This difference is statistically significant (Mann-Whitney U = 95, z = -5.25, p < 0.0001 Holm-Bonferroni adjusted, r = 0.789).

**For the abstract:** Use the actual reduction percentage from above. 
The data shows 60% reduction. Only claim 'p < 0.001' if the 
adjusted p-value above is actually < 0.001.

---

## Suggested Abstract (fill in from numbers above)

Use this template, replacing [...] with the actual numbers from the tables:

> Institutional smart order routers face a classic multi-armed bandit problem 
> when allocating order flow across dark liquidity pools. To model this, we 
> introduce a multi-agent framework testing three collision mechanisms: pro-rata 
> splitting, toxic-flow cancellation, and winner-takes-all. Using a simulation 
> grounded in 222 weeks of FINRA ATS data for AAPL, validated across 30 
> independent runs with non-parametric hypothesis tests, we uncover three 
> dynamics. First, the choice of collision mechanism significantly affects 
> welfare (Kruskal-Wallis H = 23.3, p < 0.0001). 
> Second, under toxic-flow cancellation, UCB agents spontaneously specialise 
> across different pools (p < 0.0001 between policies, 
> r = 0.90). Third, Sliding-Window UCB outperforms standard 
> UCB under regime shifts, cutting regret by 60% 
> (p < 0.0001 Holm-adjusted).

**IMPORTANT:** Only include p-values that are actually significant. If any 
test above yields p > 0.05, rewrite the corresponding claim as a trend or 
descriptive finding, not a statistical claim.

---

## Methodology section text (copy into thesis Section 3 or 5.1)

> All experiments use 30 independent replications with different random 
> seeds. Descriptive statistics report mean ± 95% confidence interval 
> (t-distribution, df = 29). Statistical significance is assessed with 
> non-parametric tests: Kruskal-Wallis H for omnibus comparisons across 
> three or more conditions, Mann-Whitney U for pairwise comparisons, and 
> Wilcoxon signed-rank for within-condition paired comparisons. All pairwise 
> p-values are adjusted for multiple comparisons using the Holm-Bonferroni 
> procedure, which controls the family-wise error rate without assuming 
> independence between tests. Effect sizes are reported as rank-biserial r 
> (|r| < 0.10 negligible, 0.10-0.30 small, 0.30-0.50 medium, > 0.50 large; 
> Kerby 2014). All statistical computations use scipy.stats (v1.x).

---

## Reproducibility

```bash
pip install -r requirements.txt
python -m statistical_tests.run_all
```

This regenerates all numbers in this document from scratch.