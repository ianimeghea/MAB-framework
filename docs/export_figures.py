"""
export_figures.py
Saves every experimental figure as a standalone PDF in docs/figures/.
Run from the repo root:  python3 docs/export_figures.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from multi_agent_bandits.experiments.generate_report import (
    rq1_base, rq1_agent_scaling, rq1_temporal,
    rq1_arm_configs, rq1_strategy_comparison,
    rq2_rolling_entropy, rq2_arm_distributions,
    rq2_crowding, rq2_collision_temporal, rq2_per_agent_specialisation,
    rq3_static_vs_adaptive, rq3_regime_shift, rq3_competitive,
    fig_rq1_base, fig_rq1_scaling, fig_rq1_temporal,
    fig_rq1_arm_configs, fig_rq1_strategy_welfare,
    fig_rq2_rolling_entropy, fig_rq2_arm_distributions,
    fig_rq2_crowding, fig_rq2_collision_temporal, fig_rq2_specialisation,
    fig_rq3_static_vs_adaptive, fig_rq3_regime_shift, fig_rq3_competitive,
)

OUT = 'docs/figures'
os.makedirs(OUT, exist_ok=True)

seeds       = [42, 123, 7, 99, 555]
short_seeds = seeds[:3]
steps       = 1500
long_steps  = 2000

rq3_sl = 200   # session length
rq3_ns = 8    # num sessions


def save(name, fig):
    path = os.path.join(OUT, f'{name}.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')


# ── RQ1 ──────────────────────────────────────────────────────────────────────
print('RQ1 data...')
d_base                  = rq1_base(steps, seeds)
d_scale, agent_counts   = rq1_agent_scaling([2, 3, 5, 8], steps, seeds)
d_temp                  = rq1_temporal(long_steps, short_seeds)
d_arms, config_names    = rq1_arm_configs(steps, seeds)
d_strat                 = rq1_strategy_comparison(steps, seeds)

print('RQ1 figures...')
save('rq1_base',      fig_rq1_base(d_base))
save('rq1_scaling',   fig_rq1_scaling(d_scale, agent_counts))
save('rq1_temporal',  fig_rq1_temporal(d_temp, long_steps))
save('rq1_armcfg',    fig_rq1_arm_configs(d_arms, config_names))
save('rq1_strategy',  fig_rq1_strategy_welfare(d_strat))

# ── RQ2 ──────────────────────────────────────────────────────────────────────
print('RQ2 data...')
na = 3
d_ent   = rq2_rolling_entropy(long_steps, na, window=150, seeds=short_seeds)
d_dist  = rq2_arm_distributions(steps, na, seeds)
d_crowd, crowd_counts = rq2_crowding([1,2,3,5,7,10], steps, seeds), [1,2,3,5,7,10]
d_coll  = rq2_collision_temporal(long_steps, na, window=200, seeds=short_seeds)
d_spec  = rq2_per_agent_specialisation(steps, na, seeds)

print('RQ2 figures...')
save('rq2_entropy',    fig_rq2_rolling_entropy(d_ent, long_steps, na))
save('rq2_dist',       fig_rq2_arm_distributions(d_dist, na))
save('rq2_crowding',   fig_rq2_crowding(d_crowd, crowd_counts))
save('rq2_collision',  fig_rq2_collision_temporal(d_coll, long_steps, na))
save('rq2_spec',       fig_rq2_specialisation(d_spec, steps, na))

# ── RQ3 ──────────────────────────────────────────────────────────────────────
print('RQ3 data...')
d_static                     = rq3_static_vs_adaptive(steps, seeds)
d_regime, oracle, session_len = rq3_regime_shift(rq3_sl, rq3_ns, short_seeds)
d_comp                       = rq3_competitive(rq3_sl, rq3_ns, seeds)

print('RQ3 figures...')
save('rq3_static',    fig_rq3_static_vs_adaptive(d_static))
save('rq3_regime',    fig_rq3_regime_shift(d_regime, oracle, session_len))
save('rq3_comp',      fig_rq3_competitive(d_comp))

print(f'\nDone — {len(os.listdir(OUT))} PDFs in {OUT}/')
