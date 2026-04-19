"""
calibrate.py
Recomputes ARMS_CALIBRATED from the raw FINRA ATS JSON data.
Run from the repo root: python3 data/finra_ats/calibrate.py

Outputs the ARMS_CALIBRATED list ready to paste into generate_report.py,
and writes venue_stats_full.csv with all venue statistics.
"""

import json
import csv
import numpy as np
from collections import defaultdict

RAW_FILE  = 'data/finra_ats/aapl_weekly_raw.json'
OUT_CSV   = 'data/finra_ats/venue_stats_full.csv'
MIN_WEEKS = 150    # stability filter: venues must have this many weeks of data
TOP_N     = 5      # number of arms to use
SCALE_MIN = 1.5    # lower bound of scaled quality range
SCALE_MAX = 5.0    # upper bound (best venue = SCALE_MAX)

with open(RAW_FILE) as f:
    raw = json.load(f)

print(f"Loaded {len(raw)} records")
print(f"Date range: {min(r['weekStartDate'] for r in raw)} "
      f"to {max(r['weekStartDate'] for r in raw)}")

# Aggregate by venue
venue_weeks = defaultdict(list)
venue_names = {}
for r in raw:
    mpid = r['MPID']
    venue_weeks[mpid].append(r['totalWeeklyShareQuantity'])
    venue_names[mpid] = r['marketParticipantName']

# Compute statistics
stats = []
for mpid, shares in venue_weeks.items():
    if len(shares) < MIN_WEEKS:
        continue
    arr = np.array(shares, dtype=float)
    stats.append({
        'mpid':    mpid,
        'name':    venue_names[mpid],
        'n_weeks': len(shares),
        'mean':    float(arr.mean()),
        'std':     float(arr.std()),
        'cv':      float(arr.std() / arr.mean()),
        'total':   float(arr.sum()),
    })

# Sort by mean volume descending
stats.sort(key=lambda x: -x['mean'])
top5 = stats[:TOP_N]
max_mean = top5[0]['mean']

# Scale parameters
scale_range = SCALE_MAX - SCALE_MIN
for s in stats:
    s['scaled_mean'] = SCALE_MIN + (s['mean'] / max_mean) * scale_range
    s['scaled_std']  = (s['std'] / max_mean) * scale_range

# Write full CSV
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
    writer.writeheader()
    writer.writerows(stats)
print(f"Wrote {len(stats)} venues to {OUT_CSV}")

# Print results
print(f"\nTop {TOP_N} venues (≥{MIN_WEEKS} weeks):")
print(f"{'MPID':<6} {'Venue':<32} {'Weeks':>5} {'Mean':>8} {'Std':>8} {'CV':>6}")
print("─" * 68)
for s in top5:
    print(f"{s['mpid']:<6} {s['name'][:31]:<32} {s['n_weeks']:>5} "
          f"{s['scaled_mean']:>8.3f} {s['scaled_std']:>8.3f} {s['cv']:>6.3f}")

print("\nARMS_CALIBRATED = [")
for s in reversed(top5):
    print(f"    ({s['scaled_mean']:.3f}, {s['scaled_std']:.3f}),  "
          f"# {s['mpid']} — {s['name']}  (CV={s['cv']:.3f}, {s['n_weeks']}w)")
print("]")
print(f"ORACLE_CALIBRATED = {top5[0]['scaled_mean']:.3f}  "
      f"# {top5[0]['mpid']} — {top5[0]['name']}")
