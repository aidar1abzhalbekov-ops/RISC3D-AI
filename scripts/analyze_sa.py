"""
Summarize results.jsonl from sa_study.py.

Groups by (start, N, schedule), reports mean and sample std of best peak over seeds,
the per-seed best peaks, mean wall time per proposal, and the overlap fraction.
"""

import json
import math
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/refine/results.jsonl"
groups = defaultdict(list)
with open(path) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        groups[(r["start"], r["N"], r["schedule"])].append(r)

print(f"{'start':<7}{'N':>6}  {'sched':<8}{'seeds':>6}{'mean peak':>12}{'std':>9}{'min':>10}{'s/prop':>9}{'overlap%':>10}")
for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
    rows = groups[key]
    peaks = [r["best_peak_c"] for r in rows]
    n = len(peaks)
    mean = sum(peaks) / n
    std = math.sqrt(sum((p - mean) ** 2 for p in peaks) / (n - 1)) if n > 1 else float("nan")
    spp = sum(r["s_per_proposal"] for r in rows) / n
    ov = sum(r["overlap_candidates"] / max(r["N"], 1) for r in rows) / n * 100
    start, N, sched = key
    print(f"{start:<7}{N:>6}  {sched:<8}{n:>6}{mean:>12.3f}{std:>9.3f}{min(peaks):>10.3f}{spp:>9.3f}{ov:>9.1f}%")
    print(f"         per-seed: {', '.join(f'{p:.3f}' for p in peaks)}")
