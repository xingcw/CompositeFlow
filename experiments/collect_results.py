#!/usr/bin/env python
"""Collect hopper-morph results: return at 400K gradient steps (paper protocol) per run, mean +/- std over seeds.
Reads eval/target_return from each run's tensorboard log. Usage: python experiments/collect_results.py [runs_dir]"""
import sys, glob, os, re, collections
import numpy as np
from tensorboardX import SummaryWriter  # noqa (ensures package present)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'runs', 'hopper_morph')
rows = collections.defaultdict(list)
for run in sorted(glob.glob(os.path.join(root, '*', ''))):
    name = os.path.basename(os.path.dirname(run))
    tbs = glob.glob(os.path.join(run, '**', 'tb'), recursive=True)
    if not tbs: continue
    ea = EventAccumulator(tbs[0]); ea.Reload()
    if 'eval/target_return' not in ea.Tags().get('scalars', []): continue
    ev = ea.Scalars('eval/target_return')
    steps = {e.step: e.value for e in ev}
    last_step = max(steps)
    final = steps.get(400000)
    group = re.sub(r'_s\d+$', '', name)
    rows[group].append((name, last_step, final, steps[last_step], os.path.exists(os.path.join(run, 'DONE'))))
paper = {'vflow_medium-replay': '355 +/- 6', 'vflow_medium': '604 +/- 173', 'vflow_medium-expert': '462 +/- 89',
         'bc_sac_medium-replay': '346 +/- 4', 'bc_sac_medium': '436 +/- 45', 'bc_sac_medium-expert': '349 +/- 47'}
print(f"{'group':42s} {'n':>2s} {'return@400K (mean +/- std)':>28s}   {'paper Table 1':>14s}   per-run (step: return)")
for g, rs in sorted(rows.items()):
    finals = [r[2] for r in rs if r[2] is not None]
    key = re.sub(r'_beta[^_]+_xi[^_]+$', '', g)
    stat = f"{np.mean(finals):7.1f} +/- {np.std(finals):5.1f}" if finals else "  (not at 400K yet)"
    detail = ', '.join(f"{r[0].split('_s')[-1]}:{r[1]//1000}K={r[3]:.0f}" for r in rs)
    print(f"{g:42s} {len(finals):>2d} {stat:>28s}   {paper.get(key, '-'):>14s}   {detail}")
