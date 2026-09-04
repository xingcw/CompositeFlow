"""Summarise the C1 ablation: final scores and gap reliability per arm."""

import argparse
import json
import pathlib
import statistics

ARM_LABEL = {
    "ot": "ot (conditioning gathered)",
    "ot_uncoupled": "ot_uncoupled (reference)",
    "identity": "identity",
}


def load_runs(root):
    """{arm: {seed: [metric rows]}} from the run directories."""
    runs = {}
    for done in sorted(pathlib.Path(root).glob("*.done")):
        tag = done.stem                      # e.g. "ot_uncoupled-s0"
        arm, _, seed = tag.rpartition("-s")
        matches = list((pathlib.Path(root) / tag).glob("*/metrics.json"))
        if not matches:
            print(f"[analyze] {tag}: no metrics.json, skipping")
            continue
        runs.setdefault(arm, {})[int(seed)] = json.loads(matches[0].read_text())
    return runs


def _last(rows, field):
    vals = [r[field] for r in rows if field in r]
    return vals[-1] if vals else None


def _mean_over_tail(rows, field, n=5):
    vals = [r[field] for r in rows if field in r]
    return statistics.fmean(vals[-n:]) if vals else None


def summarise(runs, tail=5):
    print(f"\n{'arm':32s} {'seeds':>5s} {'final score':>16s} "
          f"{'score (last %d)' % tail:>16s} {'gap rho':>16s} {'filter prec':>16s}")
    print("-" * 116)
    for arm in sorted(runs, key=lambda a: (a != "ot", a)):
        per_seed = runs[arm]
        finals = [_last(rows, "normalized_score") for rows in per_seed.values()]
        tails = [_mean_over_tail(rows, "normalized_score", tail) for rows in per_seed.values()]
        rhos = [_mean_over_tail(rows, "gap_spearman", tail) for rows in per_seed.values()]
        rhos = [r for r in rhos if r is not None]
        precs = [_mean_over_tail(rows, "gap_filter_precision", tail) for rows in per_seed.values()]
        precs = [p for p in precs if p is not None]
        chance = [_mean_over_tail(rows, "gap_filter_chance", tail) for rows in per_seed.values()]
        chance = [c for c in chance if c is not None]

        def fmt(vals):
            if not vals:
                return "n/a"
            if len(vals) == 1:
                return f"{vals[0]:.2f}"
            return f"{statistics.fmean(vals):.2f} +/- {statistics.stdev(vals):.2f}"

        print(f"{ARM_LABEL.get(arm, arm):32s} {len(per_seed):5d} "
              f"{fmt(finals):>16s} {fmt(tails):>16s} {fmt(rhos):>16s} {fmt(precs):>16s}")
    if chance:
        print(f"\n  (filter precision at chance = {statistics.fmean(chance):.3f}; "
              "rho is Spearman between the estimated gap and the true one, "
              "computed by stepping both simulators)")
    print()


def curves(runs):
    print("normalised score by step\n")
    steps = sorted({r["step"] for per_seed in runs.values()
                    for rows in per_seed.values() for r in rows})
    header = "step".rjust(8) + "".join(f"{ARM_LABEL.get(a, a)[:22]:>24s}"
                                       for a in sorted(runs))
    print(header)
    for step in steps:
        line = f"{step:8d}"
        for arm in sorted(runs):
            vals = [r["normalized_score"] for rows in runs[arm].values()
                    for r in rows if r["step"] == step]
            line += f"{statistics.fmean(vals):>24.2f}" if vals else f"{'-':>24s}"
        print(line)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/c1_ablation")
    ap.add_argument("--tail", default=5, type=int,
                    help="average the last N evaluations to reduce eval noise")
    ap.add_argument("--curves", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"no completed runs under {args.root}")
    summarise(runs, args.tail)
    if args.curves:
        curves(runs)
