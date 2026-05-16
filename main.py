"""
Main runner for MOGVRPTW-TV experiments.

Runs NSGA-II on selected instances and reports:
    TC — Total Transportation Cost
    TT — Total Travel Time
    NV — Number of Vehicles

Instance mapping:
    R20   → Solomon R101_MTW  (first 20 customers)
    R100  → Solomon R101_MTW  (all 100 customers)
    C100  → Solomon C101_MTW  (all 100 customers)
    RC100 → Solomon RC101_MTW (all 100 customers)

Homberger (200-customer) instances are also evaluated if requested.
"""

import os
import sys
import time
import random
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional

# Ensure src/ is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_instance, Instance, Customer
from src.nsga2 import NSGAII, Individual
from src.problem import SolutionEval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def subset_instance(inst: Instance, n_customers: int) -> Instance:
    """Return a copy of inst with only the first n_customers customers."""
    from src.data_loader import Instance as Inst
    subs = Inst(
        name=f"{inst.name}_n{n_customers}",
        dataset_type=inst.dataset_type,
        vehicle_capacity=inst.vehicle_capacity,
        max_vehicles=inst.max_vehicles,
        depot=inst.depot,
        customers=inst.customers[:n_customers],
    )
    return subs


def summarise_pareto(pareto: List[Individual]) -> Dict[str, float]:
    """Compute representative metrics from the Pareto front using raw (unpenalised) values."""
    if not pareto:
        return {"TC": float("nan"), "TT": float("nan"), "NV": float("nan")}

    # Use raw values so penalty doesn't distort reporting
    min_nv_sol = min(pareto, key=lambda x: x.raw_nv)
    min_tc_sol = min(pareto, key=lambda x: x.raw_tc)
    min_tt_sol = min(pareto, key=lambda x: x.raw_tt)

    return {
        "TC_min":      min_tc_sol.raw_tc,
        "TT_min":      min_tt_sol.raw_tt,
        "NV_min":      min_nv_sol.raw_nv,
        "TC_at_minNV": min_nv_sol.raw_tc,
        "TT_at_minNV": min_nv_sol.raw_tt,
        "NV":          min_nv_sol.raw_nv,
        # Pareto averages (raw)
        "TC_avg":      sum(x.raw_tc for x in pareto) / len(pareto),
        "TT_avg":      sum(x.raw_tt for x in pareto) / len(pareto),
        "NV_avg":      sum(x.raw_nv for x in pareto) / len(pareto),
        "pareto_size": len(pareto),
    }


# ---------------------------------------------------------------------------
# Instance registry
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

SOLOMON_DIR    = DATA_DIR / "solomon"
HOMBERGER_BASE = DATA_DIR / "Homberger_datasets"

INSTANCE_REGISTRY = {
    # name        : (filepath,                         n_customers_limit)
    "R20"  : (SOLOMON_DIR / "R101_MTW.csv",   20),
    "R100" : (SOLOMON_DIR / "R101_MTW.csv",  100),
    "C100" : (SOLOMON_DIR / "C101_MTW.csv",  100),
    "RC100": (SOLOMON_DIR / "RC101_MTW.csv", 100),
}

# Add Homberger 200-customer instances (first R1 instance per type)
HOMBERGER_200 = HOMBERGER_BASE / "homberger_200_customer_instances"
for prefix in ["R1", "R2", "C1", "C2", "RC1", "RC2"]:
    fname = f"{prefix}_2_1.TXT"
    fpath = HOMBERGER_200 / fname
    key   = f"H_{prefix}_200"
    INSTANCE_REGISTRY[key] = (fpath, None)   # None = all customers


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

DEFAULT_INSTANCES = ["R20", "R100", "C100", "RC100"]

NSGA2_PARAMS = {
    "pop_size":    100,
    "generations": 200,
    "cx_prob":     0.85,
    "mut_prob":    0.15,
}


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    instance_keys:  List[str],
    n_runs:         int   = 1,
    verbose:        bool  = True,
    save_results:   bool  = True,
    output_dir:     str   = "results",
    pop_size:       int   = 100,
    generations:    int   = 200,
) -> Dict[str, list]:
    """
    Run NSGA-II on each instance for n_runs independent seeds.
    Returns aggregated results dict keyed by instance name.
    """
    results = {}
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_rows = []   # for CSV export

    for key in instance_keys:
        if key not in INSTANCE_REGISTRY:
            print(f"[WARN] Unknown instance key: {key}, skipping.")
            continue

        fpath, n_limit = INSTANCE_REGISTRY[key]
        if not fpath.exists():
            print(f"[WARN] File not found: {fpath}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Instance: {key}  ({fpath.name})")
        print(f"{'='*60}")

        base_inst = load_instance(str(fpath))
        if n_limit is not None:
            inst = subset_instance(base_inst, n_limit)
        else:
            inst = base_inst

        print(f"  Customers: {inst.n}, Capacity: {inst.vehicle_capacity}, "
              f"Max vehicles: {inst.max_vehicles}")

        run_summaries = []
        for run_idx in range(n_runs):
            seed = 42 + run_idx * 13
            print(f"\n  --- Run {run_idx+1}/{n_runs} (seed={seed}) ---")
            t0 = time.time()

            algo = NSGAII(
                instance=inst,
                pop_size=pop_size,
                generations=generations,
                cx_prob=0.85,
                mut_prob=0.15,
                seed=seed,
                verbose=verbose,
            )
            pareto, history = algo.run()

            elapsed = time.time() - t0
            summary = summarise_pareto(pareto)
            summary["elapsed_s"] = elapsed
            summary["run"]       = run_idx + 1
            summary["instance"]  = key
            run_summaries.append(summary)

            print(f"\n  [OK] Done in {elapsed:.1f}s | Pareto={summary['pareto_size']}")
            print(f"    TC_min={summary['TC_min']:.2f}  "
                  f"TT_min={summary['TT_min']:.2f}  "
                  f"NV_min={summary['NV_min']}")
            print(f"    [MinNV solution] TC={summary['TC_at_minNV']:.2f}  "
                  f"TT={summary['TT_at_minNV']:.2f}  "
                  f"NV={summary['NV']}")

            # Save history CSV per run
            if save_results:
                hist_path = out_path / f"{key}_run{run_idx+1}_history.csv"
                with open(hist_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=history[0].keys())
                    writer.writeheader()
                    writer.writerows(history)

        results[key] = run_summaries
        all_rows.extend(run_summaries)

    # Print summary table
    print(f"\n\n{'='*72}")
    print(f"{'RESULTS SUMMARY':^72}")
    print(f"{'='*72}")
    header = f"{'Instance':<12} {'Run':>4} {'TC':>10} {'TT':>10} {'NV':>5} {'Time(s)':>8}"
    print(header)
    print("-" * 72)
    for row in all_rows:
        print(f"{row['instance']:<12} {row['run']:>4} "
              f"{row['TC_at_minNV']:>10.2f} "
              f"{row['TT_at_minNV']:>10.2f} "
              f"{row['NV']:>5} "
              f"{row['elapsed_s']:>8.1f}")
    print("=" * 72)

    # Aggregate (if multiple runs)
    if n_runs > 1:
        print(f"\n{'AGGREGATED (avg ± std)':^72}")
        print("-" * 72)
        import statistics
        for key, summaries in results.items():
            tc_vals = [s["TC_at_minNV"] for s in summaries]
            tt_vals = [s["TT_at_minNV"] for s in summaries]
            nv_vals = [s["NV"]          for s in summaries]
            tc_m  = statistics.mean(tc_vals)
            tt_m  = statistics.mean(tt_vals)
            nv_m  = statistics.mean(nv_vals)
            tc_s  = statistics.stdev(tc_vals) if len(tc_vals) > 1 else 0
            tt_s  = statistics.stdev(tt_vals) if len(tt_vals) > 1 else 0
            nv_s  = statistics.stdev(nv_vals) if len(nv_vals) > 1 else 0
            print(f"{key:<12} TC={tc_m:.2f}±{tc_s:.2f}  "
                  f"TT={tt_m:.2f}±{tt_s:.2f}  "
                  f"NV={nv_m:.1f}±{nv_s:.1f}")

    # Save summary CSV
    if save_results and all_rows:
        summary_path = out_path / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults saved -> {summary_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MOGVRPTW-TV solver (NSGA-II) on Solomon/Homberger datasets"
    )
    parser.add_argument(
        "--instances", "-i",
        nargs="+",
        default=DEFAULT_INSTANCES,
        help=(
            "Instance keys to run. "
            f"Defaults: {DEFAULT_INSTANCES}. "
            "Available: R20, R100, C100, RC100, H_R1_200, H_R2_200, H_C1_200, H_C2_200, H_RC1_200, H_RC2_200"
        ),
    )
    parser.add_argument(
        "--runs", "-r",
        type=int, default=1,
        help="Number of independent runs per instance (default 1)."
    )
    parser.add_argument(
        "--pop", "-p",
        type=int, default=100,
        help="Population size (default 100)."
    )
    parser.add_argument(
        "--gen", "-g",
        type=int, default=200,
        help="Number of generations (default 200)."
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default="results",
        help="Output directory (default: results/)."
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-generation verbose output."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        instance_keys=args.instances,
        n_runs=args.runs,
        verbose=not args.quiet,
        save_results=True,
        output_dir=args.output,
        pop_size=args.pop,
        generations=args.gen,
    )
