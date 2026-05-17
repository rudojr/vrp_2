import os
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_instance, Instance
from src.alns_sa     import EpsilonALNSSA, Solution, evaluate_solution


DATA_DIR    = Path(__file__).parent / "data"
SOLOMON_DIR = DATA_DIR / "solomon"

SIZE_RANGES = {
    "R":  [20, 40, 60, 80, 100],
    "C":  [20, 40, 60, 80, 100],
    "RC": [20, 40, 60, 80],
}

INSTANCE_IDS = [100, 200, 300, 400, 500, 600, 700, 800]


def _solomon_file(instance_type: str, instance_id: int) -> Path:
    hundreds = instance_id // 100
    suffix   = f"{instance_type}{hundreds}01_MTW.csv"
    return SOLOMON_DIR / suffix


def build_registry() -> Dict[str, Tuple[Path, int]]:

    reg: Dict[str, Tuple[Path, int]] = {}
    for itype, sizes in SIZE_RANGES.items():
        for size in sizes:
            for iid in INSTANCE_IDS:
                fpath = _solomon_file(itype, iid)
                key   = f"{itype}{size}_{iid}"
                reg[key] = (fpath, size)
    return reg


REGISTRY = build_registry()

def subset_instance(inst: Instance, n: int) -> Instance:
    from src.data_loader import Instance as Inst
    return Inst(
        name=f"{inst.name}_n{n}",
        dataset_type=inst.dataset_type,
        vehicle_capacity=inst.vehicle_capacity,
        max_vehicles=inst.max_vehicles,
        depot=inst.depot,
        customers=inst.customers[:n],
    )


def summarise_pareto(pareto: List[Solution]) -> Dict:
    if not pareto:
        nan = float("nan")
        return dict(TC_min=nan, TT_min=nan, NV_min=nan,
                    TC_at_minNV=nan, TT_at_minNV=nan, NV=nan,
                    TC_avg=nan, TT_avg=nan, NV_avg=nan, pareto_size=0)

    min_nv = min(pareto, key=lambda s: s.NV)
    min_tc = min(pareto, key=lambda s: s.TC)
    min_tt = min(pareto, key=lambda s: s.TT)
    return {
        "TC_min":      round(min_tc.TC, 4),
        "TT_min":      round(min_tt.TT, 4),
        "NV_min":      min_nv.NV,
        "TC_at_minNV": round(min_nv.TC, 4),
        "TT_at_minNV": round(min_nv.TT, 4),
        "NV":          min_nv.NV,
        "TC_avg":      round(sum(s.TC for s in pareto) / len(pareto), 4),
        "TT_avg":      round(sum(s.TT for s in pareto) / len(pareto), 4),
        "NV_avg":      round(sum(s.NV for s in pareto) / len(pareto), 2),
        "pareto_size": len(pareto),
    }


def fmt_time(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m)}m{s:.1f}s" if m >= 1 else f"{sec:.2f}s"

def run_experiment(
    instance_keys: List[str],
    n_runs:        int   = 1,
    iterations:    int   = 1000,
    n_remove_pct:  float = 0.15,
    T0:            Optional[float] = None,
    verbose:       bool  = True,
    save_results:  bool  = True,
    output_dir:    str   = "results_alns",
    summary_name:  str   = "summary.csv",
) -> Dict[str, list]:

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results  = {}
    all_rows = []

    for key in instance_keys:
        if key not in REGISTRY:
            print(f"[WARN] Unknown key: {key!r}  (skipping)")
            print(f"       Sample keys: {list(REGISTRY)[:5]} ...")
            continue

        fpath, n_limit = REGISTRY[key]
        if not fpath.exists():
            print(f"[WARN] File not found: {fpath}  (skipping)")
            continue

        print(f"\n{'='*65}")
        print(f"  Instance : {key}   ({fpath.name}, n<={n_limit})")
        print(f"{'='*65}")

        base_inst = load_instance(str(fpath))
        inst      = subset_instance(base_inst, n_limit)

        print(f"  Customers={inst.n}  Capacity={inst.vehicle_capacity}  "
              f"MaxVehicles={inst.max_vehicles}")

        run_summaries = []

        for run_idx in range(n_runs):
            seed = 42 + run_idx * 17
            print(f"\n  --- Run {run_idx+1}/{n_runs}  seed={seed} ---")
            t0 = time.time()

            solver = EpsilonALNSSA(
                instance=inst,
                iterations=iterations,
                n_remove_pct=n_remove_pct,
                T0=T0,
                seed=seed,
                verbose=verbose,
            )
            pareto, history = solver.solve()

            elapsed = time.time() - t0
            ts      = fmt_time(elapsed)
            summary = summarise_pareto(pareto)
            summary["computing_time_s"] = round(elapsed, 4)
            summary["run"]              = run_idx + 1
            summary["instance"]         = key
            run_summaries.append(summary)

            print(f"\n  [OK] Done in {ts}  |  Pareto={summary['pareto_size']}")
            print(f"    TC_min={summary['TC_min']:.2f}  "
                  f"TT_min={summary['TT_min']:.2f}  "
                  f"NV_min={summary['NV_min']}")
            print(f"    [MinNV sol] TC={summary['TC_at_minNV']:.2f}  "
                  f"TT={summary['TT_at_minNV']:.2f}  "
                  f"NV={summary['NV']}")
            print(f"    Computing time : {ts}")

            if save_results and history:
                hist_path = out_path / f"{key}_run{run_idx+1}_history.csv"
                with open(hist_path, "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=history[0].keys())
                    writer.writeheader()
                    writer.writerows(history)

        results[key] = run_summaries
        all_rows.extend(run_summaries)

    # ---- Summary table ----
    print(f"\n\n{'='*75}")
    print(f"{'RESULTS SUMMARY (Eps-ALNS-SA)':^75}")
    print(f"{'='*75}")
    hdr = (f"{'Instance':<16} {'Run':>3} {'TC':>10} {'TT':>10} "
           f"{'NV':>4} {'PF':>4} {'Time':>9}")
    print(hdr)
    print("-" * 75)
    for row in all_rows:
        ts = fmt_time(row["computing_time_s"])
        print(f"{row['instance']:<16} {row['run']:>3} "
              f"{row['TC_at_minNV']:>10.2f} {row['TT_at_minNV']:>10.2f} "
              f"{row['NV']:>4} {row['pareto_size']:>4} {ts:>9}")
    print("=" * 75)

    if n_runs > 1:
        import statistics
        print(f"\n{'AGGREGATED (avg +/- std)':^75}")
        print("-" * 75)
        for key, summaries in results.items():
            tc_v = [s["TC_at_minNV"]      for s in summaries]
            tt_v = [s["TT_at_minNV"]      for s in summaries]
            nv_v = [s["NV"]               for s in summaries]
            ct_v = [s["computing_time_s"] for s in summaries]
            def ms(v): return statistics.mean(v), (statistics.stdev(v) if len(v)>1 else 0)
            tc_m, tc_s = ms(tc_v)
            tt_m, tt_s = ms(tt_v)
            nv_m, nv_s = ms(nv_v)
            ct_m, ct_s = ms(ct_v)
            print(f"{key:<16} TC={tc_m:.2f}+/-{tc_s:.2f}  "
                  f"TT={tt_m:.2f}+/-{tt_s:.2f}  "
                  f"NV={nv_m:.1f}+/-{nv_s:.1f}  "
                  f"Time={ct_m:.2f}s+/-{ct_s:.2f}s")

    if save_results and all_rows:
        summary_path = out_path / summary_name
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults saved -> {summary_path}")

    return results


_DEFAULTS = ["R20_100", "C20_100", "RC20_100"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "MOGVRPTW-TV -- Epsilon-constraint ALNS-SA solver\n\n"
            "Instance key format:  <TYPE><SIZE>_<ID>\n"
            "  TYPE : R | C | RC\n"
            "  SIZE : R/C -> 20,40,60,80,100   RC -> 20,40,60,80\n"
            "  ID   : 100,200,...,800\n\n"
            "Examples:\n"
            "  R20_100   R40_200   C100_300   RC60_800"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instances", "-i", nargs="+", default=_DEFAULTS, metavar="KEY",
        help=f"Instance keys to run (default: {_DEFAULTS})",
    )
    parser.add_argument("--runs", "-r", type=int, default=1,
                        help="Independent runs per instance (default 1)")
    parser.add_argument("--iter", "-n", type=int, default=1000,
                        help="ALNS-SA iterations per epsilon level (default 1000)")
    parser.add_argument("--remove", type=float, default=0.15,
                        help="Fraction of customers removed each iter (default 0.15)")
    parser.add_argument("--T0", type=float, default=None,
                        help="Initial SA temperature (default: auto)")
    parser.add_argument("--output", "-o", default="results_alns",
                        help="Output directory (default: results_alns/)")
    parser.add_argument("--summary-name", default="summary.csv",
                        help="Summary CSV filename (default: summary.csv)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress per-epsilon verbose output")

    parser.add_argument("--type", "-t", choices=["R", "C", "RC", "ALL"], default=None,
                        help="Run all instances of a type (overrides --instances)")
    parser.add_argument("--size", "-s", type=int, default=None,
                        help="Filter by customer size when using --type")
    parser.add_argument("--id", type=int, default=None,
                        help="Filter by instance ID when using --type")
    return parser.parse_args()


def _filter_registry(itype: Optional[str], size: Optional[int], iid: Optional[int]) -> List[str]:
    keys = list(REGISTRY.keys())
    if itype and itype != "ALL":
        keys = [k for k in keys if (
            k.startswith(itype + str(size or "")) or
            k.startswith(itype + "_") or
            k.startswith(itype + "2") or
            k.startswith(itype + "4") or
            k.startswith(itype + "6") or
            k.startswith(itype + "8") or
            k.startswith(itype + "1")
        ) and (
            k[len(itype):len(itype)+1].isdigit() or k[len(itype):len(itype)+1] == ""
        )]
    if size is not None:
        keys = [k for k in keys if k.split("_")[0].endswith(str(size))]
    if iid is not None:
        keys = [k for k in keys if k.endswith(f"_{iid}")]
    return sorted(keys)


if __name__ == "__main__":
    args = parse_args()

    if args.type:
        instance_keys = _filter_registry(args.type, args.size, args.id)
        if not instance_keys:
            print("[ERROR] No instances matched --type/--size/--id filters.")
            sys.exit(1)
    else:
        instance_keys = args.instances

    # Sort by SIZE (numeric) then ID for predictable run order
    def _sort_key(k):
        try:
            part = k.split("_")[0]          # e.g. "C80"
            size = int(''.join(filter(str.isdigit, part)))
            iid  = int(k.split("_")[1])
            return (size, iid)
        except Exception:
            return (0, 0)
    instance_keys = sorted(instance_keys, key=_sort_key)

    print(f"\nEps-ALNS-SA solver  |  instances={instance_keys}")
    print(f"  iterations={args.iter}  remove_pct={args.remove:.0%}  "
          f"runs={args.runs}  T0={'auto' if args.T0 is None else args.T0}")

    run_experiment(
        instance_keys=instance_keys,
        n_runs=args.runs,
        iterations=args.iter,
        n_remove_pct=args.remove,
        T0=args.T0,
        verbose=not args.quiet,
        save_results=True,
        output_dir=args.output,
        summary_name=args.summary_name,
    )
