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


DATA_DIR       = Path(__file__).parent / "data"
SOLOMON_DIR    = DATA_DIR / "solomon"
HOMBERGER_DIR  = DATA_DIR / "Homberger_datasets"

# ---------------------------------------------------------------------------
# Solomon registry  (key: <TYPE><SIZE>_<ID>)
# ---------------------------------------------------------------------------

SOLOMON_SIZE_RANGES = {
    "R":  [20, 40, 60, 80, 100],
    "C":  [20, 40, 60, 80, 100],
    "RC": [20, 40, 60, 80],
}
SOLOMON_IDS = [100, 200, 300, 400, 500, 600, 700, 800]


def _solomon_file(instance_type: str, instance_id: int) -> Path:
    hundreds = instance_id // 100
    suffix   = f"{instance_type}{hundreds}01_MTW.csv"
    return SOLOMON_DIR / suffix


def _build_solomon_registry() -> Dict[str, Tuple[Path, int]]:
    reg: Dict[str, Tuple[Path, int]] = {}
    for itype, sizes in SOLOMON_SIZE_RANGES.items():
        for size in sizes:
            for iid in SOLOMON_IDS:
                fpath = _solomon_file(itype, iid)
                key   = f"{itype}{size}_{iid}"
                reg[key] = (fpath, size)
    return reg


# ---------------------------------------------------------------------------
# Homberger registry
#
# Key format:  H_<TYPE>_<SCALE>_<VARIANT>
#   TYPE    : C1 | C2 | R1 | R2 | RC1 | RC2
#   SCALE   : 100 | 200 | 400 | 600 | 800  (customers to USE)
#   VARIANT : 1 .. 10  (instance number within folder)
#
# Source file mapping (smallest folder that fits the scale):
#   SCALE 100 -> homberger_200_customer_instances  (take first 100)
#   SCALE 200 -> homberger_200_customer_instances  (use all 200)
#   SCALE 400 -> homberger_400_customer_instances  (use all 400)
#   SCALE 600 -> homberger_600_customer_instances  (use all 600)
#   SCALE 800 -> homberger_800_customer_instances  (use all 800)
#
# Examples:
#   H_C1_100_1  -> C1_2_1.TXT (200-cust folder), first 100 customers
#   H_C1_200_1  -> C1_2_1.TXT (200-cust folder), all 200 customers
#   H_R1_400_3  -> R1_4_3.TXT (400-cust folder), all 400 customers
#   H_RC2_800_5 -> RC2_8_5.TXT (800-cust folder), all 800 customers
# ---------------------------------------------------------------------------

HOMBERGER_TYPES    = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
HOMBERGER_VARIANTS = list(range(1, 11))   # 10 instances per folder

HOMBERGER_SCALE_TO_FOLDER = {
    100: 200,   # take first 100 from 200-customer file
    200: 200,   # use all 200
    400: 400,   # use all 400
    600: 600,   # use all 600
    800: 800,   # use all 800
}


def _homberger_file(htype, folder_n, variant):
    size_code = folder_n // 100
    subdir    = f"homberger_{folder_n}_customer_instances"
    filename  = f"{htype}_{size_code}_{variant}.TXT"
    return HOMBERGER_DIR / subdir / filename


def _build_homberger_registry():
    reg = {}
    for htype in HOMBERGER_TYPES:
        for scale, folder_n in HOMBERGER_SCALE_TO_FOLDER.items():
            for v in HOMBERGER_VARIANTS:
                fpath = _homberger_file(htype, folder_n, v)
                key   = f"H_{htype}_{scale}_{v}"
                reg[key] = (fpath, scale)
    return reg


# ---------------------------------------------------------------------------
# Combined registry
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Tuple[Path, int]] = {}
REGISTRY.update(_build_solomon_registry())
REGISTRY.update(_build_homberger_registry())


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

    parser.add_argument("--type", "-t",
                        choices=["R", "C", "RC", "ALL",
                                 "H_R1", "H_R2", "H_C1", "H_C2", "H_RC1", "H_RC2", "H_ALL"],
                        default=None,
                        help="Filter by instance type (Solomon: R/C/RC/ALL  Homberger: H_R1/H_R2/H_C1/H_C2/H_RC1/H_RC2/H_ALL)")
    parser.add_argument("--size", "-s", type=int, default=None,
                        help="Filter by customer size (Solomon: 20/40/60/80/100  Homberger: 200/400/600/800)")
    parser.add_argument("--id", type=int, default=None,
                        help="Filter by instance ID (Solomon: 100-800  Homberger: 200/400/600/800 = customer count)")
    parser.add_argument("--variant", type=int, default=None,
                        help="Homberger only: filter by variant number 1-10 within a folder")
    return parser.parse_args()


def _filter_registry(
    itype:   Optional[str],
    size:    Optional[int],
    iid:     Optional[int],
    variant: Optional[int] = None,
) -> List[str]:
    """Filter REGISTRY keys.

    Solomon  key: <TYPE><SIZE>_<ID>        e.g. C80_100
      --type  : R | C | RC | ALL
      --size  : 20/40/60/80/100
      --id    : 100-800

    Homberger key: H_<TYPE>_<ID>_<VARIANT>  e.g. H_R1_200_1
      --type   : H_R1 | H_R2 | H_C1 | H_C2 | H_RC1 | H_RC2 | H_ALL
      --id     : 200 | 400 | 600 | 800  (= number of customers)
      --variant: 1-10  (instance number within folder)
    """
    keys = list(REGISTRY.keys())

    if not itype or itype == "ALL":
        # Keep only Solomon keys
        keys = [k for k in keys if not k.startswith("H_")]
    elif itype == "H_ALL":
        keys = [k for k in keys if k.startswith("H_")]
    elif itype.startswith("H_"):
        # Homberger filter: key starts with "H_<TYPE>_"
        prefix = itype + "_"
        keys = [k for k in keys if k.startswith(prefix)]
    else:
        # Solomon filter: key starts with <TYPE><digit>
        # Avoid RC matching R: check that first char after TYPE is a digit
        keys = [
            k for k in keys
            if k.startswith(itype)
            and not k.startswith("H_")
            and len(k) > len(itype)
            and k[len(itype)].isdigit()
        ]

    if size is not None:
        # Solomon: key[:-4] ends with str(size)  e.g. "C80" in "C80_100"
        # Homberger: second segment is size  e.g. "200" in "H_R1_200_1"
        def _matches_size(k):
            if k.startswith("H_"):
                parts = k.split("_")  # ["H", "R1", "200", "1"]
                return len(parts) >= 3 and parts[2] == str(size)
            else:
                return k.split("_")[0].endswith(str(size))
        keys = [k for k in keys if _matches_size(k)]

    # Homberger --id filters by customer count (200/400/600/800)
    # Homberger --variant filters by instance number (1-10)
    if iid is not None:
        def _matches_id(k):
            if k.startswith("H_"):
                parts = k.split("_")           # H, TYPE, ID, VARIANT
                return len(parts) >= 3 and parts[2] == str(iid)
            else:
                return k.endswith(f"_{iid}")
        keys = [k for k in keys if _matches_id(k)]

    if variant is not None:
        # Only applies to Homberger keys (ends with _<variant>)
        keys = [k for k in keys if k.startswith("H_") and k.endswith(f"_{variant}")]

    return sorted(keys)



if __name__ == "__main__":
    args = parse_args()

    if args.type:
        instance_keys = _filter_registry(
            args.type, args.size, args.id, getattr(args, 'variant', None)
        )
        if not instance_keys:
            print("[ERROR] No instances matched filters.")
            print("  Solomon --type: R/C/RC/ALL  --size: 20-100  --id: 100-800")
            print("  Homberger --type: H_R1/H_C1/...  --id: 200/400/600/800  --variant: 1-10")
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
