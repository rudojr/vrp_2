# MOGVRPTW-TV — Python Implementation

**Multi-Objective Green Vehicle Routing Problem with Time Windows under Time-Varying conditions**

Solved with **NSGA-II** (Non-dominated Sorting Genetic Algorithm II).

---

## Problem Description

| Objective | Symbol | Meaning |
|-----------|--------|---------|
| Minimise  | **TC** | Total Transportation Cost (fixed vehicle cost + distance/fuel/emission cost) |
| Minimise  | **TT** | Total Travel Time (all routes combined) |
| Minimise  | **NV** | Number of Vehicles used |

### Key Modelling Features

- **Multiple Time Windows (MTW)**: Solomon CSV files contain 3 time windows per customer; service within *any* window is feasible.
- **Time-Varying (TV) Speeds**: Three congestion periods:
  - `[0, 480)` — free-flow (speed × 1.0)
  - `[480, 720)` — morning peak (speed × 0.6)
  - `[720, 960]` — off-peak (speed × 0.8)
- **Green/Emission Model**: Fuel cost based on Comprehensive Modal Emissions (CME) formula: `fuel = (α + β × load/capacity) × dist`
- **Penalty-based feasibility**: Infeasible solutions are retained in the population but penalised on TC and TT to guide search.

---

## Project Structure

```
VRP_2/
├── data/
│   ├── solomon/                  # Solomon benchmark CSV (MTW format)
│   │   ├── R101_MTW.csv
│   │   ├── C101_MTW.csv
│   │   ├── RC101_MTW.csv
│   │   └── ...
│   └── Homberger_datasets/       # Homberger benchmark TXT
│       ├── homberger_200_customer_instances/
│       └── ...
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Instance parsing (Solomon & Homberger)
│   ├── problem.py                # Objectives, TV speed, fuel model, evaluator
│   └── nsga2.py                  # NSGA-II metaheuristic
├── main.py                       # CLI experiment runner
├── requirements.txt
└── results/                      # Output CSVs (auto-created)
```

---

## Quick Start

### Run Default Instances (R20, R100, C100, RC100)

```bash
python main.py
```

### Verbose Mode (per-generation stats every 50 gens)

```bash
python main.py --instances R20 R100 C100 RC100 --gen 200 --pop 100
```

### Quiet Mode

```bash
python main.py --instances R20 R100 C100 RC100 --gen 200 --pop 100 --quiet
```

### Multiple Runs (for statistical averaging)

```bash
python main.py --instances R100 --runs 5 --gen 300 --pop 150 --quiet
```

### Homberger 200-customer instances

```bash
python main.py --instances H_R1_200 H_C1_200 H_RC1_200 --gen 300 --pop 150 --quiet
```

---

## Instance Keys

| Key      | Dataset  | File           | Customers |
|----------|----------|----------------|-----------|
| `R20`    | Solomon  | R101_MTW.csv   | 20        |
| `R100`   | Solomon  | R101_MTW.csv   | 100       |
| `C100`   | Solomon  | C101_MTW.csv   | 100       |
| `RC100`  | Solomon  | RC101_MTW.csv  | 100       |
| `H_R1_200`  | Homberger | R1_2_1.TXT | 200    |
| `H_R2_200`  | Homberger | R2_2_1.TXT | 200    |
| `H_C1_200`  | Homberger | C1_2_1.TXT | 200    |
| `H_C2_200`  | Homberger | C2_2_1.TXT | 200    |
| `H_RC1_200` | Homberger | RC1_2_1.TXT | 200   |
| `H_RC2_200` | Homberger | RC2_2_1.TXT | 200   |

---

## Output

- `results/summary.csv` — one row per run with TC, TT, NV, elapsed time
- `results/{instance}_run{n}_history.csv` — per-generation Pareto front stats

---

## Algorithm: NSGA-II

| Component | Detail |
|-----------|--------|
| Representation | Giant-tour permutation + greedy capacity-split decoder |
| Crossover | Order Crossover (OX), prob = 0.85 |
| Mutation | Swap (2%) + Inversion (5%) + Or-opt (10%) |
| Selection | Binary tournament (rank + crowding distance) |
| Population | 100 (default) |
| Generations | 200 (default) |
| Objectives | TC (penalised), TT (penalised), NV (exact) |

---

## Sample Results (pop=100, gen=200, seed=42)

| Instance | TC       | TT       | NV |
|----------|----------|----------|----|
| R20      | 699.44   | 615.21   | 2  |
| R100     | 3999.54  | 3709.91  | 8  |
| C100     | 4659.67  | 4207.55  | 10 |
| RC100    | 5331.52  | 5097.48  | 9  |

*(TC and TT are in cost/time units consistent with the distance scale of the dataset.)*
