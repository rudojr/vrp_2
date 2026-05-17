# MOGVRPTW-TV — Python Implementation

**Multi-Objective Green Vehicle Routing Problem with Time Windows under Time-Varying conditions**

Solved with **NSGA-II** (`main.py`) and **Epsilon-constraint ALNS-SA** (`main_alns.py`).

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
- **Penalty-based feasibility**: Infeasible solutions are retained but penalised on TC and TT to guide search.

---

## Project Structure

```
VRP_2/
├── data/
│   ├── solomon/                  # Solomon benchmark CSV (MTW format)
│   │   ├── R101_MTW.csv
│   │   ├── R201_MTW.csv
│   │   ├── C101_MTW.csv
│   │   ├── C202_MTW.csv
│   │   ├── RC101_MTW.csv
│   │   ├── RC201_MTW.csv
│   │   └── ...
│   └── Homberger_datasets/       # Homberger benchmark TXT
│       ├── homberger_200_customer_instances/
│       └── ...
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Instance parsing (Solomon & Homberger)
│   ├── problem.py                # Objectives, TV speed, fuel model, evaluator
│   ├── nsga2.py                  # NSGA-II metaheuristic
│   └── alns_sa.py                # Epsilon-constraint ALNS-SA solver
├── main.py                       # NSGA-II CLI runner
├── main_alns.py                  # Eps-ALNS-SA CLI runner
├── requirements.txt
├── results/                      # NSGA-II output CSVs (auto-created)
└── results_alns/                 # ALNS-SA output CSVs (auto-created)
```

---

## Solver 1 — NSGA-II (`main.py`)

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

## Solver 2 — Epsilon-constraint ALNS-SA (`main_alns.py`)

### Instance Key Format

```
<TYPE><SIZE>_<ID>
```

| Field  | Values | Meaning |
|--------|--------|---------|
| `TYPE` | `R`, `C`, `RC` | Solomon instance type |
| `SIZE` | R/C → `20, 40, 60, 80, 100`  ·  RC → `20, 40, 60, 80` | Number of customers to use |
| `ID`   | `100, 200, 300, 400, 500, 600, 700, 800` | Solomon instance number (e.g. 100 = R**1**01) |

**Examples:**

| Key | File | Customers used |
|-----|------|---------------|
| `R20_100`  | `R101_MTW.csv`  | first 20  |
| `R40_100`  | `R101_MTW.csv`  | first 40  |
| `R100_100` | `R101_MTW.csv`  | all 100   |
| `C20_100`  | `C101_MTW.csv`  | first 20  |
| `C40_200`  | `C202_MTW.csv`  | first 40  |
| `RC20_100` | `RC101_MTW.csv` | first 20  |
| `RC80_200` | `RC201_MTW.csv` | first 80  |

> **Note:** Only instance files present in `data/solomon/` are usable.
> Currently available: `R101`, `R201`, `C101`, `C202`, `RC101`, `RC201`.

---

### Running Specific Instance Keys

#### Một instance đơn lẻ

```bash
python main_alns.py --instances R20_100 --iter 1000
```

#### Nhiều instance keys cụ thể

```bash
python main_alns.py --instances R20_100 R40_100 R60_100 --iter 1000 --quiet
```

#### Chạy C20 và C40

```bash
# Chỉ C20 (instance ID 100)
python main_alns.py --instances C20_100 --iter 1000 --quiet

# Chỉ C40 (instance ID 100)
python main_alns.py --instances C40_100 --iter 1000 --quiet

# C20 + C40 cùng lúc (cả 2 file có sẵn)
python main_alns.py --instances C20_100 C20_200 C40_100 C40_200 --iter 1000 --quiet
```

#### Chạy toàn bộ một Type

```bash
# Tất cả R instances (mọi size × mọi ID có file)
python main_alns.py --type R --iter 1000 --quiet

# Tất cả C instances, chỉ size 20
python main_alns.py --type C --size 20 --iter 1000 --quiet

# Tất cả RC instances, chỉ ID 100
python main_alns.py --type RC --id 100 --iter 1000 --quiet
```

#### Chạy nhiều Runs để lấy thống kê

```bash
python main_alns.py --instances R100_100 --runs 5 --iter 2000 --quiet
```

#### Tùy chỉnh tham số SA

```bash
# Tăng iterations và fraction customers bị remove mỗi bước
python main_alns.py --instances C100_100 --iter 3000 --remove 0.20 --quiet

# Đặt nhiệt độ SA ban đầu thủ công
python main_alns.py --instances RC60_100 --iter 2000 --T0 100.0 --quiet

# Đổi thư mục output
python main_alns.py --instances R20_100 C20_100 --output my_results/ --quiet
```

---

### CLI Arguments — `main_alns.py`

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--instances` | `-i` | `R20_100 C20_100 RC20_100` | Danh sách instance keys |
| `--type` | `-t` | — | Chạy toàn bộ type: `R`, `C`, `RC`, `ALL` |
| `--size` | `-s` | — | Lọc theo số khách hàng (dùng với `--type`) |
| `--id` | — | — | Lọc theo instance ID (dùng với `--type`) |
| `--runs` | `-r` | `1` | Số lần chạy độc lập |
| `--iter` | `-n` | `1000` | Số iterations ALNS-SA mỗi epsilon level |
| `--remove` | — | `0.15` | Tỷ lệ khách hàng bị remove mỗi iter |
| `--T0` | — | auto | Nhiệt độ SA ban đầu |
| `--output` | `-o` | `results_alns/` | Thư mục lưu kết quả |
| `--quiet` | `-q` | — | Tắt verbose output |

---

## Instance Keys — NSGA-II (`main.py`)

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

## Output Files

### NSGA-II (`results/`)
- `summary.csv` — one row per run: TC, TT, NV, computing_time_s
- `{instance}_run{n}_history.csv` — per-generation Pareto front stats

### ALNS-SA (`results_alns/`)
- `summary.csv` — one row per run: TC, TT, NV, pareto_size, computing_time_s
- `{key}_run{n}_history.csv` — per-iteration stats including temperature, pareto_TC, elapsed_s

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

## Algorithm: Epsilon-constraint ALNS-SA

| Component | Detail |
|-----------|--------|
| Outer loop | Epsilon-constraint on NV (fix max vehicles, minimise TC & TT) |
| Phase 1 | Unconstrained ALNS-SA warmup → find first feasible solution |
| Phase 2 | Eps-constraint loop: NV_max → 1, collect Pareto front |
| Destroy ops | Random · Worst-cost · TW-violation · Shaw-cluster (4 ops) |
| Repair ops | Greedy best-insert · Regret-2 · Random insert (3 ops) |
| Acceptance | Simulated Annealing (geometric cooling) |
| Adaptation | Roulette-wheel operator weights, updated every 100 iters |
| Objectives | TC, TT, NV (non-dominated archive) |

---

## Sample Results — ALNS-SA (iter=500, seed=42)

| Instance | TC      | TT     | NV | Pareto | Time  |
|----------|---------|--------|----|--------|-------|
| R20_100  | 1055.63 | 651.09 | 5  | 32     | 0.57s |
| C20_100  | 828.17  | 340.74 | 5  | 310    | 0.68s |

---

## Sample Results — NSGA-II (pop=100, gen=200, seed=42)

| Instance | TC       | TT       | NV |
|----------|----------|----------|----|
| R20      | 699.44   | 615.21   | 2  |
| R100     | 3999.54  | 3709.91  | 8  |
| C100     | 4659.67  | 4207.55  | 10 |
| RC100    | 5331.52  | 5097.48  | 9  |

*(TC and TT are in cost/time units consistent with the distance scale of the dataset.)*
