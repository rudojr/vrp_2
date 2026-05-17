from src.data_loader import load_instance, Instance
from src.alns_sa import EpsilonALNSSA, Solution, evaluate_solution, build_initial_solution
from src.problem import SolutionEval
import time

inst_f = load_instance('data/solomon/C101_MTW.csv')
inst = Instance(name='C80', dataset_type='solomon',
                vehicle_capacity=inst_f.vehicle_capacity,
                max_vehicles=inst_f.max_vehicles,
                depot=inst_f.depot,
                customers=inst_f.customers[:80])

se = SolutionEval(inst)

# Build initial
sol = build_initial_solution(inst, seed=42)
evaluate_solution(sol, se)
print(f'Init: NV={sol.NV}  feasible={sol.feasible}  tw_viol={sol.tw_viol:.4f}')

# Check what T0 would be
n = inst.n
T0 = max(200.0, n * 5.0)
print(f'T0={T0}  n_remove=max(2, int({n}*0.15))={max(2, int(n*0.15))}')
print(f'warmup_iters=max(1000, {n}*30)={max(1000, n*30)}')

# Manually test if we can get feasible with greedy repair
import random
rng = random.Random(42)
from src.alns_sa import destroy_random, repair_greedy_insert

t0 = time.time()
current = sol.copy()
best_viol = sol.tw_viol
for it in range(500):
    partial, removed = destroy_random(current, max(2, int(n*0.15)), rng)
    candidate = repair_greedy_insert(partial, removed, inst, rng)
    evaluate_solution(candidate, se)
    if candidate.tw_viol < best_viol:
        best_viol = candidate.tw_viol
        current = candidate
    if candidate.feasible:
        print(f'  FEASIBLE at iter {it+1}! NV={candidate.NV} TC={candidate.TC:.2f}')
        break

print(f'After 500 iters: best_tw_viol={best_viol:.4f}  feasible={current.feasible}')
print(f'Time: {time.time()-t0:.2f}s')
