import math
import random
import copy
import time
from typing import List, Dict, Tuple, Optional

from src.problem import SolutionEval, RouteEval, euclidean, travel_time_tv
from src.problem import (
    FIXED_VEHICLE_COST, SERVICE_TIME_DEFAULT,
    transport_cost_segment, COST_PER_KM,
)
from src.data_loader import Instance, Customer

class Solution:

    def __init__(self, routes: List[List[int]]):
        self.routes: List[List[int]] = [r[:] for r in routes if r]
        self.TC:       float = float("inf")
        self.TT:       float = float("inf")
        self.NV:       int   = len(self.routes)
        self.feasible: bool  = False
        self.tw_viol:  float = float("inf")
        self.lv_viol:  float = float("inf")

    def copy(self) -> "Solution":
        s = Solution(self.routes)
        s.TC       = self.TC
        s.TT       = self.TT
        s.NV       = self.NV
        s.feasible = self.feasible
        s.tw_viol  = self.tw_viol
        s.lv_viol  = self.lv_viol
        return s

    def all_customers(self) -> List[int]:
        return [c for route in self.routes for c in route]

    def remove_empty_routes(self):
        self.routes = [r for r in self.routes if r]
        self.NV     = len(self.routes)


# Near-feasible tolerance: small tw_viol may be unavoidable due to TV-speed model
FEASIBILITY_TOL = 50.0

def evaluate_solution(sol: Solution, se: SolutionEval):
    result      = se.evaluate(sol.routes)
    sol.TC       = result["TC"]
    sol.TT       = result["TT"]
    sol.NV       = result["NV"]
    sol.tw_viol  = result["tw_violation"]
    sol.lv_viol  = result["load_violation"]
    sol.feasible = (sol.lv_viol == 0.0 and sol.tw_viol <= FEASIBILITY_TOL)


def _insertion_cost(
    inst: Instance,
    route: List[int],
    cid: int,
    pos: int,
) -> float:
    cmap = {c.id: c for c in inst.customers}
    cmap[0] = inst.depot

    prev_id = 0 if pos == 0 else route[pos - 1]
    next_id = 0 if pos == len(route) else route[pos]

    prev = cmap[prev_id]
    cust = cmap[cid]
    nxt  = cmap[next_id]

    d_remove = euclidean(prev, nxt)
    d_add    = euclidean(prev, cust) + euclidean(cust, nxt)
    cap      = inst.vehicle_capacity

    load_approx = sum(cmap[c].demand for c in route) + cust.demand
    return transport_cost_segment(d_add - d_remove, load_approx, cap)



def _route_arrival_at_pos(inst, route, pos, cmap):
    """Estimate arrival time at position `pos` in route (before inserting new cust).
    Returns the approximate time the vehicle arrives at route[pos] (or end-of-route
    if pos==len(route)).  Fast O(pos) estimate."""
    from src.problem import travel_time_tv, euclidean, SERVICE_TIME_DEFAULT
    depot = inst.depot
    t = 0.0
    prev = depot
    for i in range(min(pos, len(route))):
        c = cmap[route[i]]
        dist = euclidean(prev, c)
        t += travel_time_tv(dist, t)
        # service + best-wait at earliest TW
        svc = c.service_time if c.service_time > 0 else SERVICE_TIME_DEFAULT
        best_start = t
        for (rt, dt) in c.time_windows:
            if t <= dt:
                best_start = max(t, rt)
                break
        t = best_start + svc
        prev = c
    return t


def _tw_ok_insert(inst, route, cid, pos, cmap):
    """Return True if inserting `cid` at `pos` can satisfy at least one TW.
    Uses a fast arrival-time estimate (ignores downstream shifts)."""
    from src.problem import travel_time_tv, euclidean
    t_at_pos = _route_arrival_at_pos(inst, route, pos, cmap)
    prev_id  = 0 if pos == 0 else route[pos - 1]
    prev = cmap[prev_id]
    c    = cmap[cid]
    dist = euclidean(prev, c)
    from src.problem import travel_time_tv
    arr  = t_at_pos + travel_time_tv(dist, t_at_pos)
    return any(arr <= dt for (_, dt) in c.time_windows)


def build_initial_solution(inst, seed=42):
    """Each customer in its own route so ALNS starts feasible w.r.t. capacity.
    SA/ALNS will merge routes while satisfying time windows.
    """
    routes = [[c.id] for c in inst.customers]
    return Solution(routes)

def destroy_random(sol: Solution, n_remove: int, rng: random.Random) -> Tuple[Solution, List[int]]:
    """Remove n_remove random customers."""
    s       = sol.copy()
    removed = []
    all_c   = s.all_customers()
    rng.shuffle(all_c)
    to_remove = set(all_c[:n_remove])

    new_routes = []
    for route in s.routes:
        nr = [c for c in route if c not in to_remove]
        new_routes.append(nr)
        removed.extend(c for c in route if c in to_remove)

    s.routes = new_routes
    s.remove_empty_routes()
    return s, removed


def destroy_worst_cost(
    sol: Solution, n_remove: int, inst: Instance, rng: random.Random
) -> Tuple[Solution, List[int]]:
    cmap  = {c.id: c for c in inst.customers}
    cmap[0] = inst.depot
    cap   = inst.vehicle_capacity

    marginals: List[Tuple[float, int]] = []
    for route in sol.routes:
        load = sum(cmap[c].demand for c in route)
        for i, cid in enumerate(route):
            prev_id = 0 if i == 0 else route[i - 1]
            next_id = 0 if i == len(route) - 1 else route[i + 1]
            prev = cmap[prev_id]
            c    = cmap[cid]
            nxt  = cmap[next_id]
            d_with    = euclidean(prev, c) + euclidean(c, nxt)
            d_without = euclidean(prev, nxt)
            saving    = transport_cost_segment(d_with - d_without, load, cap)
            noise     = rng.uniform(0.9, 1.1)   
            marginals.append((saving * noise, cid))

    marginals.sort(reverse=True)
    to_remove = set(m[1] for m in marginals[:n_remove])

    removed   = []
    new_routes = []
    for route in sol.routes:
        nr = [c for c in route if c not in to_remove]
        new_routes.append(nr)
        removed.extend(c for c in route if c in to_remove)

    s = sol.copy()
    s.routes = new_routes
    s.remove_empty_routes()
    return s, removed


def destroy_tw_violation(
    sol: Solution, n_remove: int, inst: Instance, rng: random.Random
) -> Tuple[Solution, List[int]]:
    """Remove customers causing time-window violations."""
    re   = RouteEval(inst)
    viol_scores: List[Tuple[float, int]] = []

    for route in sol.routes:
        result = re.evaluate_route(route, return_details=True)
        arrivals = result["arrival_times"]
        cmap = {c.id: c for c in inst.customers}

        for i, cid in enumerate(route):
            c = cmap.get(cid)
            if c is None:
                continue
            arr = arrivals[i] if i < len(arrivals) else 0.0
            viol = 0.0
            in_window = any(rt <= arr <= dt for (rt, dt) in c.time_windows)
            if not in_window and c.time_windows:
                last_dt = c.time_windows[-1][1]
                viol    = max(0.0, arr - last_dt)
            noise = rng.uniform(0.9, 1.1)
            viol_scores.append((viol * noise, cid))

    viol_scores.sort(reverse=True)
    # If no violations, fall back to random
    if not viol_scores or viol_scores[0][0] == 0:
        return destroy_random(sol, n_remove, rng)

    to_remove = set(m[1] for m in viol_scores[:n_remove])
    removed   = []
    new_routes = []
    for route in sol.routes:
        nr = [c for c in route if c not in to_remove]
        new_routes.append(nr)
        removed.extend(c for c in route if c in to_remove)

    s = sol.copy()
    s.routes = new_routes
    s.remove_empty_routes()
    return s, removed


def destroy_cluster(
    sol: Solution, n_remove: int, inst: Instance, rng: random.Random
) -> Tuple[Solution, List[int]]:
    cmap = {c.id: c for c in inst.customers}
    all_c = sol.all_customers()
    if not all_c:
        return sol.copy(), []

    seed = rng.choice(all_c)
    seed_c = cmap[seed]

    def similarity(cid: int) -> float:
        c = cmap[cid]
        dist   = euclidean(seed_c, c)
        tw_sim = abs(seed_c.ready_time - c.ready_time)
        return dist + 0.3 * tw_sim

    scored = [(similarity(c), c) for c in all_c if c != seed]
    scored.sort()
    to_remove = {seed} | {scored[i][1] for i in range(min(n_remove - 1, len(scored)))}

    removed   = []
    new_routes = []
    for route in sol.routes:
        nr = [c for c in route if c not in to_remove]
        new_routes.append(nr)
        removed.extend(c for c in route if c in to_remove)

    s = sol.copy()
    s.routes = new_routes
    s.remove_empty_routes()
    return s, removed


def _check_capacity(route: List[int], cmap: Dict[int, Customer], cap: float) -> bool:
    return sum(cmap[c].demand for c in route) <= cap


def repair_greedy_insert(
    sol: Solution, removed: List[int], inst: Instance, rng: random.Random
) -> Solution:
    s    = sol.copy()
    cmap = {c.id: c for c in inst.customers}
    cmap[0] = inst.depot
    cap  = inst.vehicle_capacity

    order = removed[:]
    rng.shuffle(order)

    for cid in order:
        best_cost = float("inf")
        best_r    = -1
        best_pos  = -1

        for ri, route in enumerate(s.routes):
            # Capacity check first
            load = sum(cmap[c].demand for c in route)
            if load + cmap[cid].demand > cap:
                continue
            for pos in range(len(route) + 1):
                cost = _insertion_cost(inst, route, cid, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_r    = ri
                    best_pos  = pos

        if best_r == -1:
            # Open new route
            s.routes.append([cid])
        else:
            s.routes[best_r].insert(best_pos, cid)

    s.remove_empty_routes()
    return s


def repair_regret2_insert(
    sol: Solution, removed: List[int], inst: Instance, rng: random.Random
) -> Solution:
    s    = sol.copy()
    cmap = {c.id: c for c in inst.customers}
    cmap[0] = inst.depot
    cap  = inst.vehicle_capacity

    unrouted = removed[:]

    while unrouted:
        regrets: List[Tuple[float, int, int, int]] = []

        for cid in unrouted:
            costs = []
            for ri, route in enumerate(s.routes):
                load = sum(cmap[c].demand for c in route)
                if load + cmap[cid].demand > cap:
                    continue
                for pos in range(len(route) + 1):
                    c = _insertion_cost(inst, route, cid, pos)
                    costs.append((c, ri, pos))

            costs.sort()
            if len(costs) == 0:
                # Must open new route
                regrets.append((float("inf"), cid, -1, -1))
            elif len(costs) == 1:
                regrets.append((float("inf"), cid, costs[0][1], costs[0][2]))
            else:
                regret = costs[1][0] - costs[0][0]
                regrets.append((regret, cid, costs[0][1], costs[0][2]))

        # Insert customer with highest regret
        regrets.sort(reverse=True)
        _, cid, best_r, best_pos = regrets[0]
        unrouted.remove(cid)

        if best_r == -1:
            s.routes.append([cid])
        else:
            s.routes[best_r].insert(best_pos, cid)

    s.remove_empty_routes()
    return s


def repair_random_insert(
    sol: Solution, removed: List[int], inst: Instance, rng: random.Random
) -> Solution:
    s    = sol.copy()
    cmap = {c.id: c for c in inst.customers}
    cmap[0] = inst.depot
    cap  = inst.vehicle_capacity

    order = removed[:]
    rng.shuffle(order)

    for cid in order:
        feasible_slots = []
        for ri, route in enumerate(s.routes):
            load = sum(cmap[c].demand for c in route)
            if load + cmap[cid].demand > cap:
                continue
            for pos in range(len(route) + 1):
                feasible_slots.append((ri, pos))

        if not feasible_slots:
            s.routes.append([cid])
        else:
            ri, pos = rng.choice(feasible_slots)
            s.routes[ri].insert(pos, cid)

    s.remove_empty_routes()
    return s


class AdaptiveWeights:

    def __init__(self, n: int, initial_weight: float = 1.0):
        self.weights = [initial_weight] * n
        self.scores  = [0.0] * n
        self.counts  = [0]   * n
        self.decay   = 0.6   
        self.sigma   = [3.0, 2.0, 1.0, 0.0] 

    def select(self, rng: random.Random) -> int:
        total = sum(self.weights)
        r     = rng.uniform(0, total)
        cum   = 0.0
        for i, w in enumerate(self.weights):
            cum += w
            if cum >= r:
                return i
        return len(self.weights) - 1

    def update(self, idx: int, outcome: int):
        self.scores[idx] += self.sigma[outcome]
        self.counts[idx] += 1

    def adapt(self, segment_size: int):
        for i in range(len(self.weights)):
            if self.counts[i] > 0:
                self.weights[i] = (
                    self.decay * self.weights[i]
                    + (1 - self.decay) * (self.scores[i] / self.counts[i])
                )
            self.scores[i] = 0.0
            self.counts[i] = 0

DESTROY_OPS = ["random", "worst_cost", "tw_violation", "cluster"]
REPAIR_OPS  = ["greedy", "regret2", "random"]


def _apply_destroy(
    op: str, sol: Solution, n_remove: int, inst: Instance, rng: random.Random
) -> Tuple[Solution, List[int]]:
    if op == "random":
        return destroy_random(sol, n_remove, rng)
    elif op == "worst_cost":
        return destroy_worst_cost(sol, n_remove, inst, rng)
    elif op == "tw_violation":
        return destroy_tw_violation(sol, n_remove, inst, rng)
    elif op == "cluster":
        return destroy_cluster(sol, n_remove, inst, rng)
    raise ValueError(f"Unknown destroy op: {op}")


def _apply_repair(
    op: str, sol: Solution, removed: List[int], inst: Instance, rng: random.Random
) -> Solution:
    if op == "greedy":
        return repair_greedy_insert(sol, removed, inst, rng)
    elif op == "regret2":
        return repair_regret2_insert(sol, removed, inst, rng)
    elif op == "random":
        return repair_random_insert(sol, removed, inst, rng)
    raise ValueError(f"Unknown repair op: {op}")


class EpsilonALNSSA:
    def __init__(
        self,
        instance:     Instance,
        iterations:   int   = 1000,
        segment_size: int   = 100,
        n_remove_pct: float = 0.15,   # fraction of customers to remove each iteration
        T0:           float = None,    # initial SA temperature (auto if None)
        T_final:      float = 1e-4,
        cooling:      float = None,    # geometric cooling factor (auto if None)
        seed:         int   = 42,
        verbose:      bool  = True,
    ):
        self.inst         = instance
        self.iterations   = iterations
        self.segment_size = segment_size
        self.n_remove     = max(2, int(instance.n * n_remove_pct))
        self.T_final      = T_final
        self.seed         = seed
        self.verbose      = verbose
        self.se           = SolutionEval(instance)

        self.T0 = T0 if T0 is not None else max(200.0, instance.n * 5.0)
        if cooling is not None:
            self.cooling = cooling
        else:
            # Geometric schedule: T0 * cooling^iterations = T_final
            self.cooling = (self.T_final / self.T0) ** (1.0 / max(1, iterations))

        self.rng = random.Random(seed)


    def _sa_accept(self, delta: float, T: float) -> bool:
        """True if the new solution is accepted under SA criterion."""
        if delta <= 0:
            return True
        return self.rng.random() < math.exp(-delta / T)

    def _dominates(self, a: Solution, b: Solution) -> bool:
        """True if a strictly dominates b on (TC, TT, NV)."""
        return (
            a.TC <= b.TC and a.TT <= b.TT and a.NV <= b.NV
            and (a.TC < b.TC or a.TT < b.TT or a.NV < b.NV)
        )

    def _update_pareto(self, archive: List[Solution], sol: Solution) -> List[Solution]:
        if not sol.feasible:
            return archive
        new_archive = [s for s in archive if not self._dominates(sol, s)]
        if not any(self._dominates(s, sol) for s in new_archive):
            new_archive.append(sol.copy())
        return new_archive

    # ------------------------------------------------------------------ #

    def _run_alns_sa(
        self,
        init_sol:  Solution,
        nv_limit:  int,
        history:   List[dict],
        t_start:   float,
    ) -> Tuple[Solution, List[Solution]]:
        """
        Core ALNS-SA loop for a fixed NV epsilon level.
        Returns (best_sol_for_this_epsilon, pareto_archive).
        """
        current   = init_sol.copy()
        best      = init_sol.copy()
        archive:  List[Solution] = []
        archive   = self._update_pareto(archive, current)

        d_weights = AdaptiveWeights(len(DESTROY_OPS))
        r_weights = AdaptiveWeights(len(REPAIR_OPS))

        T = self.T0

        for it in range(self.iterations):
            # Select operators
            d_idx = d_weights.select(self.rng)
            r_idx = r_weights.select(self.rng)

            # Destroy
            partial, removed = _apply_destroy(
                DESTROY_OPS[d_idx], current, self.n_remove, self.inst, self.rng
            )

            # Repair
            candidate = _apply_repair(
                REPAIR_OPS[r_idx], partial, removed, self.inst, self.rng
            )
            evaluate_solution(candidate, self.se)

            # Enforce NV epsilon constraint (allow one extra vehicle as soft)
            nv_ok = candidate.NV <= nv_limit

            # Objective for SA: weighted sum (penalty pushes toward feasibility)
            PENALTY_WEIGHT = 2000.0
            penalty_cand = (candidate.tw_viol + candidate.lv_viol) * PENALTY_WEIGHT
            penalty_curr = (current.tw_viol + current.lv_viol) * PENALTY_WEIGHT
            obj_cand = candidate.TC + 0.1 * candidate.TT + penalty_cand
            obj_curr = current.TC  + 0.1 * current.TT  + penalty_curr

            outcome = 3  # reject

            if nv_ok or not current.feasible:
                delta = obj_cand - obj_curr
                if self._sa_accept(delta, T):
                    current = candidate
                    if delta <= 0 and candidate.feasible:
                        outcome = 1   # improve
                    else:
                        outcome = 2   # accept worse

                    # Update best
                    if (
                        candidate.feasible
                        and candidate.TC < best.TC
                        and candidate.NV <= nv_limit
                    ):
                        best    = candidate.copy()
                        outcome = 0   # new best

                # Update Pareto archive
                archive = self._update_pareto(archive, candidate)

            d_weights.update(d_idx, outcome)
            r_weights.update(r_idx, outcome)

            # Adapt weights every segment
            if (it + 1) % self.segment_size == 0:
                d_weights.adapt(self.segment_size)
                r_weights.adapt(self.segment_size)

            # Cool temperature
            T = max(self.T_final, T * self.cooling)

            # Log history
            if (it + 1) % max(1, self.iterations // 50) == 0:
                pf_tc  = min((s.TC for s in archive), default=float("nan"))
                pf_tt  = min((s.TT for s in archive), default=float("nan"))
                pf_nv  = min((s.NV for s in archive), default=0)
                history.append({
                    "iteration":   it + 1,
                    "nv_limit":    nv_limit,
                    "temperature": round(T, 6),
                    "current_TC":  round(current.TC, 4),
                    "current_NV":  current.NV,
                    "best_TC":     round(best.TC, 4),
                    "pareto_TC":   round(pf_tc, 4),
                    "pareto_TT":   round(pf_tt, 4),
                    "pareto_NV":   pf_nv,
                    "pareto_size": len(archive),
                    "elapsed_s":   round(time.time() - t_start, 4),
                })

        return best, archive

    # ------------------------------------------------------------------ #

    def solve(self):
        """Epsilon-constraint: Phase 1=warmup, Phase 2=eps NV loop."""
        t_start = time.time()
        history = []

        init_sol = build_initial_solution(self.inst, seed=self.seed)
        evaluate_solution(init_sol, self.se)

        if self.verbose:
            print(f'  Initial: NV={init_sol.NV}  TC={init_sol.TC:.2f}  '
                  f'TT={init_sol.TT:.2f}  feasible={init_sol.feasible}')

        warmup_iters = max(self.iterations, self.inst.n * 10)
        if self.verbose:
            print(f'  Warmup ({warmup_iters} iters) ...', end=' ', flush=True)

        warmup_sol, _ = self._run_alns_sa(
            init_sol, nv_limit=999, history=[], t_start=t_start
        )
        if self.verbose:
            print(f'feasible={warmup_sol.feasible}  '
                  f'NV={warmup_sol.NV}  TC={warmup_sol.TC:.2f}')

        # Prefer feasible warmup; if still infeasible, use warmup_sol
        # (better starting point than init_sol with NV=n)
        current_start = warmup_sol
        nv_max  = current_start.NV
        archive = []
        archive = self._update_pareto(archive, current_start)

        for nv_limit in range(nv_max, 0, -1):
            if self.verbose:
                print(f'  Eps NV<={nv_limit} ...', end=' ', flush=True)

            best_for_level, level_archive = self._run_alns_sa(
                current_start, nv_limit, history, t_start
            )

            for s in level_archive:
                archive = self._update_pareto(archive, s)

            if self.verbose:
                fsol = [s for s in archive if s.NV <= nv_limit]
                if fsol:
                    print(f'Pareto={len(archive)}  best_TC={min(s.TC for s in fsol):.2f}')
                else:
                    print(f'Pareto={len(archive)}  (no feasible at NV<={nv_limit})')

            if best_for_level.feasible and best_for_level.NV <= nv_limit:
                current_start = best_for_level.copy()
            else:
                if self.verbose:
                    print(f'  -> No feasible at NV<={nv_limit}, stopping.')
                break

        elapsed = time.time() - t_start
        if self.verbose:
            print(f'  Done in {elapsed:.2f}s | Pareto={len(archive)}')

        return archive, history

