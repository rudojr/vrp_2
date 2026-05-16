"""
NSGA-II implementation for MOGVRPTW-TV.

Chromosome: permutation of customer IDs, split into routes by
            a Giant Tour representation with depot-separation genes.

Objectives (3):  TC, TT, NV  → minimise all three.
"""

import random
import math
import copy
from typing import List, Tuple, Dict, Any, Optional

from src.problem import SolutionEval, RouteEval
from src.data_loader import Instance


# ---------------------------------------------------------------------------
# Representation helpers
# ---------------------------------------------------------------------------

def decode_giant_tour(giant: List[int], instance: Instance) -> List[List[int]]:
    """
    Split a giant tour into feasible routes using a greedy split decoder.
    Respects vehicle capacity only (time windows checked post-hoc).
    """
    cap      = instance.vehicle_capacity
    cust_map = {c.id: c for c in instance.customers}

    routes: List[List[int]] = []
    current_route: List[int] = []
    current_load = 0.0

    for cid in giant:
        c = cust_map[cid]
        if current_load + c.demand > cap and current_route:
            routes.append(current_route)
            current_route = []
            current_load  = 0.0
        current_route.append(cid)
        current_load += c.demand

    if current_route:
        routes.append(current_route)

    return routes


# ---------------------------------------------------------------------------
# NSGA-II core
# ---------------------------------------------------------------------------

class Individual:
    def __init__(self, giant: List[int]):
        self.giant = giant
        self.objectives: List[float] = []   # [TC_penalised, TT_penalised, NV]
        self.rank      = 0
        self.crowding  = 0.0
        self.routes: List[List[int]] = []
        self.feasible  = True
        self.raw_nv    = 0          # actual vehicle count (never penalised)
        self.raw_tc    = 0.0        # actual TC before penalty
        self.raw_tt    = 0.0        # actual TT before penalty

    def copy(self) -> "Individual":
        ind = Individual(self.giant[:])
        ind.objectives = self.objectives[:]
        ind.rank       = self.rank
        ind.crowding   = self.crowding
        ind.routes     = [r[:] for r in self.routes]
        ind.feasible   = self.feasible
        ind.raw_nv     = self.raw_nv
        ind.raw_tc     = self.raw_tc
        ind.raw_tt     = self.raw_tt
        return ind


def evaluate_individual(ind: Individual, se: SolutionEval, instance: Instance):
    ind.routes  = decode_giant_tour(ind.giant, instance)
    result      = se.evaluate(ind.routes)
    ind.raw_tc  = result["TC"]
    ind.raw_tt  = result["TT"]
    ind.raw_nv  = result["NV"]
    ind.feasible = result["feasible"]
    # Penalty only applied to TC and TT (not NV, to keep it meaningful)
    penalty = (result["load_violation"] + result["tw_violation"]) * 1000.0
    ind.objectives = [
        result["TC"] + penalty,
        result["TT"] + penalty,
        float(result["NV"]),          # NV never penalised
    ]


def dominates(a: List[float], b: List[float]) -> bool:
    """Return True if `a` dominates `b` (all ≤, at least one <)."""
    all_le = all(x <= y for x, y in zip(a, b))
    any_lt = any(x < y  for x, y in zip(a, b))
    return all_le and any_lt


def non_dominated_sort(population: List[Individual]) -> List[List[int]]:
    """Fast non-dominated sorting. Returns list of fronts (indices)."""
    n = len(population)
    dominated_count = [0] * n
    dominators      = [[] for _ in range(n)]
    fronts          = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i].objectives, population[j].objectives):
                dominators[i].append(j)
            elif dominates(population[j].objectives, population[i].objectives):
                dominated_count[i] += 1

        if dominated_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front = []
        for i in fronts[current]:
            for j in dominators[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    population[j].rank = current + 1
                    next_front.append(j)
        current += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(front_indices: List[int], population: List[Individual]):
    if len(front_indices) <= 2:
        for i in front_indices:
            population[i].crowding = float("inf")
        return

    n_obj = len(population[0].objectives)
    for i in front_indices:
        population[i].crowding = 0.0

    for m in range(n_obj):
        sorted_idx = sorted(front_indices,
                            key=lambda i: population[i].objectives[m])
        population[sorted_idx[0]].crowding  = float("inf")
        population[sorted_idx[-1]].crowding = float("inf")
        obj_range = (population[sorted_idx[-1]].objectives[m]
                     - population[sorted_idx[0]].objectives[m])
        if obj_range == 0:
            continue
        for k in range(1, len(sorted_idx) - 1):
            population[sorted_idx[k]].crowding += (
                (population[sorted_idx[k + 1]].objectives[m]
                 - population[sorted_idx[k - 1]].objectives[m])
                / obj_range
            )


def tournament_select(population: List[Individual], k: int = 2) -> Individual:
    competitors = random.sample(population, k)
    best = competitors[0]
    for c in competitors[1:]:
        if (c.rank < best.rank or
                (c.rank == best.rank and c.crowding > best.crowding)):
            best = c
    return best


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def order_crossover(p1: List[int], p2: List[int]) -> List[int]:
    """OX crossover."""
    n  = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b+1] = p1[a:b+1]
    filled = set(p1[a:b+1])
    pos = (b + 1) % n
    for gene in p2[b+1:] + p2[:b+1]:
        if gene not in filled:
            child[pos] = gene
            filled.add(gene)
            pos = (pos + 1) % n
    return child


def swap_mutation(giant: List[int], rate: float = 0.02) -> List[int]:
    giant = giant[:]
    for i in range(len(giant)):
        if random.random() < rate:
            j = random.randint(0, len(giant) - 1)
            giant[i], giant[j] = giant[j], giant[i]
    return giant


def inversion_mutation(giant: List[int], rate: float = 0.05) -> List[int]:
    if random.random() < rate:
        giant = giant[:]
        a, b = sorted(random.sample(range(len(giant)), 2))
        giant[a:b+1] = giant[a:b+1][::-1]
    return giant


def or_opt_mutation(giant: List[int], rate: float = 0.05) -> List[int]:
    """Move a random segment of length 1-3 to a random position."""
    if random.random() < rate:
        giant = giant[:]
        n  = len(giant)
        seg_len = random.randint(1, min(3, n - 1))
        a  = random.randint(0, n - seg_len)
        seg = giant[a:a + seg_len]
        del giant[a:a + seg_len]
        ins = random.randint(0, len(giant))
        giant[ins:ins] = seg
    return giant


# ---------------------------------------------------------------------------
# NSGA-II runner
# ---------------------------------------------------------------------------

class NSGAII:
    def __init__(
        self,
        instance: Instance,
        pop_size:    int   = 100,
        generations: int   = 200,
        cx_prob:     float = 0.85,
        mut_prob:    float = 0.15,
        seed:        int   = 42,
        verbose:     bool  = True,
    ):
        self.instance    = instance
        self.pop_size    = pop_size
        self.generations = generations
        self.cx_prob     = cx_prob
        self.mut_prob    = mut_prob
        self.se          = SolutionEval(instance)
        random.seed(seed)
        self.verbose = verbose

    # ---- initialise ----

    def _random_individual(self) -> Individual:
        cids = [c.id for c in self.instance.customers]
        random.shuffle(cids)
        return Individual(cids)

    def _init_population(self) -> List[Individual]:
        pop = [self._random_individual() for _ in range(self.pop_size)]
        for ind in pop:
            evaluate_individual(ind, self.se, self.instance)
        return pop

    # ---- main loop ----

    def run(self) -> Tuple[List[Individual], List[dict]]:
        """
        Returns (pareto_front, history).
        history: list of dicts with generation stats.
        """
        population = self._init_population()
        history    = []

        fronts = non_dominated_sort(population)
        for f in fronts:
            crowding_distance(f, population)

        for gen in range(self.generations):
            # Generate offspring
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = tournament_select(population)
                p2 = tournament_select(population)

                if random.random() < self.cx_prob:
                    child_genes = order_crossover(p1.giant, p2.giant)
                else:
                    child_genes = p1.giant[:]

                child_genes = swap_mutation(child_genes, rate=0.02)
                child_genes = inversion_mutation(child_genes, rate=0.05)
                child_genes = or_opt_mutation(child_genes, rate=0.10)

                child = Individual(child_genes)
                evaluate_individual(child, self.se, self.instance)
                offspring.append(child)

            # Combine and select
            combined = population + offspring
            fronts   = non_dominated_sort(combined)

            new_pop: List[Individual] = []
            for front in fronts:
                crowding_distance(front, combined)
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(combined[i] for i in front)
                else:
                    remaining = self.pop_size - len(new_pop)
                    sorted_front = sorted(
                        front,
                        key=lambda i: combined[i].crowding,
                        reverse=True
                    )
                    new_pop.extend(combined[i] for i in sorted_front[:remaining])
                    break

            population = new_pop

            # Stats — use raw values for readability
            front0 = [ind for ind in population if ind.rank == 0]
            tc_vals = [ind.raw_tc for ind in front0]
            tt_vals = [ind.raw_tt for ind in front0]
            nv_vals = [ind.raw_nv for ind in front0]

            stat = {
                "generation": gen + 1,
                "pareto_size": len(front0),
                "TC_min": min(tc_vals) if tc_vals else float("nan"),
                "TC_avg": sum(tc_vals) / len(tc_vals) if tc_vals else float("nan"),
                "TT_min": min(tt_vals) if tt_vals else float("nan"),
                "TT_avg": sum(tt_vals) / len(tt_vals) if tt_vals else float("nan"),
                "NV_min": int(min(nv_vals)) if nv_vals else 0,
            }
            history.append(stat)

            if self.verbose and (gen + 1) % 50 == 0:
                print(f"  Gen {gen+1:4d} | Pareto={stat['pareto_size']:3d} | "
                      f"TC_min={stat['TC_min']:8.1f} | "
                      f"TT_min={stat['TT_min']:8.1f} | "
                      f"NV_min={stat['NV_min']:3d}")

        # Return Pareto front
        pareto = [ind for ind in population if ind.rank == 0]
        return pareto, history
