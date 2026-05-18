import math
from typing import List, Tuple, Optional
from src.data_loader import Customer, Instance


FIXED_VEHICLE_COST  = 100.0   
COST_PER_KM         = 1.0      
FUEL_ALPHA          = 0.0761   
FUEL_BETA           = 0.0004  
FUEL_COST_PER_LITRE = 1.5      
EMISSION_PER_LITRE  = 2.62     # kg CO₂ / L
EMISSION_COST       = 0.05     # $/kg CO₂  (carbon price)
SERVICE_TIME_DEFAULT = 10.0    # minutes (used when dataset doesn't specify)

TV_ZONES: List[Tuple[float, float, float]] = [
    (0.0,   480.0, 1.0),   # free-flow
    (480.0, 720.0, 0.6),   # morning peak
    (720.0, 960.0, 0.8),   # afternoon / off-peak
]

BASE_SPEED = 1.0  
def euclidean(a: Customer, b: Customer) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def travel_time_tv(dist: float, depart_time: float) -> float:
    if dist <= 0.0:
        return 0.0

    remaining = dist
    t = depart_time

    for (z_start, z_end, factor) in TV_ZONES:
        if t >= z_end:
            continue                    
        if remaining <= 0.0:
            break

        zone_start_t = max(t, z_start)
        zone_time    = z_end - zone_start_t  # time available in this zone

        speed = BASE_SPEED * factor
        dist_in_zone = speed * zone_time

        if remaining <= dist_in_zone:
            # finish within this zone
            seg_time = remaining / speed
            return (t - depart_time) + seg_time
        else:
            # consume entire zone
            remaining -= dist_in_zone
            t = z_end

    # After last zone: use final zone's speed
    last_speed = BASE_SPEED * TV_ZONES[-1][2]
    remaining_time = remaining / last_speed
    return (t - depart_time) + remaining_time


def fuel_consumption(dist: float, load: float, capacity: float) -> float:
    load_ratio = min(load / capacity, 1.0)
    return (FUEL_ALPHA + FUEL_BETA * load_ratio) * dist


def transport_cost_segment(dist: float, load: float, capacity: float) -> float:
    fuel = fuel_consumption(dist, load, capacity)
    return (
        COST_PER_KM * dist
        + FUEL_COST_PER_LITRE * fuel
        + EMISSION_COST * EMISSION_PER_LITRE * fuel
    )


class RouteEval:

    def __init__(self, instance: Instance):
        self.inst = instance

    def evaluate_route(
        self,
        route: List[int],           # customer indices (1-based, no depot)
        return_details: bool = False
    ) -> dict:
        inst      = self.inst
        depot     = inst.depot
        cap       = inst.vehicle_capacity
        customers = inst.cmap

        cost        = 0.0
        travel_time = 0.0
        load        = 0.0
        t           = 0.0          # current time
        prev        = depot
        load_viol   = 0.0
        tw_viol     = 0.0
        arrivals    = []

        for cid in route:
            c    = customers[cid]
            load += c.demand

        load_viol = max(0.0, load - cap)

        # Reset for time tracking
        load_traversed = 0.0
        t  = 0.0
        prev = depot

        for cid in route:
            c    = customers[cid]
            dist = inst.dist_matrix[prev.id][c.id]
            tt   = travel_time_tv(dist, t)
            seg_cost = transport_cost_segment(dist, load, cap)

            t           += tt
            travel_time += tt
            cost        += seg_cost

            # Service time
            svc = c.service_time if c.service_time > 0 else SERVICE_TIME_DEFAULT

            # Check time windows (MTW: any window is acceptable)
            arrived = t
            arrivals.append(arrived)
            tw_feasible = False
            best_wait   = float("inf")

            for (rt, dt) in c.time_windows:
                wait = max(0.0, rt - arrived)
                if arrived <= dt:
                    tw_feasible = True
                    best_wait   = wait
                    break

            if not tw_feasible:
                # Penalise by the amount we're late relative to LAST window
                last_dt  = c.time_windows[-1][1]
                tw_viol += max(0.0, arrived - last_dt)
                # Still add service at arrived time
                t += svc
            else:
                t += best_wait + svc

            prev = c

        # Return to depot
        dist = inst.dist_matrix[prev.id][depot.id]
        tt   = travel_time_tv(dist, t)
        cost += transport_cost_segment(dist, load, cap)
        travel_time += tt
        t += tt

        # Check depot due time (closing time)
        depot_due = depot.time_windows[-1][1] if depot.time_windows else 960.0
        if t > depot_due:
            tw_viol += (t - depot_due)

        return {
            "cost":           cost,
            "travel_time":    travel_time,
            "load_violation": load_viol,
            "tw_violation":   tw_viol,
            "feasible":       (load_viol == 0.0 and tw_viol == 0.0),
            "arrival_times":  arrivals if return_details else [],
        }


class SolutionEval:
    def __init__(self, instance: Instance):
        self.re = RouteEval(instance)
        self.inst = instance

    def evaluate(self, routes: List[List[int]]) -> dict:
        """
        routes: list of routes, each route is list of customer IDs.
        Returns TC, TT, NV, feasibility, and violations.
        """
        TC   = 0.0
        TT   = 0.0
        NV   = len([r for r in routes if r])  # non-empty routes
        lv   = 0.0
        twv  = 0.0

        TC += FIXED_VEHICLE_COST * NV

        for route in routes:
            if not route:
                continue
            res  = self.re.evaluate_route(route)
            TC  += res["cost"]
            TT  += res["travel_time"]
            lv  += res["load_violation"]
            twv += res["tw_violation"]

        return {
            "TC":             TC,
            "TT":             TT,
            "NV":             NV,
            "load_violation": lv,
            "tw_violation":   twv,
            "feasible":       (lv == 0.0 and twv == 0.0),
        }
