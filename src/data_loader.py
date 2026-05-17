import csv
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    service_time: float = 0.0
    time_windows: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def ready_time(self) -> float:
        """Return the start of the first time window (backward compat)."""
        return self.time_windows[0][0] if self.time_windows else 0.0

    @property
    def due_time(self) -> float:
        """Return the end of the first time window (backward compat)."""
        return self.time_windows[0][1] if self.time_windows else float("inf")


@dataclass
class Instance:
    name: str
    dataset_type: str          # "solomon" | "homberger"
    vehicle_capacity: float
    max_vehicles: int
    depot: Customer
    customers: List[Customer]

    @property
    def all_nodes(self) -> List[Customer]:
        return [self.depot] + self.customers

    @property
    def n(self) -> int:
        return len(self.customers)

def load_solomon(filepath: str) -> Instance:
    name = os.path.splitext(os.path.basename(filepath))[0]
    customers: List[Customer] = []
    depot: Optional[Customer] = None

    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid   = int(float(row["CUST_NO"]))
            x     = float(row["XCOORD"])
            y     = float(row["YCOORD"])
            dem   = float(row["DEMAND"])
            tws   = [
                (float(row["READY_TIME_1"]), float(row["DUE_TIME_1"])),
                (float(row["READY_TIME_2"]), float(row["DUE_TIME_2"])),
                (float(row["READY_TIME_3"]), float(row["DUE_TIME_3"])),
            ]
            c = Customer(id=cid, x=x, y=y, demand=dem,
                         service_time=0.0, time_windows=tws)
            if cid == 0:
                depot = c
            else:
                customers.append(c)

    assert depot is not None, f"No depot (CUST_NO=0) found in {filepath}"

    return Instance(
        name=name,
        dataset_type="solomon",
        vehicle_capacity=200.0,
        max_vehicles=25,
        depot=depot,
        customers=customers,
    )


def load_homberger(filepath: str) -> Instance:
    name = os.path.splitext(os.path.basename(filepath))[0]
    customers: List[Customer] = []
    depot: Optional[Customer] = None

    with open(filepath, encoding="utf-8") as fh:
        lines = fh.readlines()

    vehicle_line = lines[4].split()
    max_vehicles = int(vehicle_line[0])
    capacity     = float(vehicle_line[1])

    for line in lines[9:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        cid  = int(parts[0])
        x    = float(parts[1])
        y    = float(parts[2])
        dem  = float(parts[3])
        rt   = float(parts[4])
        dt   = float(parts[5])
        st   = float(parts[6])
        c = Customer(
            id=cid, x=x, y=y, demand=dem,
            service_time=st,
            time_windows=[(rt, dt)],
        )
        if cid == 0:
            depot = c
        else:
            customers.append(c)

    assert depot is not None, f"No depot found in {filepath}"

    return Instance(
        name=name,
        dataset_type="homberger",
        vehicle_capacity=capacity,
        max_vehicles=max_vehicles,
        depot=depot,
        customers=customers,
    )

def load_instance(filepath: str) -> Instance:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return load_solomon(filepath)
    elif ext == ".txt":
        return load_homberger(filepath)
    else:
        raise ValueError(f"Unknown file extension: {ext}")
