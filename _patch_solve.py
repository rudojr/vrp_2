"""
Patch: Revert TW-check in repair operators (too expensive).
Keep only the soft feasibility fix (FEASIBILITY_TOL=50).
Also reduce warmup_iters formula: max(iter, n*10) instead of n*30.
"""
with open('src/alns_sa.py', 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 1. Remove TW check from greedy repair
OLD_G = """            for pos in range(len(route) + 1):
                if not _tw_ok_insert(inst, route, cid, pos, cmap):
                    continue
                cost = _insertion_cost(inst, route, cid, pos)"""
NEW_G = """            for pos in range(len(route) + 1):
                cost = _insertion_cost(inst, route, cid, pos)"""
if OLD_G in content:
    content = content.replace(OLD_G, NEW_G, 1)
    changes += 1

# 2. Remove TW check from regret2 repair
OLD_R = """                for pos in range(len(route) + 1):
                    if not _tw_ok_insert(inst, route, cid, pos, cmap):
                        continue
                    c = _insertion_cost(inst, route, cid, pos)"""
NEW_R = """                for pos in range(len(route) + 1):
                    c = _insertion_cost(inst, route, cid, pos)"""
if OLD_R in content:
    content = content.replace(OLD_R, NEW_R, 1)
    changes += 1

# 3. Remove TW check from random repair
OLD_RD = """            for pos in range(len(route) + 1):
                if _tw_ok_insert(inst, route, cid, pos, cmap):
                    feasible_slots.append((ri, pos))"""
NEW_RD = """            for pos in range(len(route) + 1):
                feasible_slots.append((ri, pos))"""
if OLD_RD in content:
    content = content.replace(OLD_RD, NEW_RD, 1)
    changes += 1

# 4. Reduce warmup_iters: n*30 -> n*10
OLD_WU = "        warmup_iters = max(self.iterations, self.inst.n * 30)"
NEW_WU = "        warmup_iters = max(self.iterations, self.inst.n * 10)"
if OLD_WU in content:
    content = content.replace(OLD_WU, NEW_WU, 1)
    changes += 1

with open('src/alns_sa.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done, {changes} changes applied:")
print("  - Removed TW check from 3 repair operators (too slow)")
print("  - Reduced warmup_iters: n*30 -> n*10")
print("  - Kept FEASIBILITY_TOL=50 for soft feasibility (already applied)")
