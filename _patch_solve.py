"""Patch: replace Homberger registry with scale-based design."""

NEW_HOMBERGER = '''# ---------------------------------------------------------------------------
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

# scale -> folder_size (source file to read from)
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


'''

with open('main_alns.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find start and end markers
start_marker = '# ---------------------------------------------------------------------------\n# Homberger registry'
end_marker = '# ---------------------------------------------------------------------------\n# Combined registry'

idx_start = content.find(start_marker)
idx_end   = content.find(end_marker)

if idx_start == -1 or idx_end == -1:
    print(f"ERROR: markers not found (start={idx_start}, end={idx_end})")
else:
    new_content = content[:idx_start] + NEW_HOMBERGER + content[idx_end:]
    with open('main_alns.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"OK - replaced Homberger registry ({idx_end-idx_start} chars -> {len(NEW_HOMBERGER)} chars)")
    # Quick verify
    from importlib import import_module
    import importlib, sys
    if 'main_alns' in sys.modules:
        del sys.modules['main_alns']
