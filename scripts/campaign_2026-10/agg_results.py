"""Compact five-task eval500 table for the psmgoal arms of 2026-10-01..05 (reads $PSM_DATA/logs/*.json)."""
import glob
import json
import os
import re
import sys

LOGS = '/mnt/home/amohan/psm-data/logs'
PAT = re.compile(r'(psmgoal_(?:gc|db|dbu|dbo|ja|sm_code)_[a-z0-9_]+?_(?:cube|antmaze))_(sd\d+)_(\d+)_([a-z_0-9]+?)_task(\d)\.json$')


def succ(p):
    d = json.load(open(p))
    for k in ('success', 'evaluation/success', 'mean_success', 'success_rate'):
        if k in d:
            return float(d[k])
    for v in d.values():
        if isinstance(v, dict):
            for k in ('success', 'evaluation/success'):
                if k in v:
                    return float(v[k])


res = {}
for p in glob.glob(os.path.join(LOGS, 'psmgoal_*_task*.json')):
    m = PAT.search(os.path.basename(p))
    if not m:
        continue
    s = succ(p)
    if s is None:
        continue
    res.setdefault((m.group(1), m.group(3), m.group(4)), {}).setdefault(m.group(2), {})[m.group(5)] = s

only = sys.argv[1] if len(sys.argv) > 1 else None
n_cells = 0
for key, d in sorted(res.items()):
    if only and only not in key[0]:
        continue
    per = {sd: round(sum(v.values()) / len(v), 3) for sd, v in sorted(d.items()) if len(v) == 5}
    cells = sum(len(v) for v in d.values())
    n_cells += cells
    mean = round(sum(per.values()) / len(per), 3) if per else None
    print(f'{key[0]:28s} {key[1]:>6s} {key[2]:16s} mean {mean!s:6s} seeds {per} cells {cells}/15')
print('TOTAL_CELLS', n_cells)
