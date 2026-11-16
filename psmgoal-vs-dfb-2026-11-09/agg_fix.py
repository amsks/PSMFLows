"""Five-task means of every psmz_fix_* eval JSON in $PSM_DATA/logs (actor alone). Usage: .venv/bin/python agg_fix.py"""
import glob, json, os, re
import numpy as np
L = os.path.join(os.environ.get("PSM_DATA", "/mnt/home/amohan/psm-data"), "logs")
done = {}
for f in glob.glob(os.path.join(L, "psmz_fix_*_actor_e500_task*.json")):
    m = re.match(r"(psmz_fix_\w+?)_(sd\d+)_(\d+)_actor_e500_task(\d)\.json", os.path.basename(f))
    if m:
        done.setdefault((m[1], int(m[3])), {}).setdefault(m[2], {})[int(m[4])] = float(json.load(open(f))["success"])
for (g, ep), sds in sorted(done.items()):
    full = [v for v in sds.values() if len(v) == 5]
    if not full:
        print(g, ep, "partial:", {s: len(v) for s, v in sds.items()}); continue
    per = np.array([np.mean([v[t] for t in range(1, 6)]) for v in full])
    ci = 4.303 * per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else float("nan")
    print(f"{g} {ep}: {per.mean():.3f} ± {ci:.3f} seeds {np.round(per, 3)} tasks {np.round(np.mean([[v[t] for t in range(1, 6)] for v in full], 0), 2)}")
