"""Assemble docs/tables/psm_raw_nobc_cube_ladder.md from the eval500 report JSONs.

Pure JSON I/O, like tools/make_tables.py: it reads
`$PSM_DATA/logs/eval500_<group>_<epoch>k_sd<seed>.json` and prints the checkpoint ladder
(one row per 50k epoch, one column per seed, mean +/- sd across seeds) plus the LATE mean
over 300k-500k, which is the only summary the 2026-09-05 non-stationarity finding allows
(no single checkpoint of a PSM arm may be quoted).
"""
import argparse
import glob
import json
import os
import statistics

EPOCHS = list(range(50000, 550000, 50000))


def load(logs, group):
    out = {}
    for path in glob.glob(os.path.join(logs, f'eval500_{group}_*k_sd*.json')):
        base = os.path.basename(path)[len(f'eval500_{group}_'):-len('.json')]
        try:
            ep_s, sd_s = base.split('_sd')
            epoch = int(ep_s.rstrip('k')) * 1000
            seed = int(sd_s)
        except ValueError:
            continue
        with open(path) as f:
            d = json.load(f)
        if d.get('success') is not None:
            out[(epoch, seed)] = (float(d['success']), d.get('wilson95'), int(d.get('episodes', 0)))
    return out


def render(rows, seeds, title, sub):
    lines = [f'### {title}', '', sub, '',
             '| epoch | ' + ' | '.join(f'sd{s}' for s in seeds) + ' | mean ± sd |',
             '|---|' + '---|' * (len(seeds) + 1)]
    for epoch in EPOCHS:
        vals = [rows.get((epoch, s)) for s in seeds]
        if all(v is None for v in vals):
            continue
        cells = [f'{v[0]:.3f}' if v else '—' for v in vals]
        got = [v[0] for v in vals if v]
        agg = (f'{statistics.mean(got):.3f} ± {statistics.stdev(got):.3f}' if len(got) > 1
               else (f'{got[0]:.3f} (n=1)' if got else '—'))
        lines.append(f'| {epoch // 1000}k | ' + ' | '.join(cells) + f' | {agg} |')
    late = [v[0] for (e, _s), v in rows.items() if e >= 300000]
    if late:
        sd = statistics.stdev(late) if len(late) > 1 else 0.0
        lines += ['', (f'**Late mean (300k-500k, n={len(late)} checkpoint x seed measurements, '
                       f'500 episodes each): {statistics.mean(late):.3f}, sd {sd:.3f}, '
                       f'min {min(late):.3f}, max {max(late):.3f}.**')]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default=os.path.join(os.environ.get('PSM_DATA', '.'), 'logs'))
    ap.add_argument('--out', default='docs/tables/psm_raw_nobc_cube_ladder.md')
    ap.add_argument('--seeds', default='0,1,2')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    arm_a_sub = ('Affine measure `M(s,a,x) = Phi(s,a,x)*w + b(s,a,x)` (factored '
                 '`Phi = A(s,a) phi_x(x)`), amortized `ddpgbc` actor, actor loss '
                 '`-Q.mean()`. No pessimism term of any kind.')
    arm_b_sub = ('Bilinear measure `M(s,a,x) = psi(s,z,a)^T phi(x)` (arXiv 2411.19418 '
                 'port), amortized `ddpgbc` actor, actor loss `-Q.mean()` with the '
                 "agent's own `actor_pessimism_penalty=0.5` ensemble-disagreement "
                 'penalty kept on.')
    arms = [('psm_raw_nobc_cube', 'Arm A - `affine_psm`, raw actions, `bc_coeff=0`', arm_a_sub),
            ('psm_bilinear_raw_nobc_cube',
             'Arm B - `psm` (bilinear PSM), raw actions, `bc_coeff=0`', arm_b_sub)]
    parts = []
    for group, title, sub in arms:
        rows = load(args.logs, group)
        if rows:
            parts.append(render(rows, seeds, title, sub))
    body = '\n\n'.join(parts) if parts else '_no eval500 reports found_'
    print(body)
    return body


if __name__ == '__main__':
    main()
