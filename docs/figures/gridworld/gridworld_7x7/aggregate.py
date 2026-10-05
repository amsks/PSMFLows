"""Collect runs/*.json into diag_gridworld_psm.json and print a compact table."""
import glob
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
rows = {}
for f in sorted(glob.glob(f'{here}/runs/*.json')):
    r = json.load(open(f))
    rows[os.path.basename(f)[:-5]] = {k: v for k, v in r.items() if k != 'train_log'}

cols = ['corr', 'within_z_corr', 'z_specific_corr', 'slope_learned_on_true', 'rel_l2_raw']
hdr = (f"{'run':22s} {'corr':>6s} {'zcorr':>6s} {'zspec':>6s} {'slope':>6s} {'relL2':>6s} "
       f"{'diagL':>7s} {'diagT':>7s} {'offL':>6s} {'offT':>6s} {'EpL':>6s} {'EpT':>5s} {'maxL':>7s} "
       f"{'atcap':>5s} {'wcos':>6s} {'zcorL':>6s} {'zcorT':>6s} {'rank':>5s} {'actL':>5s} {'actT':>5s} {'spike':>6s}")
print(hdr)
for n, r in rows.items():
    L, T = r['learned'], r['true']
    sp = r.get('spike_rows', {}).get('exact_over_same_cell')

    def f(x, w=6, p=3):
        return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}s}"
    print(f"{n:22s} {f(r['corr'])} {f(r['within_z_corr'])} {f(r['z_specific_corr'])} "
          f"{f(r['slope_learned_on_true'])} {f(r['rel_l2_raw'])} {f(L['diag_mean'],7,2)} {f(T['diag_mean'],7,2)} "
          f"{f(L['offdiag_mean'])} {f(T['offdiag_mean'])} {f(L['rho_avg_mean'])} {f(T['rho_avg_mean'],5,2)} "
          f"{f(L['max'],7,1)} {f(r['frac_at_cap'],5,3)} {f(r['w_mean_pairwise_cos'])} "
          f"{f(r['z_distinct_learned']['pairwise_corr'])} {f(r['z_distinct_true']['pairwise_corr'])} "
          f"{f(r['phi_eff_rank'],5,1)} {f(r['action_share_learned'],5,2)} {f(r['action_share_true'],5,2)} {f(sp)}")

out = {'description': 'Gridworld recovery test of psmgoal successor-measure loss (7x7, uniform-random data, '
                      '100k rows, 30k steps, batch 256, D=64, hidden 256, lr 1e-4, tau 0.01). '
                      'Truth = Mtrue/rho\' (psmgoal) or Mtrue/((1-gamma) rho\') (FB); see EXPECTED.txt.',
       'script': f'{here}/gw_psm.py', 'runs': rows}
json.dump(out, open(f'{here}/diag_gridworld_psm.json', 'w'), indent=1)
