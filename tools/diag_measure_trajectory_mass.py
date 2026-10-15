"""Does the psmgoal measure put mass on the transition's OWN later states, or only on its exact
training next state? (2026-10-15)

diag_measure_fullness read p(x|s,u;g) = softmax_x M(s,u,x;g)/measure_temp over 1024 random
dataset states and found it flat (entropy 6.79 of 6.93), with 1/1024 on the dataset state
nearest the real next state. This tool asks whether the measure transfers along the
transition's own trajectory: for a dataset transition (s_t, u_t, s_{t+1}) it puts the EXACT
later states of the same trajectory into the normalisation set and reads their mass.

Per sampled transition i (a dataset row with at least `min_future` later rows in its own
trajectory): u_t = the row's point preimage latent (u=data) or one fresh clipped prior draw
(u=prior); the goal g = the agent's own hindsight goal (goal_sampling / goal_discount /
goal_random_frac / goal_cur_frac from critic A's flags.json, `hindsight_goal_idxs`); w = h_C(g)
for each critic C in {A, B}. The normalisation set X (n_ref states) is
    n_ref - K - 1 random dataset next states drawn off the row's own trajectory,
    the row's own later states at the K offsets k in `offsets`: next_observations[i + k - 1]
      (k = 1 is the row's own s_{t+1}, the training next state),
    the hindsight goal g itself (when g is one of the offset states the slot is a duplicate: it
      is dropped, one more random state is drawn, and `goal_dup_offset` records which k).
The control set X_ctrl swaps the K own-trajectory slots for the later states at the same
offsets of a DIFFERENT random transition j (same random states, same goal slot).

Per critic, u source, set and offset k: mean +- SE over transitions of p(own state at k), its
ratio to the mean p of the set's random states (1/n_ref is the uniform level), and the fraction
of transitions where that slot ranks in the top 1% / top 5% of X (strict rank: fewer than
ceil(frac * n_ref) entries score above it). The same for the goal slot. The ratio under X_ctrl
is the non-trajectory-specific part (what any same-offset state of another trajectory gets).

Run (GPU):
  .venv/bin/python tools/diag_measure_trajectory_mass.py agent=psmgoal \\
    env_name=cube-single-play-singletask-task1-v0 \\
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \\
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz agent.use_point_preimage=true \\
    restore_path=<run dir A> +restore_path_b=<run dir B> restore_epoch=500000 \\
    +n_states=512 seed=0 report_out=$PSM_DATA/logs/measure_fullness/<tag>.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (imported inside run())
from tools.diag_measure_fullness import measure_to_dist, parse_int_list
from utils.datasets import hindsight_goal_idxs, trajectory_ends

FORCED_KEYS = {'acting': 'gpi', 'coef_source': 'hgoal_each'}
OFFSETS = (1, 2, 5, 10, 20, 50)
TOP_FRACS = (0.01, 0.05)
U_SOURCES = ('data', 'prior')
SETS = ('own', 'ctrl')


# --------------------------------------------------------------------- pure functions
def eligible_rows(terminal_locs, size, min_future):
    """Rows i whose own trajectory has at least `min_future` later rows: end(i) - i >= min_future,
    with end(i) the first entry of `terminal_locs` at or after i (`trajectory_ends`)."""
    idx = np.arange(int(size))
    ends = trajectory_ends(idx, np.asarray(terminal_locs), int(size))
    return idx[ends - idx >= int(min_future)]


def offset_rows(idxs, terminal_locs, size, offsets):
    """The dataset ROW whose next_observation is each sampled row's own state k steps ahead,
    (len(idxs), len(offsets)): row i + k - 1 (k = 1 is the row itself, i.e. its own s'), and a
    bool mask of the same shape that is False where i + k - 1 runs past the trajectory end.
    Rows past the end are clipped to the end (the trajectory's final state), as the hindsight
    goal draw does; callers filter on the mask."""
    idxs = np.asarray(idxs, np.int64)
    offs = np.asarray(offsets, np.int64)
    assert np.all(offs >= 1), 'offsets are >= 1 (1 = the row\'s own next state)'
    ends = trajectory_ends(idxs, np.asarray(terminal_locs), int(size))
    raw = idxs[:, None] + offs[None, :] - 1
    valid = raw <= ends[:, None]
    return np.minimum(raw, ends[:, None]).astype(np.int64), valid


def in_top_fraction(p, slot, frac):
    """True when entry `slot` of p ranks within the top ceil(frac * len(p)) entries, counting
    strictly: fewer than ceil(frac * n) entries score above p[slot]."""
    p = np.asarray(p, np.float64).reshape(-1)
    n_top = int(np.ceil(float(frac) * p.size))
    return bool(np.sum(p > p[slot]) < n_top)


def build_reference_rows(row, own_rows, goal_row, random_rows, n_ref, excluded_rows):
    """Row indices of the normalisation set for one transition: random rows first (drawn from
    `random_rows`, an ordered candidate pool, skipping `excluded_rows`), then the K own-offset
    rows, then the goal row unless it duplicates an own-offset row. Returns (rows (n_ref,),
    slots dict: 'random' slice, 'own' (K,) indices, 'goal' index or None, 'goal_dup_offset' the
    position in own_rows the goal duplicated or None)."""
    own_rows = [int(r) for r in own_rows]
    goal_row = int(goal_row)
    dup = own_rows.index(goal_row) if goal_row in own_rows else None
    special = own_rows + ([] if dup is not None else [goal_row])
    n_random = int(n_ref) - len(special)
    assert n_random >= 1, 'n_ref must exceed the number of special slots'
    excluded = {int(r) for r in excluded_rows} | set(special)
    rand = [int(r) for r in random_rows if int(r) not in excluded][:n_random]
    assert len(rand) == n_random, 'random pool too small after exclusions'
    rows = np.asarray(rand + special, np.int64)
    slots = {'random': slice(0, n_random), 'own': np.arange(n_random, n_random + len(own_rows)),
             'goal': (n_random + dup) if dup is not None else n_random + len(own_rows),
             'goal_dup_offset': dup}
    return rows, slots


def slot_stats(p, slots, top_fracs=TOP_FRACS):
    """Per-slot numbers of one distribution p over the set: p of each own slot and of the goal
    slot, the mean p of the random slots, the top-fraction indicators."""
    p = np.asarray(p, np.float64).reshape(-1)
    out = {'p_random_mean': float(p[slots['random']].mean()), 'p_own': [], 'p_goal': float(p[slots['goal']]),
           'own_top': {str(f): [] for f in top_fracs}, 'goal_top': {}}
    for s in slots['own']:
        out['p_own'].append(float(p[s]))
        for f in top_fracs:
            out['own_top'][str(f)].append(in_top_fraction(p, int(s), f))
    for f in top_fracs:
        out['goal_top'][str(f)] = in_top_fraction(p, int(slots['goal']), f)
    return out


def mean_se(x):
    x = np.asarray(x, np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'mean': None, 'se': None, 'n': 0}
    se = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return {'mean': float(x.mean()), 'se': se, 'n': int(x.size)}


def aggregate(stats, offsets, top_fracs=TOP_FRACS):
    """Over the per-transition `slot_stats` records of one (critic, u source, set): per offset
    and for the goal slot, mean +- SE of p, the ratio of mean p to the mean random-slot p, the
    ratio to uniform (p * n_ref) and the top-fraction rates."""
    if not stats:
        return {}
    p_rand = np.asarray([r['p_random_mean'] for r in stats])
    n_ref = round(1.0 / p_rand.mean()) if p_rand.mean() > 0 else None
    out = {'p_random_mean': mean_se(p_rand), 'per_offset': {}, 'n': len(stats)}
    for j, k in enumerate(offsets):
        p = np.asarray([r['p_own'][j] for r in stats])
        out['per_offset'][str(int(k))] = {
            'p': mean_se(p), 'ratio_to_random': float(p.mean() / max(p_rand.mean(), 1e-300)),
            'ratio_to_random_per_transition': mean_se(p / np.maximum(p_rand, 1e-300)),
            'log10_ratio_median': float(np.median(np.log10(np.maximum(p / np.maximum(p_rand, 1e-300), 1e-300)))),
            **{f'top{f}': float(np.mean([r['own_top'][str(f)][j] for r in stats])) for f in top_fracs},
        }
    p = np.asarray([r['p_goal'] for r in stats])
    out['goal'] = {
        'p': mean_se(p), 'ratio_to_random': float(p.mean() / max(p_rand.mean(), 1e-300)),
        'ratio_to_random_per_transition': mean_se(p / np.maximum(p_rand, 1e-300)),
        'log10_ratio_median': float(np.median(np.log10(np.maximum(p / np.maximum(p_rand, 1e-300), 1e-300)))),
        **{f'top{f}': float(np.mean([r['goal_top'][str(f)] for r in stats])) for f in top_fracs},
    }
    out['n_ref_implied'] = n_ref
    return out


# --------------------------------------------------------------------- the agents
def load_critic(cfg, restore_path, label):
    """Restore one psmgoal checkpoint as tools/eval_state_reward.py does, FORCED_KEYS on top."""
    import ml_collections
    from omegaconf import OmegaConf

    from agents import agents
    from main import _lists_to_tuples
    from tools.eval_checkpoint import _cli_agent_keys, merge_run_config
    from utils.flax_utils import restore_agent

    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, restore_path, _cli_agent_keys())
    assert merged.get('agent_name') == 'psmgoal', f'{label}: diag_measure_trajectory_mass reads psmgoal checkpoints'
    assert str(merged.get('policy_index')) == 'goal', f'{label}: needs policy_index=goal (h(g) coefficient)'
    for k, v in FORCED_KEYS.items():
        merged[k] = v
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    return config, merged, prov, agents, restore_agent


# --------------------------------------------------------------------- main
def run(cfg):
    import jax
    import jax.numpy as jnp

    from envs.env_utils import make_env_and_datasets
    from tools.eval_state_reward import build_dataset
    from utils.log_utils import write_report

    t_start = time.time()
    assert cfg.restore_path is not None, 'restore_path (critic A) is required'
    restore_b = cfg.get('restore_path_b', None)
    n_states = int(cfg.get('n_states', 512))
    n_ref = int(cfg.get('n_ref', 1024))
    offsets = tuple(parse_int_list(cfg.get('offsets', None), OFFSETS))
    min_future = int(cfg.get('min_future', 60))
    chunk = int(cfg.get('state_chunk', 32))
    seed = int(cfg.seed)
    assert min_future >= max(offsets), 'min_future must cover the largest offset'
    K = len(offsets)

    cfg_a, merged_a, prov_a, agents, restore_agent = load_critic(cfg, str(cfg.restore_path), 'A')
    runs = {'A': str(cfg.restore_path)}
    configs, merged, prov = {'A': cfg_a}, {'A': merged_a}, {'A': prov_a}
    if restore_b:
        configs['B'], merged['B'], prov['B'], _, _ = load_critic(cfg, str(restore_b), 'B')
        runs['B'] = str(restore_b)
    loss = str(merged_a.get('measure_loss', 'squared'))
    temp = float(merged_a.get('measure_temp', 1.0))
    print(f'=== diag_measure_trajectory_mass: n_states={n_states} n_ref={n_ref} offsets={offsets} '
          f'min_future={min_future} seed={seed}')
    for lab, m in merged.items():
        print(f'    {lab}={runs[lab]}\n       fb_bc_coeff={m.get("fb_bc_coeff")} bootstrap_source={m.get("bootstrap_source")} '
              f'measure_loss={m.get("measure_loss")} measure_temp={m.get("measure_temp")} '
              f'goal_sampling={m.get("goal_sampling")} goal_discount={m.get("goal_discount")} '
              f'goal_random_frac={m.get("goal_random_frac")} goal_cur_frac={m.get("goal_cur_frac")} '
              f'u_clip={m.get("u_clip")}', flush=True)
    print(f'    epoch={cfg.restore_epoch} forced={FORCED_KEYS}; read: measure_loss={loss} temp={temp}', flush=True)

    # Dataset, the eligible rows, their preimage latents and hindsight goals.
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    _, _, train_dataset, _ = make_env_and_datasets(str(cfg.env_name), frame_stack=cfg.frame_stack)
    ds = build_dataset(train_dataset, cfg_a)
    size = int(ds.size)
    next_obs = np.asarray(ds['next_observations'], np.float32)
    elig = eligible_rows(ds.terminal_locs, size, min_future)
    assert elig.size >= n_states, f'only {elig.size} rows have {min_future} later steps'
    idxs = np.sort(rng.choice(elig, size=n_states, replace=False))
    other = rng.choice(elig, size=n_states, replace=True)
    same = trajectory_ends(other, ds.terminal_locs, size) == trajectory_ends(idxs, ds.terminal_locs, size)
    while np.any(same):  # the control trajectory must be a different one
        other[same] = rng.choice(elig, size=int(same.sum()), replace=True)
        same = trajectory_ends(other, ds.terminal_locs, size) == trajectory_ends(idxs, ds.terminal_locs, size)
    own_rows, own_ok = offset_rows(idxs, ds.terminal_locs, size, offsets)
    ctrl_rows, ctrl_ok = offset_rows(other, ds.terminal_locs, size, offsets)
    assert own_ok.all() and ctrl_ok.all()
    goal_rows, goal_is_random = hindsight_goal_idxs(
        idxs, ds.terminal_locs, size, float(merged_a.get('goal_discount', 0.98)),
        float(merged_a.get('goal_random_frac', 0.0)), rng=rng, cur_frac=float(merged_a.get('goal_cur_frac', 0.0)))
    if str(merged_a.get('goal_sampling', 'geometric')) != 'geometric':
        print(f'    NOTE: critic A trained with goal_sampling={merged_a.get("goal_sampling")}; '
              'goals here are geometric hindsight draws', flush=True)
    goal_offset = goal_rows - idxs + 1
    ends = trajectory_ends(idxs, ds.terminal_locs, size)
    batch = ds.sample(n_states, idxs=idxs)
    obs_np = np.asarray(batch['observations'], np.float32)
    u_clip = float(cfg_a['u_clip'])
    u_data = np.clip(np.asarray(batch['noise_preimage'], np.float32), -u_clip, u_clip)
    d_a = int(u_data.shape[-1])
    u_np = {'data': u_data,
            'prior': np.clip(rng.standard_normal((n_states, d_a)).astype(np.float32), -u_clip, u_clip)}
    np.testing.assert_allclose(np.asarray(batch['next_observations']), next_obs[idxs])

    # The normalisation sets: random rows drawn off the row's own trajectory, own / control slots.
    ref_rows = {s: np.zeros((n_states, n_ref), np.int64) for s in SETS}
    slots = {s: [] for s in SETS}
    for i in range(n_states):
        pool = rng.choice(size, size=n_ref + K + 1 + 64, replace=False)
        excluded = np.arange(idxs[i], ends[i] + 1)
        rows_o, sl_o = build_reference_rows(idxs[i], own_rows[i], goal_rows[i], pool, n_ref, excluded)
        rows_c, sl_c = build_reference_rows(idxs[i], ctrl_rows[i], goal_rows[i], pool, n_ref, excluded)
        ref_rows['own'][i], ref_rows['ctrl'][i] = rows_o, rows_c
        slots['own'].append(sl_o)
        slots['ctrl'].append(sl_c)
    n_goal_dup = int(sum(sl['goal_dup_offset'] is not None for sl in slots['own']))

    ex = ds.sample(1)
    critics = {}
    for lab, run_dir in runs.items():
        ag = restore_agent(agents['psmgoal'].create(seed, ex['observations'], ex['actions'], configs[lab]),
                           run_dir, cfg.restore_epoch)
        assert str(ag.config.get('measure_loss', 'squared')) == loss, f'{lab}: measure_loss differs from A'
        assert float(ag.config['measure_temp']) == temp, f'{lab}: measure_temp differs from A'

        def rows_fn(obs, u, g, X, agent=ag):
            """M(s,u,x;g) over the set X for a chunk: obs (n, ob), u (n, d), g (n, ob), X (n, n_ref, ob)."""
            hw = agent.w_star(g)                                                  # (n, z) = h(g)

            def one(o, uu, Xi, h):
                phi, b = agent.mesh_phi_b(o[None], uu[None], Xi)                  # (1, n_ref, z), (1, n_ref)
                return phi[0] @ h + b[0]
            return jax.vmap(one)(obs, u, X, hw)

        critics[lab] = jax.jit(rows_fn)

    # Measure rows for every (critic, u source, set) in chunks of states.
    goals_np = next_obs[goal_rows]
    stats = {lab: {us: {s: [] for s in SETS} for us in U_SOURCES} for lab in critics}
    per_state_p = {lab: {us: {s: {'own': [], 'goal': [], 'random_mean': []} for s in SETS} for us in U_SOURCES}
                   for lab in critics}
    t_loop = time.time()
    for c0 in range(0, n_states, chunk):
        c1 = min(c0 + chunk, n_states)
        obs_j, g_j = jnp.asarray(obs_np[c0:c1]), jnp.asarray(goals_np[c0:c1])
        for s in SETS:
            X_j = jnp.asarray(next_obs[ref_rows[s][c0:c1]])                         # (n, n_ref, ob)
            for us in U_SOURCES:
                u_j = jnp.asarray(u_np[us][c0:c1])
                for lab, fn in critics.items():
                    m = np.asarray(fn(obs_j, u_j, g_j, X_j), np.float64)             # (n, n_ref)
                    for i in range(c0, c1):
                        p = measure_to_dist(m[i - c0], loss, temp)
                        st = slot_stats(p, slots[s][i])
                        stats[lab][us][s].append(st)
                        per_state_p[lab][us][s]['own'].append(st['p_own'])
                        per_state_p[lab][us][s]['goal'].append(st['p_goal'])
                        per_state_p[lab][us][s]['random_mean'].append(st['p_random_mean'])
        el = time.time() - t_loop
        print(f'  states {c1}/{n_states}  {el:.0f}s', flush=True)

    summary = {lab: {us: {s: aggregate(stats[lab][us][s], offsets) for s in SETS} for us in U_SOURCES}
               for lab in critics}

    # Table: rows k and goal; columns per critic (own ratio, top5%) and the control ratio.
    def cell(lab, us, s, key, field):
        node = summary[lab][us][s]
        rec = node['goal'] if key == 'goal' else node['per_offset'][str(key)]
        return rec[field]

    labs = list(critics)
    print(f'\n==== p(own state | s_t, u_t; g) over X (n_ref={n_ref}, uniform = 1/n_ref); ratio = mean p / mean '
          f'p of a random state; top5% = share of transitions with the slot in the top 5% of X ====')
    print(f'    n={n_states} transitions, goal = hindsight draw (offset median {int(np.median(goal_offset))}, '
          f'{n_goal_dup} goals coincide with an offset slot, {int(goal_is_random.sum())} random goals)')
    for us in U_SOURCES:
        head = f'{"u=" + us:10s}' + ''.join(f'{"ratio " + lab:>11s}{"top5% " + lab:>10s}{"top1% " + lab:>10s}'
                                            for lab in labs) + ''.join(f'{"ctrl " + lab:>10s}' for lab in labs)
        print(head)
        for key in list(offsets) + ['goal']:
            name = 'goal' if key == 'goal' else f'k={key}'
            row = ''.join(f'{cell(lab, us, "own", key, "ratio_to_random"):11.2f}'
                          f'{cell(lab, us, "own", key, "top0.05"):10.3f}{cell(lab, us, "own", key, "top0.01"):10.3f}'
                          for lab in labs)
            row += ''.join(f'{cell(lab, us, "ctrl", key, "ratio_to_random"):10.2f}' for lab in labs)
            print(f'{name:10s}{row}')
    report = {
        'tool': 'diag_measure_trajectory_mass',
        'what': __doc__.split('\n\n')[0],
        'runs': runs, 'restore_epoch': int(cfg.restore_epoch), 'seed': seed, 'env_name': str(cfg.env_name),
        'n_states': n_states, 'n_ref': n_ref, 'offsets': [int(k) for k in offsets], 'min_future': min_future,
        'top_fracs': list(TOP_FRACS), 'u_sources': list(U_SOURCES), 'sets': list(SETS),
        'measure_loss': loss, 'measure_temp': temp, 'u_clip': u_clip, 'action_dim': d_a,
        'forced_keys': FORCED_KEYS, 'n_goal_dup_offset': n_goal_dup,
        'goal_sampling': {k: merged_a.get(k) for k in ('goal_sampling', 'goal_discount', 'goal_random_frac',
                                                       'goal_cur_frac')},
        'rows': idxs.tolist(), 'control_rows': other.tolist(), 'goal_rows': goal_rows.tolist(),
        'goal_offset': goal_offset.tolist(), 'goal_is_random': goal_is_random.astype(int).tolist(),
        'trajectory_ends': ends.tolist(),
        'goal_dup_offset': [sl['goal_dup_offset'] for sl in slots['own']],
        'summary': summary,
        'per_state': per_state_p,
        'agent_config': {lab: json.loads(json.dumps(merged[lab], default=str)) for lab in merged},
        'config_provenance': prov,
        'wall_seconds': round(time.time() - t_start, 1),
    }
    write_report(report, cfg, 'diag_measure_trajectory_mass.json')


if __name__ == '__main__':
    import hydra

    @hydra.main(version_base=None, config_path='../configs', config_name='config')
    def _main(cfg):
        run(cfg)

    _main()
