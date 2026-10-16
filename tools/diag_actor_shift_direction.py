"""Where does the flowbc actor's shift point? (2026-10-17)

The actor's bootstrap latent is u' = clip(eps + delta(s, g, eps)). diag_actor_mechanism (10-15)
measured the shift's size (||delta|| 0.09 against ||eps|| 2.1), that 82% of its second moment is
a per-(s, g) constant, and its cosine with the critic's gradient in u (0.30). It did not measure
what the shift points AT. This tool does, on the same sampled (s, g) rows and the same critic A.

Per sampled (s, g) -- n_states dataset rows with their point preimage u_data, their recorded
action a_data and a hindsight goal drawn by the agent's own goal sampling -- with critic A's
online basis and its actor delta:

  1. reference directions at the state, in latent space:
       c_top    centroid of the top top_frac of n_draws clipped prior draws under the point
                value q_A(s, u; g) (the deployed hgoal_each share on one goal);
       u_data   the clipped point preimage of the recorded action;
       gbar     the mean over n_grad draws of d q_A / d u at the BC draw;
       outward  the draw itself, eps_i (a positive cosine = the shift lengthens the draw).
     dbar = mean over eps of delta(s, g, eps) (the constant part). Reported: cos(dbar, c_top),
     cos(dbar, u_data), cos(dbar, gbar), the length of dbar's projection on unit(c_top) over
     ||dbar||, and the pairwise cosines between the references (cos(c_top, u_data),
     cos(gbar, c_top)) so the reader knows whether the references agree with each other.
     Per draw: cos(delta_i, c_top - u_bc_i), cos(delta_i, u_data - u_bc_i), cos(delta_i, grad_i),
     cos(delta_i, u_bc_i); each with the fraction > 0. Random reference: the mean |cos| of two
     random directions in d dims (the mean cos is 0 and the fraction > 0 is 0.5 by symmetry).
  2. goal dependence: at the same s, dbar for n_goals goals = [the hindsight goal] + random
     dataset next states. The share of the shift's variance that is across goals at a fixed
     state against across states (mean over goals), and cos(dbar(s, g_hind), dbar(s, g_rand)).
  3. action space: decode u_bc and u' through the frozen flow for n_dec draws. Per action
     dimension (cube: pos_x, pos_y, pos_z, ori, gripper) the mean and mean absolute change
     a' - a_bc; cos(a' - a_bc, a_data - a_bc) (toward the recorded action) and
     cos(a' - a_bc, a_top - a_bc) (toward the decoded centroid of the top set), each with the
     fraction > 0; the gripper sign-flip rate between a_bc and a' and the rate at which the
     flip is toward a_data's gripper sign.

Everything is persisted (per-state arrays included) via report_out. The checkpoint is
restored as diag_actor_mechanism does (flags.json defaults, typed agent.* overrides on top,
acting=gpi and coef_source=hgoal_each forced; neither key changes a parameter shape).

Run (GPU):
  .venv/bin/python tools/diag_actor_shift_direction.py agent=psmgoal \\
    env_name=cube-single-play-singletask-task1-v0 \\
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \\
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz agent.use_point_preimage=true \\
    restore_path=<run dir A> restore_epoch=500000 \\
    +n_states=512 +n_draws=2000 seed=0 report_out=$PSM_DATA/logs/actor_shift/<tag>.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax (imported inside run())
from tools.diag_actor_mechanism import (
    FORCED_KEYS,
    chunked,
    cosine_rows,
    load_critic,
    make_fns,
    mean_explained_fraction,
    summarise_rows,
)

ACTION_DIM_NAMES = ('pos_x', 'pos_y', 'pos_z', 'ori', 'gripper')
GRIPPER_DIM = 4


# ------------------------------------------------------------------ pure helpers (CPU-tested)
def unit(v):
    v = np.asarray(v, np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def cosine(a, b):
    """Cosine between two vectors."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def random_cosine_reference(d, rng, n=20000):
    """Mean |cos| between two random directions in d dims (mean cos is 0 by symmetry)."""
    a = rng.standard_normal((n, d))
    b = rng.standard_normal((n, d))
    return float(np.abs(cosine_rows(a, b)).mean())


def shift_direction_stats(delta, eps, u_bc, c_top, u_data, grad):
    """Where the shift points, at one (s, g).

    delta (N, d): the actor's step at each draw; eps (N, d): the unclipped draws; u_bc (N, d):
    the clipped draws; c_top (d,): centroid of the critic's top set; u_data (d,): the data
    latent; grad (G, d): d q / d u at u_bc[:G]. Returns a flat dict of scalars."""
    delta, eps, u_bc = (np.asarray(x, np.float64) for x in (delta, eps, u_bc))
    c_top, u_data, grad = np.asarray(c_top, np.float64), np.asarray(u_data, np.float64), np.asarray(grad, np.float64)
    dbar = delta.mean(0)
    gbar = grad.mean(0)
    G = grad.shape[0]
    cos_top = cosine_rows(delta, c_top[None, :] - u_bc)
    cos_data = cosine_rows(delta, u_data[None, :] - u_bc)
    cos_grad = cosine_rows(delta[:G], grad)
    cos_out = cosine_rows(delta, u_bc)
    return {
        'dbar_norm': float(np.linalg.norm(dbar)),
        'delta_norm': float(np.linalg.norm(delta, axis=1).mean()),
        'mean_explained': mean_explained_fraction(delta),
        'c_top_norm': float(np.linalg.norm(c_top)),
        'u_data_norm': float(np.linalg.norm(u_data)),
        # the constant part of the shift against the references
        'cos_dbar_ctop': cosine(dbar, c_top),
        'cos_dbar_udata': cosine(dbar, u_data),
        'cos_dbar_gbar': cosine(dbar, gbar),
        'proj_dbar_on_ctop': float(dbar @ unit(c_top) / max(np.linalg.norm(dbar), 1e-12)),
        # do the references agree with each other?
        'cos_ctop_udata': cosine(c_top, u_data),
        'cos_gbar_ctop': cosine(gbar, c_top),
        'cos_gbar_udata': cosine(gbar, u_data),
        # per draw: the step against the direction from that draw to the reference
        'cos_delta_to_ctop': float(cos_top.mean()), 'frac_pos_to_ctop': float(np.mean(cos_top > 0)),
        'cos_delta_to_udata': float(cos_data.mean()), 'frac_pos_to_udata': float(np.mean(cos_data > 0)),
        'cos_delta_grad': float(cos_grad.mean()), 'frac_pos_grad': float(np.mean(cos_grad > 0)),
        'cos_delta_outward': float(cos_out.mean()), 'frac_pos_outward': float(np.mean(cos_out > 0)),
    }


def goal_variance_split(shifts):
    """shifts (S, K, d): dbar at state s for goal k, k = 0 the hindsight goal, k >= 1 random goals.
    Returns the share of the total variance of dbar that is across goals at a fixed state
    (1 = the shift is all goal, 0 = all state), and the mean cosine between the hindsight-goal
    shift and each random-goal shift at the same state."""
    shifts = np.asarray(shifts, np.float64)
    S, K, d = shifts.shape
    grand = shifts.reshape(S * K, d).mean(0)
    total = float(((shifts - grand) ** 2).sum(-1).mean())
    state_mean = shifts.mean(1, keepdims=True)                                     # (S, 1, d)
    within = float(((shifts - state_mean) ** 2).sum(-1).mean())                      # across goals, fixed s
    across = float(((state_mean[:, 0, :] - grand) ** 2).sum(-1).mean())              # across states
    cos = []
    for s in range(S):
        for k in range(1, K):
            cos.append(cosine(shifts[s, 0], shifts[s, k]))
    return {
        'var_total': total, 'var_across_goals_fixed_state': within, 'var_across_states': across,
        'goal_share': float(within / max(total, 1e-20)),
        'cos_hind_vs_random_goal': float(np.mean(cos)) if cos else 0.0,
        'frac_pos_hind_vs_random_goal': float(np.mean(np.asarray(cos) > 0)) if cos else 0.0,
    }


def action_shift_stats(a_bc, a_act, a_data, a_top, gripper_dim=GRIPPER_DIM):
    """What the shift does in action space, at one (s, g). a_bc, a_act (N, A): decoded BC
    draws and decoded actor latents (same eps); a_data (A,): the recorded action; a_top (A,):
    the decoded centroid of the critic's top set."""
    a_bc, a_act = np.asarray(a_bc, np.float64), np.asarray(a_act, np.float64)
    a_data, a_top = np.asarray(a_data, np.float64), np.asarray(a_top, np.float64)
    diff = a_act - a_bc
    to_data = a_data[None, :] - a_bc
    to_top = a_top[None, :] - a_bc
    cos_data = cosine_rows(diff, to_data)
    cos_top = cosine_rows(diff, to_top)
    out = {
        'action_shift_norm': float(np.linalg.norm(diff, axis=1).mean()),
        'action_bc_to_data_norm': float(np.linalg.norm(to_data, axis=1).mean()),
        'cos_action_to_data': float(cos_data.mean()), 'frac_pos_action_to_data': float(np.mean(cos_data > 0)),
        'cos_action_to_top': float(cos_top.mean()), 'frac_pos_action_to_top': float(np.mean(cos_top > 0)),
    }
    for j in range(diff.shape[1]):
        name = ACTION_DIM_NAMES[j] if j < len(ACTION_DIM_NAMES) else f'dim{j}'
        out[f'shift_mean_{name}'] = float(diff[:, j].mean())
        out[f'shift_abs_{name}'] = float(np.abs(diff[:, j]).mean())
    if gripper_dim is not None and gripper_dim < diff.shape[1]:
        s_bc, s_act = np.sign(a_bc[:, gripper_dim]), np.sign(a_act[:, gripper_dim])
        flip = s_bc != s_act
        toward = flip & (s_act == np.sign(a_data[gripper_dim]))
        out['gripper_flip_rate'] = float(flip.mean())
        out['gripper_flip_toward_data_rate'] = float(toward.mean())
        out['gripper_bc_agrees_data'] = float(np.mean(s_bc == np.sign(a_data[gripper_dim])))
        out['gripper_actor_agrees_data'] = float(np.mean(s_act == np.sign(a_data[gripper_dim])))
    return out


# ------------------------------------------------------------------ main
def run(cfg):
    import jax
    import jax.numpy as jnp

    from envs.env_utils import make_env_and_datasets
    from tools.eval_state_reward import build_dataset
    from utils.log_utils import write_report

    t_start = time.time()
    assert cfg.restore_path is not None, 'restore_path (critic A, the flowbc actor) is required'
    n_states = int(cfg.get('n_states', 512))
    n_draws = int(cfg.get('n_draws', 2000))
    n_pool = int(cfg.get('n_pool', 224))
    n_grad = int(cfg.get('n_grad_draws', 256))
    n_goals = int(cfg.get('n_goals', 8))
    n_goal_draws = int(cfg.get('n_goal_draws', 256))
    n_dec = int(cfg.get('n_dec', 256))
    top_frac = float(cfg.get('top_frac', 0.05))
    chunk = int(cfg.get('draw_chunk', 500))
    seed = int(cfg.seed)
    n_top = max(int(np.ceil(top_frac * n_draws)), 2)
    n_grad, n_dec = min(n_grad, n_draws), min(n_dec, n_draws)

    cfg_a, merged_a, prov_a, agents, restore_agent = load_critic(cfg, str(cfg.restore_path), 'A')
    print(f'=== diag_actor_shift_direction: n_states={n_states} n_draws={n_draws} n_pool={n_pool} top={n_top} '
          f'n_goals={n_goals} n_dec={n_dec} seed={seed}')
    print(f'    A={cfg.restore_path} epoch={cfg.restore_epoch} forced={FORCED_KEYS}')
    print(f'    fb_bc_coeff={merged_a.get("fb_bc_coeff")} bootstrap_source={merged_a.get("bootstrap_source")} '
          f'measure_loss={merged_a.get("measure_loss")} goal_sampling={merged_a.get("goal_sampling")} '
          f'goal_discount={merged_a.get("goal_discount")} u_clip={merged_a.get("u_clip")}', flush=True)

    np.random.seed(seed)
    _, _, train_dataset, _ = make_env_and_datasets(str(cfg.env_name), frame_stack=cfg.frame_stack)
    ds = build_dataset(train_dataset, cfg_a)
    ds.return_goals = True
    ds.goal_sampling = str(merged_a.get('goal_sampling', 'geometric'))
    ds.goal_discount = float(merged_a.get('goal_discount', 0.98))
    ds.goal_random_frac = float(merged_a.get('goal_random_frac', 0.3))
    ds.goal_cur_frac = float(merged_a.get('goal_cur_frac', 0.0))
    ex = ds.sample(1)
    batch = ds.sample(n_states)
    pool_np = np.asarray(ds.sample(n_pool)['next_observations'], np.float32)
    rand_goals_np = np.asarray(ds.sample(n_states * max(n_goals - 1, 0))['next_observations'], np.float32)
    obs_np = np.asarray(batch['observations'], np.float32)
    goals_np = np.asarray(batch['goals'], np.float32)
    a_data_np = np.asarray(batch['actions'], np.float32)
    u_clip = float(cfg_a['u_clip'])
    u_data_np = np.clip(np.asarray(batch['noise_preimage'], np.float32), -u_clip, u_clip)
    d_a = u_data_np.shape[-1]

    agent = restore_agent(agents['psmgoal'].create(seed, ex['observations'], ex['actions'], cfg_a),
                          str(cfg.restore_path), cfg.restore_epoch)
    q_fn, grad_fn, actor_fn, clip_fn = make_fns(agent)
    bp, wp = agent.basis.params, agent.w_star.params

    @jax.jit
    def decode_fn(obs1, u):
        obs = jnp.broadcast_to(obs1, (u.shape[0], obs1.shape[-1]))
        return agent.decode(obs, u)

    pool = jnp.asarray(pool_np)
    key = jax.random.PRNGKey(seed)
    cos_ref = random_cosine_reference(d_a, np.random.default_rng(seed + 1))

    dir_rows, act_rows = [], []
    shifts = np.zeros((n_states, n_goals, d_a), np.float64)
    arrays = {k: [] for k in ('cos_dbar_ctop', 'cos_dbar_udata', 'cos_dbar_gbar', 'cos_ctop_udata',
                              'dbar_norm', 'cos_action_to_data', 'cos_action_to_top', 'gripper_flip_rate')}
    t_loop = time.time()
    for i in range(n_states):
        key, k_eps, k_goal = jax.random.split(key, 3)
        obs1, goal1 = jnp.asarray(obs_np[i]), jnp.asarray(goals_np[i])
        eps = jax.random.normal(k_eps, (n_draws, d_a))
        u_bc = clip_fn(eps)
        u_bc_np = np.asarray(u_bc)
        # 1. references and the shift
        q_all = chunked(q_fn, u_bc, chunk, bp, wp, obs1, goal1, pool)
        top_idx = np.argsort(-q_all, kind='stable')[:n_top]
        c_top = u_bc_np[top_idx].mean(0)
        grad = np.asarray(grad_fn(bp, wp, obs1, goal1, u_bc[:n_grad], pool))
        u_act, delta = actor_fn(agent.delta.params, obs1, goal1, eps)
        delta_np = np.asarray(delta)
        row = shift_direction_stats(delta_np, np.asarray(eps), u_bc_np, c_top, u_data_np[i], grad)
        row['q_top_mean'] = float(q_all[top_idx].mean())
        row['q_mean'] = float(q_all.mean())
        dir_rows.append(row)
        # 2. goal dependence: the same state, the hindsight goal then random goals, fresh draws
        eps_g = jax.random.normal(k_goal, (n_goal_draws, d_a))
        shifts[i, 0] = np.asarray(actor_fn(agent.delta.params, obs1, goal1, eps_g)[1]).mean(0)
        for k in range(1, n_goals):
            g_k = jnp.asarray(rand_goals_np[i * (n_goals - 1) + (k - 1)])
            shifts[i, k] = np.asarray(actor_fn(agent.delta.params, obs1, g_k, eps_g)[1]).mean(0)
        # 3. action space
        a_bc = np.asarray(decode_fn(obs1, u_bc[:n_dec]))
        a_act = np.asarray(decode_fn(obs1, u_act[:n_dec]))
        a_top = np.asarray(decode_fn(obs1, jnp.asarray(u_bc_np[top_idx]))).mean(0)
        act_rows.append(action_shift_stats(a_bc, a_act, a_data_np[i], a_top))
        for k, col in arrays.items():
            col.append(row[k] if k in row else act_rows[-1][k])
        if i == 0 or (i + 1) % 32 == 0 or i + 1 == n_states:
            el = time.time() - t_loop
            print(f'  state {i + 1}/{n_states}  {el:.0f}s  ({el / (i + 1):.2f}s/state)', flush=True)

    summary = {
        'direction': summarise_rows(dir_rows),
        'goal': goal_variance_split(shifts),
        'action': summarise_rows(act_rows),
        'random_abs_cos_reference': cos_ref,
    }

    def ms(sec, k, nd=3):
        v = summary[sec].get(k)
        return 'n/a' if v is None else f'{v["mean"]:.{nd}f}±{v["sem"]:.{nd}f}'

    print('\n=== where the shift points (critic A, online basis)')
    print(f'random reference: mean |cos| of two random directions in {d_a} dims = {cos_ref:.3f}; '
          f'mean cos 0, frac>0 0.5')
    print(f'constant part dbar: norm {ms("direction", "dbar_norm")} (per-draw delta norm '
          f'{ms("direction", "delta_norm")}, mean_explained {ms("direction", "mean_explained")})')
    print(f'  cos(dbar, c_top)  {ms("direction", "cos_dbar_ctop")}   proj on unit(c_top)/|dbar| '
          f'{ms("direction", "proj_dbar_on_ctop")}')
    print(f'  cos(dbar, u_data) {ms("direction", "cos_dbar_udata")}')
    print(f'  cos(dbar, gbar)   {ms("direction", "cos_dbar_gbar")}')
    print(f'  references: cos(c_top, u_data) {ms("direction", "cos_ctop_udata")}  cos(gbar, c_top) '
          f'{ms("direction", "cos_gbar_ctop")}  cos(gbar, u_data) {ms("direction", "cos_gbar_udata")}  '
          f'|c_top| {ms("direction", "c_top_norm", 2)}  |u_data| {ms("direction", "u_data_norm", 2)}')
    print(f'per draw: to c_top {ms("direction", "cos_delta_to_ctop")} (frac>0 {ms("direction", "frac_pos_to_ctop")}), '
          f'to u_data {ms("direction", "cos_delta_to_udata")} (frac>0 {ms("direction", "frac_pos_to_udata")}), '
          f'grad {ms("direction", "cos_delta_grad")} (frac>0 {ms("direction", "frac_pos_grad")}), '
          f'outward {ms("direction", "cos_delta_outward")} (frac>0 {ms("direction", "frac_pos_outward")})')
    g = summary['goal']
    print(f'goal dependence: share of shift variance across goals at a fixed state {g["goal_share"]:.3f} '
          f'(across states {g["var_across_states"]:.2e}, across goals {g["var_across_goals_fixed_state"]:.2e}); '
          f'cos(dbar hindsight goal, dbar random goal) {g["cos_hind_vs_random_goal"]:.3f} '
          f'(frac>0 {g["frac_pos_hind_vs_random_goal"]:.3f})')
    print(f'action space: |a\' - a_bc| {ms("action", "action_shift_norm")} vs |a_data - a_bc| '
          f'{ms("action", "action_bc_to_data_norm")}; cos to data {ms("action", "cos_action_to_data")} '
          f'(frac>0 {ms("action", "frac_pos_action_to_data")}); cos to top centroid '
          f'{ms("action", "cos_action_to_top")} (frac>0 {ms("action", "frac_pos_action_to_top")})')
    for name in ACTION_DIM_NAMES[:a_data_np.shape[-1]]:
        print(f'  {name:8s} mean shift {ms("action", f"shift_mean_{name}", 4)}  mean |shift| '
              f'{ms("action", f"shift_abs_{name}", 4)}')
    print(f'  gripper flip rate {ms("action", "gripper_flip_rate")}, toward data '
          f'{ms("action", "gripper_flip_toward_data_rate")}; sign agrees with data: bc '
          f'{ms("action", "gripper_bc_agrees_data")} actor {ms("action", "gripper_actor_agrees_data")}')

    report = {
        'tool': 'diag_actor_shift_direction',
        'what': 'Where the flowbc actor\'s shift delta(s, g, eps) points: against the critic\'s top-set centroid, '
                'the data latent, the critic gradient and the draw itself; its goal dependence; its effect on '
                'the decoded action.',
        'restore_path': str(cfg.restore_path), 'restore_epoch': int(cfg.restore_epoch), 'seed': seed,
        'env_name': str(cfg.env_name),
        'n_states': n_states, 'n_draws': n_draws, 'n_pool': n_pool, 'n_top': n_top, 'top_frac': top_frac,
        'n_grad_draws': n_grad, 'n_goals': n_goals, 'n_goal_draws': n_goal_draws, 'n_dec': n_dec,
        'action_dim': int(a_data_np.shape[-1]), 'u_clip': u_clip, 'action_dim_names': list(ACTION_DIM_NAMES),
        'forced_keys': FORCED_KEYS,
        'summary': summary,
        'per_state': {'direction': dir_rows, 'action': act_rows},
        'arrays': {k: [float(x) for x in v] for k, v in arrays.items()},
        'shifts_per_state_goal': shifts.tolist(),
        'agent_config': json.loads(json.dumps(merged_a, default=str)),
        'config_provenance': prov_a,
        'wall_seconds': round(time.time() - t_start, 1),
    }
    write_report(report, cfg, 'diag_actor_shift_direction.json')


if __name__ == '__main__':
    import hydra

    @hydra.main(version_base=None, config_path='../configs', config_name='config')
    def _main(cfg):
        run(cfg)

    _main()
