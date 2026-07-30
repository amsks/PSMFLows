"""Figure: what a flow preimage IS, and why `inversion.alpha` decides whether it is usable.

Four panels, all grounded in the real cube artifacts — the 1M-row preimage npz for the
stored quantities, the Stage-A checkpoint for the flow map itself:

  A  the preimage BASIN. A 2D slice of latent space at one real dataset transition, shaded
     by the inversion target exp(-alpha*||G(s,u) - a||) at the as-run alpha=1. The stored
     point preimage, the stored posterior mean and its 2-sigma ellipse, the N(0,I) prior and
     the u_clip bound are drawn on top. The point of the panel: the preimage is a broad
     BASIN, not a point, and at alpha=1 it is far wider than the prior.
  B  the same slice, same geometry, at alpha=20. The basin contracts to prior scale. A and B
     together are the whole alpha argument.
  C  the aggregate view over all 1M rows: ||u|| for the point arm and for the posterior mean,
     against the chi_d reference the latents should follow if the flow maps data to N(0,I).
  D  the alpha sweep that picks the value: mean ESS against the D3 gate.

Run (GPU, ~5 min):
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
  .venv/bin/python tools/plot_preimage_intuition.py [--out PATH] [--npz PATH]

Reads the flow checkpoint from the npz's own .meta.json sidecar, so the figure can never
be drawn against a different flow than the one that produced the preimages.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax

import jax
import jax.numpy as jnp
import matplotlib as mpl
import ml_collections
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy import stats

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

from agents.fql import FQLAgent
from main import _lists_to_tuples
from utils.flax_utils import restore_agent

# --- design tokens (dataviz skill reference palette; validated categorical slots 1-2) ---
SURFACE = '#fcfcfb'
INK, INK2, INK3 = '#0b0b0b', '#52514e', '#8a8880'
S1, S2 = '#2a78d6', '#eb6834'          # blue, orange
GRID = '#e6e5e1'
# sequential = ONE hue, light -> dark (blue ramp steps 100..700). No rainbow.
BLUES = LinearSegmentedColormap.from_list('viz_blue', [
    '#f4f8fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
DIVERGED = '#efe9e4'                   # neutral: region where the flow itself blows up

# +-8 rather than +-5: at alpha=1 the basin is so broad that a tighter window is a solid
# wash with its 50%-of-peak edge off-panel, which reads as "no data" instead of "very wide".
GRID_HALF, GRID_N = 8.0, 141
ALPHA_LO, ALPHA_HI = 1.0, 20.0
SWEEP_ALPHAS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
SWEEP_ROWS = 512
ESS_GATE = 20.0


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc='left', pad=8, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='/var/local/amsks/exp/PSMFLows/preimages_cube_single.npz')
    ap.add_argument('--out', default='assets/preimage_intuition.png')
    args = ap.parse_args()

    meta = json.load(open(args.npz + '.meta.json'))
    inv = meta['inversion']
    z = np.load(args.npz)
    obs_all, act_all = z['observations'], z['actions']
    mu_all, cov_all = z['noise_preimage_mean'], z['noise_preimage_cov']
    point_all, ess_all = z['noise_preimage_point'], z['preimage_ess']
    d_a = act_all.shape[-1]

    finite = (np.isfinite(mu_all).all(axis=(1, 2)) & np.isfinite(cov_all).all(axis=(1, 2, 3))
              & np.isfinite(point_all).all(axis=1))
    print(f'{args.npz}: n={len(act_all):,}  finite={finite.sum():,}  d_a={d_a}')

    # ---- the flow that produced these preimages, per the sidecar ----
    with initialize_config_dir(config_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs'),
            version_base=None):
        cfg = compose(config_name='config', overrides=[
            'agent=fql', f'env_name={meta["env_name"]}',
            f'agent.flow_steps={meta["flow_steps"]}'])
    agent_cfg = ml_collections.ConfigDict(_lists_to_tuples(
        OmegaConf.to_container(cfg.agent, resolve=True)))
    agent = FQLAgent.create(0, obs_all[:1], act_all[:1], agent_cfg)
    agent = restore_agent(agent, meta['restore_path'], meta['restore_epoch'])
    print(f'flow: {meta["restore_path"]} @ {meta["restore_epoch"]}, '
          f'flow_steps={meta["flow_steps"]}')

    # ---- pick ONE representative transition: finite, and median ESS among finite rows ----
    fin_idx = np.flatnonzero(finite)
    row = int(fin_idx[np.argsort(ess_all[fin_idx])[len(fin_idx) // 2]])
    state, action = obs_all[row], act_all[row]
    u_pt, mu, cov = point_all[row], mu_all[row, 0], cov_all[row, 0]
    print(f'panel A/B transition: row {row}, ESS={ess_all[row]:.2f} (median of finite rows)')

    # ---- 2D slice: the two FLATTEST directions of J = d(action)/d(noise) ----
    _, jac = agent._get_preimage_and_jacobian(jnp.asarray(state), jnp.asarray(action),
                                              inv['n_initial_steps'])
    U_, sv, Vt = np.linalg.svd(np.asarray(jac))
    v1, v2 = Vt[-1], Vt[-2]              # smallest two singular directions
    print(f'  J singular values {np.array2string(sv, precision=4)}  '
          f'cond={sv[0] / sv[-1]:.2f} (near-isotropic, so the slice is representative)')

    g = np.linspace(-GRID_HALF, GRID_HALF, GRID_N)
    X, Y = np.meshgrid(g, g)
    # slice is centred on the exact point preimage
    U = u_pt[None, None, :] + X[..., None] * v1[None, None, :] + Y[..., None] * v2[None, None, :]
    Uf = U.reshape(-1, d_a).astype(np.float32)
    dist = np.empty(len(Uf), np.float32)
    B = 8192
    for s in range(0, len(Uf), B):
        ob = jnp.broadcast_to(jnp.asarray(state), (len(Uf[s:s + B]), state.shape[0]))
        a_hat = np.asarray(agent.compute_flow_actions(ob, noises=jnp.asarray(Uf[s:s + B])))
        dist[s:s + B] = np.linalg.norm(a_hat - np.clip(action, -1, 1)[None], axis=-1)
    dist = dist.reshape(GRID_N, GRID_N)
    diverged = ~np.isfinite(dist)
    print(f'  grid {GRID_N}x{GRID_N}: flow diverged on {diverged.mean():.2%} of it')

    # projections of the stored quantities into the slice basis
    def proj(v):
        d = v - u_pt
        return float(d @ v1), float(d @ v2)
    mu_xy = proj(mu)
    P = np.stack([v1, v2])                       # (2, d_a)
    cov2 = P @ cov @ P.T                         # marginal of the stored posterior in-slice
    ev2, evec2 = np.linalg.eigh(cov2)
    prior_xy = proj(np.zeros(d_a))               # where u = 0 (the prior's centre) sits

    fig = plt.figure(figsize=(12.8, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30,
                          left=0.068, right=0.935, top=0.835, bottom=0.085)

    # ================= A / B: the basin, at two temperatures =================
    for k, (alpha, ax_pos, tag) in enumerate([
            (ALPHA_LO, gs[0, 0], 'A'), (ALPHA_HI, gs[0, 1], 'B')]):
        ax = fig.add_subplot(ax_pos)
        dens = np.exp(-alpha * np.where(diverged, np.nan, dist))
        dens = dens / np.nanmax(dens)            # each panel to its own peak: compare SHAPE
        ax.imshow(np.where(diverged, 1.0, 0.0), extent=[-GRID_HALF, GRID_HALF] * 2,
                  origin='lower', cmap=LinearSegmentedColormap.from_list(
                      'dv', [(0, 0, 0, 0), DIVERGED]), vmin=0, vmax=1, interpolation='nearest')
        im = ax.imshow(dens, extent=[-GRID_HALF, GRID_HALF] * 2, origin='lower',
                       cmap=BLUES, vmin=0, vmax=1, interpolation='bilinear')
        # the 50%-of-peak contour is "the preimage set" at this temperature
        ax.contour(X, Y, np.nan_to_num(dens), levels=[0.5], colors=[S1],
                   linewidths=2.0, linestyles='-')

        # prior: unit circle around u=0, and the u_clip bound
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(prior_xy[0] + np.cos(th), prior_xy[1] + np.sin(th), color=INK2,
                lw=1.4, ls='--', zorder=4)
        ax.plot(prior_xy[0] + 3 * np.cos(th), prior_xy[1] + 3 * np.sin(th), color=INK3,
                lw=1.0, ls=':', zorder=4)

        # stored posterior: 2-sigma ellipse of the in-slice marginal
        ang = np.degrees(np.arctan2(evec2[1, -1], evec2[0, -1]))
        ax.add_patch(Ellipse(mu_xy, 4 * np.sqrt(max(ev2[-1], 0)), 4 * np.sqrt(max(ev2[0], 0)),
                             angle=ang, fill=False, edgecolor=S2, lw=2.0, zorder=5))
        ax.plot(*mu_xy, marker='X', ms=9, color=S2, mec=SURFACE, mew=1.5, zorder=6)
        ax.plot(0, 0, marker='*', ms=15, color=INK, mec=SURFACE, mew=1.5, zorder=6)

        title = (f'{tag}.  the preimage is a basin, not a point   ·   $\\alpha$ = 1 (as run)'
                 if k == 0 else f'{tag}.  the same basin at $\\alpha$ = 20')
        _style(ax, title, 'latent offset along the flattest direction of $J$',
               'offset along the 2nd-flattest direction' if k == 0 else None)
        ax.set_xlim(-GRID_HALF, GRID_HALF); ax.set_ylim(-GRID_HALF, GRID_HALF)
        ax.set_aspect('equal')
        if k == 0:
            # Data coords, each in its own quadrant, so the four labels cannot overlap each
            # other or run off the axes the way offset-point placement did.
            ax.text(0, -1.9, 'exact point preimage $E(s,a)$', ha='center', va='top',
                    color='#ffffff', fontsize=8)
            ax.annotate('posterior mean $\\mu$ + 2$\\sigma$', mu_xy,
                        textcoords='offset points', xytext=(30, 30), color=S2, fontsize=8,
                        arrowprops=dict(arrowstyle='-', color=S2, lw=0.9))
            ax.annotate('$N(0,I)$ prior, 1$\\sigma$',
                        (prior_xy[0] - 0.75, prior_xy[1] + 0.75), xytext=(-4.3, 3.2),
                        ha='center', va='center', color='#ffffff', fontsize=8,
                        arrowprops=dict(arrowstyle='-', color='#ffffff', lw=0.9))
            ax.annotate('$u$_clip = 3', (prior_xy[0] - 2.12, prior_xy[1] + 2.12),
                        xytext=(-1.4, 5.9), ha='center', va='center', color='#ffffff',
                        fontsize=8, arrowprops=dict(arrowstyle='-', color='#ffffff', lw=0.9))
        else:
            ax.text(0.46, 0.975, 'only the temperature changed',
                    transform=ax.transAxes, ha='center', va='top', fontsize=8, color=INK2)
            cax = ax.inset_axes([1.06, 0.0, 0.03, 1.0])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label('inversion target, relative to its own peak', color=INK2, fontsize=8)
            cb.ax.tick_params(colors=INK2, labelsize=7.5, length=2)
            cb.outline.set_visible(False)
        if diverged.any():
            ax.scatter([], [], marker='s', s=42, color=DIVERGED,
                       label='flow itself diverges')
            ax.plot([], [], color=S1, lw=2.0, label='half-peak contour')
            ax.legend(loc='lower left', frameon=True, facecolor=SURFACE, edgecolor='none',
                      framealpha=0.85, fontsize=7.5, labelcolor=INK2, handletextpad=0.5,
                      borderpad=0.4)

    # ================= C: aggregate over every row in the npz =================
    ax = fig.add_subplot(gs[1, 0])
    pt_n = np.linalg.norm(point_all[finite], axis=1)
    mu_n = np.linalg.norm(mu_all[finite][:, 0, :], axis=1)
    bins = np.linspace(0, 12, 121)
    # means carried in the legend labels rather than as floating annotations, which collided
    # with the title and with each other.
    ax.hist(pt_n, bins=bins, density=True, color=S1, alpha=0.9,
            label=f'point preimage  (mean {pt_n.mean():.2f})')
    ax.hist(mu_n, bins=bins, density=True, histtype='step', lw=2.0, color=S2,
            label=f'EM posterior mean  (mean {mu_n.mean():.2f})')
    xs = np.linspace(0, 12, 400)
    ax.plot(xs, stats.chi.pdf(xs, d_a), color=INK, lw=1.6, ls='--',
            label=f'$\\chi_{{{d_a}}}$ — what $N(0,I)$ latents look like (mean 2.13)')
    _style(ax, 'C.  all 1M rows: the point arm is typical, the mixture mean is not',
           '$\\|u\\|$', 'density')
    ax.set_xlim(0, 12); ax.set_ylim(0, 0.66)
    ax.axvline(3.0, color=INK3, lw=1.0, ls=':')
    # right of the line: to its left the label brushes the chi_d curve
    ax.text(3.12, 0.578, '$u$_clip = 3', color=INK3, fontsize=8, va='top', ha='left')
    ax.grid(axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper right')
    ax.annotate(f'{100 * (mu_n > 3).mean():.0f}% of the stored index measure\n'
                f'lies past the clip Stage C samples within',
                (5.0, 0.13), textcoords='offset points', xytext=(24, 34), color=S2,
                fontsize=8, arrowprops=dict(arrowstyle='-', color=S2, lw=0.8))

    # ================= D: the sweep that picks alpha =================
    ax = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(fin_idx, min(SWEEP_ROWS, len(fin_idx)), replace=False))
    o_s, a_s = jnp.asarray(obs_all[sub]), jnp.asarray(act_all[sub])
    keys = jax.random.split(jax.random.PRNGKey(0), len(sub))
    ess_m, width = [], []
    for alpha in SWEEP_ALPHAS:
        em = jax.jit(jax.vmap(lambda s, a, k: agent.compute_full_proposal_distribution_em(
            s, a, k, num_samples=inv['num_samples'], n_steps=inv['n_steps'],
            n_initial_steps=inv['n_initial_steps'], alpha=alpha,
            n_components=inv['num_clusters'])))
        m, c, _w, e = em(o_s, a_s, keys)
        m, c, e = np.asarray(m), np.asarray(c), np.asarray(e)[:, -1]
        ok = np.isfinite(m).all(axis=(1, 2)) & np.isfinite(c).all(axis=(1, 2, 3))
        ess_m.append(e[ok].mean())
        width.append(np.trace(c[ok][:, 0], axis1=1, axis2=2).mean() / d_a)
        print(f'  alpha={alpha:>5g}  mean ESS={ess_m[-1]:6.2f}  width/prior={width[-1]:.3f}')

    ax.axhspan(ESS_GATE, max(max(ess_m) * 1.18, 26), color=S1, alpha=0.07, lw=0)
    ax.axhline(ESS_GATE, color=INK2, lw=1.3, ls='--')
    ax.text(SWEEP_ALPHAS[0] * 1.04, ESS_GATE + 0.7, 'D3 gate: mean ESS > 20',
            color=INK2, fontsize=8)
    ax.plot(SWEEP_ALPHAS, ess_m, color=S1, lw=2.0, marker='o', ms=8,
            mec=SURFACE, mew=1.5, zorder=3)
    for a_, e_, w_ in zip(SWEEP_ALPHAS, ess_m, width):
        ax.annotate(f'{w_:.2f}x', (a_, e_), textcoords='offset points', xytext=(0, 13),
                    ha='center', color=INK2, fontsize=7.5)
    ax.plot([ALPHA_HI], [ess_m[list(SWEEP_ALPHAS).index(ALPHA_HI)]], marker='o', ms=13,
            mfc='none', mec=S1, mew=2.0, zorder=4)
    _style(ax, 'D.  raising $\\alpha$ is what clears the gate',
           'inversion.alpha   (log scale)', f'mean ESS  (of {inv["num_samples"]} samples)')
    # inside the axes, not above it: at y=1.015 this sat on top of the title
    ax.text(0.015, 0.985, 'point labels: posterior width ÷ prior width   '
            '(1.00x = as wide as the prior)',
            transform=ax.transAxes, ha='left', va='top', fontsize=8, color=INK2)
    ax.set_xscale('log')
    ax.set_xticks(SWEEP_ALPHAS)
    ax.set_xticklabels([f'{a:g}' for a in SWEEP_ALPHAS])
    ax.grid(axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.set_ylim(0, max(max(ess_m) * 1.30, 28))
    ax.annotate('as run', (SWEEP_ALPHAS[0], ess_m[0]), textcoords='offset points',
                xytext=(15, -3), color=INK2, fontsize=8)
    ax.annotate('first $\\alpha$ where the posterior is\nnarrower than its prior — i.e.\n'
                'finally behaves like a posterior',
                (ALPHA_HI, ess_m[list(SWEEP_ALPHAS).index(ALPHA_HI)]),
                textcoords='offset points', xytext=(-8, -62), color=INK, fontsize=8,
                ha='center', arrowprops=dict(arrowstyle='-', color=INK, lw=0.8))

    fig.text(0.068, 0.968, 'The noise preimage of a behaviour action, and why its '
             'temperature decides whether it is usable',
             color=INK, fontsize=13.5, fontweight='bold', va='top')
    fig.text(0.068, 0.925,
             f'Flow $G_\\theta(s,u)$ maps a latent $u$ to an action. The preimage asks the '
             f'inverse — which $u$ produced the dataset action $a$?\n'
             f'Because $\\|\\partial a/\\partial u\\| \\approx {sv.mean():.2f}$, many $u$ give '
             f'nearly the same $a$: the answer is a region, and the temperature $\\alpha$ '
             f'sets how tightly it is resolved.',
             color=INK2, fontsize=9.2, va='top', linespacing=1.5)
    fig.text(0.068, 0.028,
             f'cube-single-play · {len(act_all):,} transitions · Stage-A flow @ '
             f'{meta["restore_epoch"]} · inversion n_steps={inv["n_initial_steps"]}, '
             f'num_samples={inv["num_samples"]}, num_clusters={inv["num_clusters"]} · '
             f'A/B: row {row} (median ESS), slice through the exact preimage · '
             f'D: {len(sub)} rows',
             color=INK3, fontsize=7.5, va='top')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
