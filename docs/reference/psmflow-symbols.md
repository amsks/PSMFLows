# LatentFlowPSM reference: symbols, defaults, stabilisers

Companion to `agents/psmflow.py` — read it before touching the losses. The write-up is
`git show 5249267:PAPER/main.tex` (LatentFlowPSM section) and `PAPER/ICLR/` for the current
draft; PSM is arXiv 2411.19418. `docs/COMPENDIUM.md` §2 transcribes the same symbols.

Actions are indexed by the latent `u` of a frozen conditional flow `G(s, u)`, so every
action the measure evaluates, bootstraps or executes is a flow decode and the bootstrap
distribution equals the data distribution.

## Symbol map

| symbol | meaning | code |
|---|---|---|
| `G(s, u)` | frozen behaviour flow | params `flow_vf` / `flow_onestep`, applied by `decode` |
| `u` | latent action emitted now | the inverted dataset latent is `u_data` |
| `u'` | policy index: `pi_{u'} = G(., u')` is the continuation policy | drawn `u_index` per batch element; `_index` returns whatever fills psi's index slot under the current `policy_index` |
| `u^+` | bootstrap continuation latent at `s'` | `u_next`; equal to `u'` under `policy_index='latent'`, the actor's latent under `'task_vector'` |
| `p0` | latent prior `N(0, I)` | the write-up confines the search to the chi-square typical set `U_delta`; the code uses the L-infinity box `|u| <= u_clip = 3`, the same device at a coarser shape. `index_clip` (None = `u_clip`) narrows that box for the `u'` draws alone; `_index_clip` resolves it |
| `phi(x)` | basis over future states, `R^z_dim` (the write-up's varphi, PSM's proto basis) | `PhiMap`, field `phi`, target `target_phi` |
| `psi(s, u, u')` | successor features `int phi dM^{u->u'}`, `R^z_dim`, ensembled `P` times | `PsiMap` (free) or `AffinePsiMap` (affine); field `psi` |
| `A(s, u)` | affine basis factor, `R^{z_dim x d_w}` | head `a_out` |
| `beta(s, u)` | affine bias factor, `R^z_dim` | head `beta_out`. Assumption `affine` is that neither `A` nor `beta` sees the policy index |
| `w(u')` | policy coordinate of `pi_{u'}`, `R^d_w` | `AffinePsiMap.encode_index`, submodule `w_enc`; `w_index` in the losses |
| `w` | task vector, `R^z_dim`, `w = E_D[r(x) phi(x)]` (Cor. `reward-inference`) | `task_w` in a batch, `task_z` at eval; computed by `infer_z` |
| `m(s,u,u',x)` | successor-measure density | fitted as `psi(s,u,u')^T phi(x)` |
| `Q_w(s, u)` | `psi(s, u, u')^T w` | ensemble mean minus `kappa` times its spread |
| `Lambda_K` | the `K` prior draws of `u'` that GPI maximises over | `gpi_num_u` at eval, `index_panel` inside the actor and the expectile head |
| `P`, `kappa` | critic ensemble size and pessimism coefficient | `num_parallel`, `pessimism_penalty` / `actor_pessimism_penalty` |
| `pi_eta` | amortized latent actor `(s, w, eps) -> u` | field `actor`; `v_xi` is its conditional-flow-matching anchor over dataset latents, field `actor_vf` |

The action branch (`action_critic.enabled`, off by default) is not in the write-up. It
carries `psi_a(s, w, a)`, successor features over EXECUTED actions (field `psi_a`); `B_a`,
its own backward map under the FB graft (field `phi_a`, task vector `task_z_a`); and
`delta(s, w, u)`, a residual bounded by `residual_eps` (field `residual`).

## Defaults — the affine form of Prop. `bilinear`

    psi_form=affine       psi(s,u,u') = A(s,u)^T w(u') + beta(s,u)
    policy_index=latent   psi's index slot carries u' ~ p0; w enters only through the
                          readout Q = psi^T w
    train_actor=false     no amortized actor
    acting=gpi            per-step argmax over (u_i, u'_j) pairs (Alg. Rung 1)

Under them the bootstrap latent is a `p0` draw independent of `s'`, so Prop. `insample`'s
`C = 1` applies and `backup_explore_frac` is inert. The remaining arms are ablations:
`psi_form=free` absorbs `w(u')` into a free psi (Rem. `tradeoff`), and
`policy_index=task_vector acting=actor train_actor=true` is the latent-actor agent.

## Stabilisers of the measure loss

Both are OFF by default (`ortho_mode='fixed'`, `psi_bound='none'` reproduce the published
loss expression). They exist because `docs/design/2026-09-07-antmaze-g995-collapse.md`
measured, on three antmaze seeds at `discount=0.995`, the contrastive term `psm_loss`
growing one decade per 31k steps until it outgrew `ortho_coef * |orth_loss|` at 115-125k,
after which phi collapsed toward rank one and 500-episode success fell 0.52 -> 0.01.

- `ortho_mode='relative'` — the geometry regulariser's weight tracks the term it has to
  hold, `ortho_coef + ortho_rel_coef * stopgrad(|psm_loss|)`, so the ratio of the two
  gradient directions is scale-free and ortho can never be outgrown. Touches phi's
  gradient only.
- `psi_bound='tanh'` — bounded reparameterisation of the measure head,
  `psi = S * tanh((A^T w + beta) / S)`. phi is on the sphere and `w(u')` is unit-norm, so
  `A` and `beta` carry the only free magnitude in the model; `S` is the per-component
  ceiling a successor feature `psi_i = E[sum_t gamma^t phi_i(x_t)]` can have under
  `E[phi phi^T] = I`, i.e. `1/(1-gamma)`.
- `psi_bound='clip_target'` — the same admissible set imposed on the TD bootstrap instead:
  `target_M` clipped to `+-(S * z_dim)`, the Cauchy-Schwarz image of the psi ball. Leaves
  the forward head untouched.
