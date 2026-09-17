# Goal-indexed measure with per-triple multipliers (collation before implementation)

Date: 2026-09-16. Status: design collated; nothing implemented.

## 1. Symbols

| symbol | type | meaning | current instantiation |
|---|---|---|---|
| `u` | U = R^5 | action latent; action = G(s, u) through the frozen flow | dataset rows carry the preimage of the recorded action |
| `u'` | U | policy index; pi_{u'}(s) = G(s, u') | prior draw N(0, I), clipped |
| `w: U -> D` | D = R^128 | coefficient of the policy indexed by u' | w(u') = enc(u') / ||enc(u')|| |
| `phi: S x U x S+ -> D` | | affine basis over the triple | phi(s,u,s+) = A(s,u)^T f(s+), A in R^{128x128} |
| `b: S x U x S+ -> R` | | affine bias over the triple | b(s,u,s+) = beta(s,u)^T f(s+), beta in D |
| `f: S+ -> D` | | state basis | ||f|| = sqrt(128); trained with the orthonormality term |
| `M^{pi_{u'}}(s,u,s+)` | | successor measure | phi(s,u,s+)^T w(u') + b(s,u,s+) |
| `r(s+)` | | reward at the future state | shifted indicator, r+1 in {0,1} |
| `l(s,u,s+)` | >= 0 | Lagrange multiplier of the non-negativity constraint at the triple | see section 3 |
| `g`, `z_pi` | S+ | a goal state used as a policy index | proposed |
| `h: S+ -> D` | | goal-to-coefficient map | proposed; w*(g) = h(g) / ||h(g)|| |

Value of a policy with coefficient w_pi:

    Q^pi(s,u) = sum_{s+} ( phi(s,u,s+)^T w_pi + b(s,u,s+) ) r(s+)
              = ( A(s,u) w_pi + beta(s,u) )^T rho,      rho = sum_{s+} f(s+) r(s+)

## 2. The constrained program

    max_w   E_{(s,u)~D} sum_{s+} ( phi(s,u,s+)^T w + b(s,u,s+) ) r(s+)
    s.t.    phi(s,u,s+)^T w + b(s,u,s+) >= 0     for all (s, u, s+)

## 3. Lagrange multipliers: the correction

One inequality per triple needs one multiplier per triple:

    max_{l >= 0} min_w   - E_{(s,u)~D} sum_{s+} ( phi^T w + b ) r(s+)
                         - sum_{(s,u,s+)} l(s,u,s+) * min( phi(s,u,s+)^T w + b(s,u,s+), 0 )

PSM Eq. 10 writes lambda(s,a) multiplying sum_{s+} min(.,0). That is a penalty on the row's
total violation, not the Lagrangian of the per-triple constraints: it cannot weight the
particular s+ that violates, so the dual is coarser and need not be tight. Our previous run
used the row form (one l_i per dataset row, applied to the row's mean violation) and the
penalty never bound. In the per-triple form a persistently violated triple accumulates its
own multiplier until its term outweighs the objective gradient. With N^2 = 10^8 triples the
multipliers are amortised: l_psi(s,u,s+) >= 0, a network trained by ascent on the violation.

## 4. Previous optimisation (what ran, 2026-09-15)

B1. Head fit (A, beta, f, enc). Contrastive TD on dataset triples, index slot w(u'_i) with
u'_i ~ N(0,I) per row, bootstrap action at s'_i = u'_i (default) or the EMaQ argmax
(bootstrap=gpi_argmax, 2026-09-16). Ensemble of 2, target = min.

B2. Eq. 10 inference on the frozen head, one free omega per task:

    J(omega) = (1/N) sum_i ( A_i omega + beta_i )^T rho = a_bar^T omega + b_bar        (linear)
    m_ij(omega) = ( A_i omega + beta_i )^T f(s+_j) / S
    v_i = mean_j max(-m_ij, 0);   L(omega, l) = -J/S + mean_i l_i v_i                  (row multipliers)
    omega_{t+1} = omega_t - Adam(grad_omega L);  [sphere: omega <- omega/||omega||]
    l_i <- max( l_i + eta_l v_i, 0 );   T = 5000, R = 1024 rows x C = 256 columns per step
    omega_0 = argmax_{k<=4096} a_bar^T w(u'_k)

Outcome: grad J = a_bar constant, penalty inert (20% of triples violated already at omega_0,
a trained family member), omega* = omega_0 + eta T sign(a_bar) (||omega*|| 56.8) or
a_bar/||a_bar|| on the sphere; cos to the nearest w(u') 0.28; success 0.000-0.012 five-task.

B3. Acting: u*(s) = argmax_{m<=64} [mean_P - 0.5 spread] Q_omega(s, u_m), u_m ~ N(0,I); one
fixed omega per task, which is the fixed_index mode measured at 0.000 / 0.180.

## 5. New optimisation (proposed)

Policy index = a goal state g. Coefficient w*(g) = h_theta(g) / ||h_theta(g)||, on the sphere.
The family anchor is w(z) = f(z) / ||f(z)||, i.e. h initialised at (or regularised toward) f.
Reward for goal g is the indicator r_g(s+) = 1[s+ = g], so

    Q_g(s,u) = phi(s,u,g)^T w*(g) + b(s,u,g)                       (the sum over s+ collapses)

Samples: (s_i, u_i, s'_i, g_i) from the replay buffer with hindsight goals g_i on the same
trajectory, plus negatives s+_j (other rows' next states), plus goals g_k for the constraint.

Per step, three coupled updates:

(a) Policy evaluation -- head fit with the goal in the index slot:

    c_i = w*_theta(g_i)                                              (stop-grad into the head fit)
    u+_i = argmax_{m<=8} [mean_P - 0.5 spread] ( A(s'_i,u_m) c_i + beta(s'_i,u_m) )^T f(g_i)
                                                                      (EMaQ bootstrap on pi_{g_i}, target head)
    M_ij  = ( A(s_i,u_i) c_i + beta(s_i,u_i) )^T f(s+_j)
    Mbar_ij = ( Abar(s'_i,u+_i) c_i + betabar(s'_i,u+_i) )^T fbar(s+_j),  min over ensemble
    L_head = contrastive( M, gamma Mbar ) + 1000 ||E[f f^T] - I||^2       -> descent on A, beta, f

(b) Policy improvement -- the coefficient:

    J(theta | g) = (1/N) sum_i ( phi(s_i,u_i,g_i)^T w*_theta(g_i) + b(s_i,u_i,g_i) )   (b is theta-free)
    -> ascent on theta with the head stop-gradded

(c) Feasibility -- per-triple multipliers:

    v(s,u,s+) = max( -( phi(s,u,s+)^T w*_theta(g) + b(s,u,s+) ), 0 )   on sampled triples
    L_con = sum l_psi(s,u,s+) v(s,u,s+)
    -> descent on theta, ascent on psi (projected l >= 0)

Combined: theta minimises -J + L_con; psi maximises L_con; the head minimises L_head.

Test time: the task reward defines a goal set G (the rewarding s+ in the relabel batch).
Linearity gives the task coefficient and the readout

    w_task = normalise( sum_{g in G} w*(g) ),     Q(s,u) = ( A(s,u) w_task + beta(s,u) )^T rho_G,
    rho_G = sum_{g in G} f(g)

Acting: u*(s) = argmax_{m<=64} [pessimistic] Q(s, u_m), as today.

Two facts fixed by the algebra:

- On a frozen head, (b) with the sphere is closed-form: w*(g) = Abar^T f(g) / ||Abar^T f(g)||,
  Abar = (1/N) sum_i A_i. That is the sphere Eq. 10 result per goal, already measured at ~0.
  The proposal is only new because (a) fits the head on the coefficients w*(g) that (b) and
  test time query, and bootstraps on pi_g. (b) alone re-derives the failed result.
- Feasibility becomes a training-time property of the head, not a question asked of a frozen
  head afterwards. Whether the constraint set is non-empty is checked during training
  (violation fraction under the current w*(g)), not discovered by a diverging dual.

## 6. Previous vs new

| | previous | new |
|---|---|---|
| policy index | noise u' | goal state g |
| coefficient | enc(u') on the sphere; free omega at inference | w*(g) = h_theta(g)/||h_theta(g)|| on the sphere |
| reward inside the objective | rho = E[(r+1) f(s+)] | indicator at g: the s+ sum collapses to the triple (s,u,g) |
| head fit sees the query coefficients | no (fit on w(u'), queried at omega) | yes (fit on w*(g), queried at w*(g)) |
| bootstrap | same u' (default) / EMaQ under rho | EMaQ under w*(g), maximising the measure at g |
| multipliers | l_i per row, row-mean violation | l_psi(s,u,s+) per triple, amortised |
| goals | random other rows' next states (mix_ratio half of task_w) | hindsight goals on the trajectory, many per row |
| test-time coefficient | infer_z(rho) | normalise( sum_{g in G} w*(g) ) |
| deployment | fixed omega per task -> fixed_index floor | per-state argmax over u under w_task, as GPI today |

## 7. What has to be built

1. Hindsight goal sampling in the dataset/replay path (future state on the same trajectory).
2. h_theta(g) as the index-slot coefficient (replaces enc(task_w) under policy_index=task_vector).
3. The EMaQ bootstrap under the goal coefficient with the goal's own f(g) as the readout
   (today it reads rho).
4. The J ascent step on theta with the head stop-gradded.
5. l_psi(s,u,s+) and the constraint step (ascent on psi, descent on theta) on sampled triples.
6. Test-time goal-set coefficient w_task and rho_G.
7. Optional: the triple network phi_theta(s,u,s+) instead of A^T f (B^2 cost; negatives must be subsampled).

Kept from today: the split phi = A^T f (first version), sphere normalisation, ensemble-min
pessimism, the EMaQ argmax (N = 8), 500-episode five-task evals.
