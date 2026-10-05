# Joint-actor arms (2026-10-05): group -> hydra overrides on top of configs/agent/psmgoal.yaml.
# Base = the data-bootstrap arm psmgoal_db_sm_cube with the actor back in the bootstrap.
JA_BASE="agent.policy_index=goal agent.bootstrap_source=actor agent.train_actor=true agent.actor_input=goal agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric agent.measure_loss=softmax agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel"
declare -A JA_ARMS=(
  [psmgoal_ja_fbc3_sh_cube]="$JA_BASE agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.actor_value_kind=shaped"
  [psmgoal_ja_fbc0_sh_cube]="$JA_BASE agent.actor_kind=flowbc agent.fb_bc_coeff=0.0 agent.actor_value_kind=shaped"
  [psmgoal_ja_dsrl_sh_cube]="$JA_BASE agent.actor_kind=dsrl agent.fb_bc_coeff=0.0 agent.actor_value_kind=shaped"
  [psmgoal_ja_fbc3_pt_cube]="$JA_BASE agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.actor_value_kind=point"
)
