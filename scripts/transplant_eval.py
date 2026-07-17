#!/usr/bin/env python3
"""Transplant-eval: load reference torch weights (dumped to .npz) into our JAX PSMAgent
and eval on task2. Localizes the train/eval gap:
  ref weights score HIGH in our eval -> bug is in our TRAINING/init (not eval)
  ref weights score LOW               -> bug is in our EVAL (both audits say unlikely)
Guards against a buggy converter with phi-norm / actor-range sanity checks first.

NOTE (2026-07-17): this script targets the PRE-refactor PSMAgent API and must be updated
before use. The agent no longer has a flat `params` dict / `_apply` helper; params now
live in per-network TrainStates. Port map:
  agent.params["phi"]        -> agent.phi.params        (likewise sf_psi/proto_psi/actor/actor_vf)
  agent.params["psm_psi"]    -> agent.proto_psi.params  ("psm_psi" was renamed to "proto_psi")
  agent.params["target_*"]   -> agent.target_phi / target_proto_psi / target_sf_psi
  agent._apply("phi", p, x)  -> agent.phi(x, params=p)
  agent.replace(params=...)  -> agent.replace(phi=agent.phi.replace(params=...), ...)
  agent.z_eval               -> agent.task_z
"""
import os, json, glob
os.environ.setdefault("OGBENCH_DATASET_DIR", "/var/local/amsks/ogbench")
import numpy as np
import jax, jax.numpy as jnp
import ogbench

from agents.psm import PSMAgent
from utils.torch_to_flax import (load_phi_params, load_psi_params, dense_params, layernorm_params)
from utils.evaluation import evaluate

NPZ = os.environ.get("TP_NPZ", "/var/local/amsks/exp/ref_ckpt_s5_step100k.npz")
FLAGS = sorted(glob.glob("/var/local/amsks/exp/PSMFLows/psm_protoxplant_*/sd005_*/flags.json"))[-1]


def load_flow_actor(fix, prefix="_actor"):
    """NoiseConditionedActor (flax compact order): z_embedding first (Dense_0,LN_0,Dense_1),
    then s_embedding (Dense_2,LN_1,Dense_3), then policy (Dense_4,Dense_5), out (Dense_6).
    torch: embed_z.{0,1,3}, embed_s.{0,1,3}, policy.{0,2,4}."""
    g = lambda k: fix[f"w__{prefix}.{k}"]
    return {
        "Dense_0": dense_params(g("embed_z.0.weight"), g("embed_z.0.bias")),
        "LayerNorm_0": layernorm_params(g("embed_z.1.weight"), g("embed_z.1.bias")),
        "Dense_1": dense_params(g("embed_z.3.weight"), g("embed_z.3.bias")),
        "Dense_2": dense_params(g("embed_s.0.weight"), g("embed_s.0.bias")),
        "LayerNorm_1": layernorm_params(g("embed_s.1.weight"), g("embed_s.1.bias")),
        "Dense_3": dense_params(g("embed_s.3.weight"), g("embed_s.3.bias")),
        "Dense_4": dense_params(g("policy.0.weight"), g("policy.0.bias")),
        "Dense_5": dense_params(g("policy.2.weight"), g("policy.2.bias")),
        "Dense_6": dense_params(g("policy.4.weight"), g("policy.4.bias")),
    }


def load_flow_vf(fix, prefix="_actor_vf"):
    """FlowVectorField: 5 Dense (Dense_0..4) <- torch net.{0,2,4,6,8}."""
    g = lambda k: fix[f"w__{prefix}.net.{k}"]
    return {f"Dense_{i}": dense_params(g(f"{2*i}.weight"), g(f"{2*i}.bias")) for i in range(5)}


def tree_keys(d, pre=""):
    out = set()
    for k, v in d.items():
        if isinstance(v, dict):
            out |= tree_keys(v, pre + k + "/")
        else:
            out.add(pre + k)
    return out


def main():
    jax.config.update("jax_enable_x64", False)
    fix = {k: np.asarray(v) for k, v in np.load(NPZ).items()}
    cfg = json.load(open(FLAGS))
    acfg = cfg["agent"]
    print(f"config: z_dim={acfg['z_dim']} ortho={acfg['ortho_coef']} actor={acfg['actor']['type']}")

    ex_obs = jnp.zeros((1, 28), jnp.float32)
    ex_act = jnp.zeros((1, 5), jnp.float32)
    agent = PSMAgent.create(0, ex_obs, ex_act, acfg)

    loaded = {
        "phi": load_phi_params(fix), "target_phi": load_phi_params(fix, "target_phi"),
        "sf_psi": load_psi_params(fix, "sf_psi"),
        "psm_psi": load_psi_params(fix, "psm_psi"),
        "actor": load_flow_actor(fix), "actor_vf": load_flow_vf(fix),
    }
    # targets for psi (reuse loader against target_* keys via a remap)
    def psi_target(which):
        remap = {k.replace(f"target_{which}", which): v for k, v in fix.items() if k.startswith(f"w__target_{which}.")}
        return load_psi_params(remap, which)
    loaded["target_sf_psi"] = psi_target("sf_psi")
    loaded["target_psm_psi"] = psi_target("psm_psi")

    # verify every loaded subtree matches the agent's structure exactly
    for key in ["phi", "sf_psi", "psm_psi", "actor", "actor_vf", "target_phi", "target_sf_psi", "target_psm_psi"]:
        want = tree_keys(agent.params[key]); got = tree_keys(loaded[key])
        assert want == got, f"[{key}] tree mismatch\n want={sorted(want)}\n got ={sorted(got)}"
        for leaf_path in want:  # shape check
            wv = agent.params[key]; gv = loaded[key]
            for p in leaf_path.split("/"):
                wv, gv = wv[p], gv[p]
            assert tuple(wv.shape) == tuple(np.asarray(gv).shape), f"[{key}/{leaf_path}] {wv.shape} vs {np.asarray(gv).shape}"
    print("tree + shape check: PASS")

    agent = agent.replace(params={k: jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), v)
                                  for k, v in loaded.items()})

    # ---- converter sanity checks ----
    env, ds, _ = ogbench.make_env_and_datasets("cube-single-play-singletask-v0", compact_dataset=False)
    obs_s = jnp.asarray(ds["observations"][:256], jnp.float32)
    phi_out = agent._apply("phi", agent.params["phi"], obs_s)
    print(f"phi output L2 norm mean={float(jnp.linalg.norm(phi_out,axis=-1).mean()):.3f} (expect ~sqrt(128)=11.314)")

    # infer eval z (task2), then check actor outputs are in range
    n = min(ds["observations"].shape[0], 10000)
    zb_idx = np.random.randint(0, ds["observations"].shape[0], n)
    rew = jnp.asarray(ds["rewards"][zb_idx] + 1.0, jnp.float32)
    nobs = jnp.asarray(ds["next_observations"][zb_idx], jnp.float32)
    eval_agent = agent.infer_eval_z(nobs, rew)
    a = eval_agent.sample_actions(obs_s[:64], seed=jax.random.PRNGKey(0))
    print(f"actor action range [{float(a.min()):.3f},{float(a.max()):.3f}] mean|a|={float(jnp.abs(a).mean()):.3f} (expect in [-1,1], ~0.3)")

    # ---- task2 eval ----
    class C(dict):
        __getattr__ = dict.get
    evcfg = C(cfg)
    info, _, _ = evaluate(agent=eval_agent, env=env, config=evcfg,
                          num_eval_episodes=50, num_video_episodes=0, video_frame_skip=3)
    print(f"\n=== TRANSPLANT-EVAL: reference seed5 step100k weights, OUR eval, task2 ===")
    print(f"success = {info.get('success'):.3f}   (reference's own eval @100k ~0.20; @50k=0.08)")


if __name__ == "__main__":
    main()
