#!/usr/bin/env bash
# 500-episode evaluation of one saved checkpoint (WP0 of the ICLR figure plan).
#
# Every number destined for the paper comes from here or from a multi-seed aggregate of
# these JSONs -- never from the 50-episode in-loop evals, whose 95% CI is about +/-0.115.
#
# Usage:
#   GPU=0 bash scripts/eval500.sh psmflow|psmgoal cube <run_dir> <out_name> [extra hydra args]
#   GPU=0 bash scripts/eval500.sh bc      cube -         <out_name>
#
# No arm flags are needed on the CLI: tools/eval_checkpoint.py takes the agent config from
# the RUN'S OWN flags.json and only lets typed CLI overrides sit on top, so a checkpoint
# trained with any combination of psi_form / policy_index / train_actor / acting restores
# under whatever this checkout's defaults happen to be. (A flags.json that predates a key
# falls back to that key's pre-2026-09-04 value, not to today's default -- see
# LEGACY_AGENT_DEFAULTS in tools/eval_checkpoint.py.) Pass a flag here only to evaluate a
# checkpoint DELIBERATELY off its own config.
#
# The second argument selects the env/flow/preimage triple; `bc` evaluates the frozen
# Stage-A flow with per-step prior latents (agent=fql agent.bc_only=true), which is the
# control every Stage-C number must be quoted beside.
set -euo pipefail

# Everything below lives in a FUNCTION, and that is load-bearing, not style.
#
# bash reads a script INCREMENTALLY: it parses one top-level command, runs it, then seeks
# back to where it left off and parses the next. A 500-episode eval sits inside one such
# command for up to 90 minutes, and this file lives in a shared checkout that other people
# edit. Rewrite it in place during that window and the running job resumes at a stale BYTE
# OFFSET into the new text -- mid-line, mid-quote -- and dies at the tail with a garbage
# error, after its python call has already succeeded and written the JSON. That is exactly
# what happened on 2026-09-07:
#
#   jobs 2491881-2491889  line 100: <preimage npz>: Permission denied   (exit 126)
#   job  2491924          line 134: unexpected EOF while looking for matching '"'  (exit 2)
#
# Neither was ever a real syntax error -- no committed or working-tree version of this file
# has ever failed `bash -n`, and 2491924's 56-second window contains a write to this file
# at 01:24:42. A function body is parsed to its closing brace in ONE pass before anything
# runs, so bash never re-reads the file and a concurrent edit cannot reach a live job.
#
# Two rules follow: keep the executable part inside main(), and edit this file by writing a
# temp file and `mv`-ing it into place (a new inode) rather than truncating it, so a job
# that is already running keeps reading the version it started with.
main() {

  # Checkpoint epoch to restore. Defaults to 500000, the only epoch every recorded eval500
  # number has ever used; set it to compare arms at a common EARLIER checkpoint (a run that
  # has not finished yet). Whatever it is, it is recorded in the report JSON's restore_epoch.
  RESTORE_EPOCH="${RESTORE_EPOCH:-500000}"

  MODE="${1:?psmflow|psmgoal|bc}"
  ENVKEY="${2:?cube|antmaze|pointmaze|scene}"
  RUN_DIR="${3:?run dir, or - for bc}"
  OUT="${4:?output json basename}"
  shift 4
  EXTRA=("$@")

  case "$MODE" in psmflow|psmgoal|bc) ;; *) echo "unknown mode: $MODE" >&2; exit 1 ;; esac

  # Paths default to midi-01, where every earlier eval500 JSON was produced. On another
  # machine (KISSKI/SLURM) override them in the environment rather than editing this file:
  #   PSM_REPO   checkout root            EVAL_LOGS  where the report JSON is written
  #   EXP_ROOT   experiment root          FLOW_DIR   frozen Stage-A ckpt dir (overrides ENVKEY)
  #   PRE_NPZ    preimage npz (overrides ENVKEY)
  #   ENV_NAME   full OGBench env id (overrides ENVKEY's default)
  #
  # ENV_NAME exists for the multi-task protocol. A bare `...-singletask-v0` id evaluates ONE
  # OGBench task (cube's default is task 2, the mazes' is task 1), so every number this
  # script produced before 2026-09-06 is that single task. `...-singletask-task{1..5}-v0`
  # selects the others: same gym env, same dataset file, only the RELABELED REWARDS differ
  # (ogbench/utils.py relabel_dataset), and since no training loss reads rewards -- psmflow
  # touches them only in infer_z at eval -- an existing checkpoint evaluates zero-shot on all
  # five by changing this variable alone. The flow and preimage paths are unchanged.
  REPO="${PSM_REPO:-/u/amsks/git/PSMFLows}"
  PY="${PY:-$REPO/.venv/bin/python}"
  LOGS="${EVAL_LOGS:-/data-local/amsks/PSMFLows/logs}"
  EXP="${EXP_ROOT:-/var/local/amsks/exp/PSMFLows}"

  case "$ENVKEY" in
    cube)
      ENV_NAME="${ENV_NAME:-cube-single-play-singletask-v0}"
      FLOW="${FLOW_DIR:-$EXP/bcflow_cube_single_20260726_135032/sd000_20260726_135037}"
      PRE="${PRE_NPZ:-/data-local/amsks/PSMFLows/preimages_cube_single_a20_n200.npz}" ;;
    antmaze)
      ENV_NAME="${ENV_NAME:-antmaze-medium-navigate-singletask-v0}"
      FLOW="${FLOW_DIR:-$EXP/bcflow_antmaze-medium-navigate_20260805_014546/sd000_20260805_014548}"
      PRE="${PRE_NPZ:-/data-local/amsks/PSMFLows/preimages_antmaze_medium_a20_n200.npz}" ;;
    pointmaze)
      ENV_NAME="${ENV_NAME:-pointmaze-medium-navigate-singletask-task1-v0}"
      FLOW="${FLOW_DIR:-$EXP/bcflow_pointmaze-medium-navigate_20260729_142219/sd000_20260729_142225}"
      PRE="${PRE_NPZ:-/data-local/amsks/PSMFLows/preimages_pointmaze_medium_a20_n200.npz}" ;;
    scene)
      ENV_NAME="${ENV_NAME:-scene-play-singletask-v0}"
      FLOW="${FLOW_DIR:-$EXP/flow/scene-play}"
      PRE="${PRE_NPZ:-$EXP/preimages/scene-play.npz}" ;;
    *) echo "unknown env key: $ENVKEY" >&2; exit 1 ;;
  esac

  [ -d "$FLOW" ] || { echo "no such flow dir: $FLOW" >&2; exit 1; }
  if [ "$MODE" != "bc" ]; then
    [ -f "$PRE" ] || { echo "no such preimage npz: $PRE" >&2; exit 1; }
  fi

  # How many worker PROCESSES the 500 episodes are split over on the one GPU. The rollout
  # is MuJoCo on a single CPU core with a batch-1 flow decode per step -- measured
  # 2026-09-07 at 42% of a core and 3.2 GB of an 80 GB H100 -- so the device is nowhere
  # near saturated on memory and several processes fit.
  #
  # They do NOT scale linearly. Without MPS the driver time-slices between CUDA contexts,
  # and a rollout keeps a kernel resident ~73% of the time, so the aggregate saturates
  # fast: measured 1.43x at N=8 on antmaze (92.3 -> 64.5 min) and 1.50x for three
  # concurrent cube rollouts. N=4 gets essentially all of it; N=8 only costs more cores.
  #
  # 1 is the historical single-process path and reproduces every pre-2026-09-07 number bit
  # for bit. N>1 draws a DIFFERENT sample of 500 episode inits (worker w is seeded
  # `seed*N + w`; see tools/eval_checkpoint.py's docstring), so it agrees with an old
  # number within the Wilson interval, not exactly. Validated 2026-09-07 on affine-strict
  # antmaze sd1 @250k: N=8 gave 93/500 = 0.186 [0.154, 0.223] against the N=1 run's
  # 89/500 = 0.178 [0.147, 0.214]. Keep N=1 when a number has to be byte-reproduced.
  #
  # The DEFAULT here is 1, deliberately. A SLURM job that was already queued when this file
  # changed runs the sbatch body it was submitted with but reads THIS file fresh off disk at
  # launch, so a default of 8 would silently re-shape jobs someone else queued -- and give
  # them 8 workers on the 8 cores their old header asked for. `scripts/slurm/eval500.sbatch`
  # exports EVAL_WORKERS=4 (the measured knee), so every NEW submission gets the
  # parallel path and nothing already in the queue changes behaviour.
  #
  # Make sure the job has the cores: --cpus-per-task must be >= EVAL_WORKERS.
  export EVAL_WORKERS="${EVAL_WORKERS:-1}"

  cd "$REPO"
  export CUDA_VISIBLE_DEVICES="${GPU:-0}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  # The WHOLE-JOB fraction; eval_checkpoint.py divides it by EVAL_WORKERS before spawning.
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.30}"
  export MUJOCO_GL=egl
  export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-/var/local/amsks/ogbench}"
  mkdir -p "$LOGS"

  if [ "$MODE" = "bc" ]; then
    # The frozen flow acting alone: sample_actions draws a fresh N(0, I) latent per step
    # and decodes it. No z inference, no preimages.
    "$PY" tools/eval_checkpoint.py agent=fql agent.bc_only=true \
      env_name="$ENV_NAME" restore_path="$FLOW" restore_epoch="$RESTORE_EPOCH" \
      eval_episodes=500 report_out="$LOGS/$OUT.json" "${EXTRA[@]}"
  else
    "$PY" tools/eval_checkpoint.py agent="$MODE" \
      env_name="$ENV_NAME" \
      agent.flow_ckpt_path="$FLOW" agent.flow_ckpt_epoch=500000 \
      agent.preimage_path="$PRE" agent.use_point_preimage=true \
      restore_path="$RUN_DIR" restore_epoch="$RESTORE_EPOCH" \
      eval_episodes=500 report_out="$LOGS/$OUT.json" "${EXTRA[@]}"
  fi
}

main "$@"
