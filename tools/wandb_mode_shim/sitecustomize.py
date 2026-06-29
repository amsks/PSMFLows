"""Auto-imported (via PYTHONPATH) shim for the vendored OGBench child process.

Two independent, opt-in behaviors (each a no-op unless its env var is set), so
this file is inert for any unrelated Python process and never edits
third_party/ (which must stay byte-for-byte upstream per its PROVENANCE rules):

1. GCIQL_WANDB_FORCE_MODE = offline|disabled
   OGBench's setup_wandb hardcodes wandb.init(mode='online'); force the mode
   (wandb >= 0.27 lets the explicit kwarg override WANDB_MODE/WANDB_DISABLED).

2. GCIQL_FB_ENV = <ogbench env_name, e.g. cube-single-play-v0>
   Mirror GCIQL's eval metrics into FB's wandb key scheme, ADDING keys
   alongside OGBench's originals so FB and GCIQL overlay on the same panels:
     evaluation/overall_success           -> eval/reward/eval/success
     evaluation/task<N>_<desc>_success    -> eval/reward/<fb_task>/success
   where <fb_task> = "<env_base>-singletask-task<N>-v0" and env_base is
   GCIQL_FB_ENV with any trailing "-v0" removed.
"""
import functools
import os
import re

_forced = os.environ.get("GCIQL_WANDB_FORCE_MODE")
_fb_env = os.environ.get("GCIQL_FB_ENV")
# OGBench hardcodes wandb.init(project='OGBench', entity=None); an explicit
# project= kwarg overrides WANDB_PROJECT, so override the init kwargs here to
# land GCIQL runs in FB's project/entity for direct comparison.
_project = os.environ.get("GCIQL_WANDB_PROJECT")
_entity = os.environ.get("GCIQL_WANDB_ENTITY")

if _forced in ("offline", "disabled") or _fb_env or _project or _entity:
    try:
        import wandb as _wandb

        if _forced in ("offline", "disabled") or _project or _entity:
            _orig_init = _wandb.init

            @functools.wraps(_orig_init)
            def _init(*args, **kwargs):  # override caller kwargs as configured
                if _forced in ("offline", "disabled"):
                    kwargs["mode"] = _forced
                if _project:
                    kwargs["project"] = _project
                if _entity:
                    kwargs["entity"] = _entity
                return _orig_init(*args, **kwargs)

            _wandb.init = _init

        if _fb_env:
            _env_base = _fb_env[:-3] if _fb_env.endswith("-v0") else _fb_env

            # OGBench logs its tiled eval clip as
            # wandb.Video(ndarray, fps=15, format='mp4') where the array is
            # (t, c, H, W) uint8 (get_wandb_video -> reshape_video in
            # third_party/ogbench/impls/utils/log_utils.py). Re-encode it as a
            # GIF with FB's exact imageio params so GCIQL eval media matches
            # FB's recording (evals/ogbench.py).
            #
            # Why patch __init__ instead of rebinding wandb.Video to a wrapper
            # function: OGBench's CsvLogger uses
            #   isinstance(v, (wandb.Image, wandb.Video, wandb.Histogram))
            # to filter media out of the CSV header (log_utils.py:26). If
            # wandb.Video is replaced with a function, that tuple contains a
            # non-type and isinstance raises TypeError on the first eval,
            # crashing every GCIQL pixel run. Mutating __init__ keeps the
            # class identity intact.
            import numpy as _np
            import tempfile as _tempfile
            import imageio as _imageio

            _orig_video_init = _wandb.Video.__init__

            def _video_init(
                self, data_or_path, caption=None, fps=None, format=None
            ):
                try:
                    if isinstance(data_or_path, _np.ndarray) and format == "mp4":
                        _frames = _np.transpose(data_or_path, (0, 2, 3, 1))
                        _tmp = _tempfile.NamedTemporaryFile(
                            suffix=".gif", delete=False
                        )
                        _tmp.close()
                        _imageio.mimsave(
                            _tmp.name, _frames, format="GIF", fps=30, loop=0
                        )
                        return _orig_video_init(
                            self, _tmp.name, caption=caption, fps=30, format="gif"
                        )
                except Exception:
                    pass
                return _orig_video_init(
                    self, data_or_path, caption=caption, fps=fps, format=format
                )

            _wandb.Video.__init__ = _video_init

            _task_re = re.compile(r"^evaluation/task(\d+)_.*_success$")
            _orig_log = _wandb.log

            def _fb_keys(data):
                extra = {}
                for k, v in list(data.items()):
                    if k == "evaluation/overall_success":
                        extra["eval/reward/eval/success"] = v
                        continue
                    if k == "video":
                        extra["eval_video/all_tasks"] = v
                        continue
                    m = _task_re.match(k)
                    if m:
                        n = m.group(1)
                        extra[
                            f"eval/reward/{_env_base}-singletask-task{n}-v0/success"
                        ] = v
                return extra

            @functools.wraps(_orig_log)
            def _log(data=None, *args, **kwargs):
                try:
                    if isinstance(data, dict):
                        data = {**data, **_fb_keys(data)}
                except Exception:
                    pass
                return _orig_log(data, *args, **kwargs)

            _wandb.log = _log
    except Exception:
        pass

# --- DrQ visual front-end injection (independent of the wandb block above) ---
# Register the JAX DrQ encoder so configs may select encoder='drq', and -- only
# when OGBENCH_DRQ_FRONTEND=1 -- swap GCDataset.augment for FB's random_shifts.
# Guarded: this file is also auto-imported by non-OGBench/non-jax processes (e.g.
# the wandb-shim tests under .venv), where these imports must fail silently.
# run_gciql.py runs `python main.py` with cwd=third_party/ogbench/impls, so
# sys.path[0] is that dir and `utils.*` import here at interpreter startup.
try:
    import utils.encoders as _ogb_encoders
    from ogbench_drq import DrQEncoder as _DrQEncoder

    _ogb_encoders.encoder_modules.setdefault("drq", _DrQEncoder)

    if os.environ.get("OGBENCH_DRQ_FRONTEND") == "1":
        import utils.datasets as _ogb_datasets
        from ogbench_drq import gcdataset_augment as _drq_augment

        _ogb_datasets.GCDataset.augment = _drq_augment
except Exception:
    pass

# --- CRL+FlowBC agent injection (independent of the blocks above) ---
# Register CRLFlowBCAgent so configs may select agent_name='crl_flowbc' via
# --agent=tools/wandb_mode_shim/crl_flowbc.py. Guarded: inert in non-jax
# processes (e.g. the .venv tests) where these imports fail.
try:
    import agents as _ogb_agents
    from crl_flowbc import CRLFlowBCAgent as _CRLFlowBCAgent

    _ogb_agents.agents.setdefault("crl_flowbc", _CRLFlowBCAgent)
except Exception:
    pass

# --- Resume support (opt-in via env; set by run_gciql for restore_* runs) ---
# Vendored main.py always counts the loop i=1..train_steps and writes params_{i},
# truncates train.csv/eval.csv, and logs wandb step=i. To make a run resumed from
# checkpoint epoch N reproduce the SAME on-disk layout a single 0..T run would
# have produced, patch (here, at interpreter startup, BEFORE main.py's
# `from utils... import ...` binds these names):
#   GCIQL_EXP_NAME    -> pin get_exp_name so save_dir + wandb run name reuse the
#                        original timestamped dir instead of making a new one.
#   GCIQL_STEP_OFFSET -> add N to every saved checkpoint epoch and every wandb /
#                        CSV step, so we get params_{N+i} and contiguous axes.
#   GCIQL_CSV_APPEND  -> append to an existing train.csv/eval.csv (no truncation,
#                        reuse its header) instead of overwriting.
# Fully inert (no patches) unless one of these is set, so unrelated runs and the
# .venv shim tests are unaffected.
_exp_name = os.environ.get("GCIQL_EXP_NAME")
_step_offset = int(os.environ.get("GCIQL_STEP_OFFSET") or 0)
_csv_append = os.environ.get("GCIQL_CSV_APPEND") == "1"

if _exp_name or _step_offset or _csv_append:
    try:
        import utils.log_utils as _lu
        import utils.flax_utils as _fu

        if _exp_name:
            _lu.get_exp_name = lambda seed, _n=_exp_name: _n

        if _step_offset:
            _orig_save_agent = _fu.save_agent

            @functools.wraps(_orig_save_agent)
            def _save_agent(agent, save_dir, epoch, _o=_step_offset, _f=_orig_save_agent):
                return _f(agent, save_dir, epoch + _o)

            _fu.save_agent = _save_agent

        if _step_offset or _csv_append:
            _OrigCsvLogger = _lu.CsvLogger

            class _ResumeCsvLogger(_OrigCsvLogger):
                """Offset every logged step by N and (optionally) append to an
                existing CSV instead of truncating it on the first write."""

                def __init__(self, path):
                    super().__init__(path)
                    self._append = (
                        _csv_append
                        and os.path.exists(path)
                        and os.path.getsize(path) > 0
                    )
                    if self._append:
                        with open(path) as _fh:
                            _first = _fh.readline().strip()
                        self.header = _first.split(",") if _first else None

                def log(self, row, step):
                    step = step + _step_offset
                    if self._append and self.file is None:
                        # First write of a resumed run: open in append mode and
                        # reuse the existing header (skip the truncating 'w' open
                        # in the upstream first-write branch).
                        self.file = open(self.path, "a")
                        _row = {**row, "step": step}
                        if self.header is None:
                            self.header = [
                                k for k, v in _row.items()
                                if not isinstance(v, self.disallowed_types)
                            ]
                        _filtered = {
                            k: v for k, v in _row.items()
                            if not isinstance(v, self.disallowed_types)
                        }
                        self.file.write(
                            ",".join(str(_filtered.get(k, "")) for k in self.header) + "\n"
                        )
                        self.file.flush()
                        return
                    super().log(row, step)

            _lu.CsvLogger = _ResumeCsvLogger
    except Exception:
        pass

# wandb step offset is patched separately: it must compose with the _fb_keys
# wrapper installed above and stay independent of utils.* importability.
if _step_offset:
    try:
        import wandb as _wandb_resume

        _prev_wandb_log = _wandb_resume.log

        @functools.wraps(_prev_wandb_log)
        def _log_step_offset(data=None, *args, _o=_step_offset, _f=_prev_wandb_log, **kwargs):
            if kwargs.get("step") is not None:
                kwargs["step"] = kwargs["step"] + _o
            return _f(data, *args, **kwargs)

        _wandb_resume.log = _log_step_offset
    except Exception:
        pass
