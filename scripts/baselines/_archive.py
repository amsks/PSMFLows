"""Read-only bridge to the FROZEN agents under `archive/`.

`archive/` is deliberately not importable from `agents/__init__.py` (see archive/README.md):
its agents are the negative results of record and the registry omission is the point. A
*baseline* run still has to instantiate one, so this module supplies the two things such a
run needs and nothing else:

  * `archive_agents()` -- name -> class, imported lazily as a namespace package
    (`archive.agents.<name>`), so nothing moves out of archive/;
  * `register_archive_configs()` -- splices `archive/configs` into hydra's search path so
    `agent=affine_psm` resolves to `archive/configs/agent/affine_psm.yaml` with the usual
    `agent.<dotted>=<value>` override syntax on top.

Nothing under archive/ is read for anything but import, and nothing there is modified.
Deleting this directory (`scripts/baselines/`) reverts the repo exactly.
"""
import importlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVE_CONFIGS = os.path.join(REPO, 'archive', 'configs')

# agent_name -> (module under archive.agents, class). Only the successor-measure agents are
# listed; the off-the-shelf baselines (iql/sac/...) are reachable the same way if needed.
_ARCHIVE_AGENTS = {
    'psm': ('archive.agents.psm', 'PSMAgent'),
    'affine_psm': ('archive.agents.affine_psm', 'AffinePSMAgent'),
    'latent_affine_psm': ('archive.agents.latent_affine_psm', 'LatentAffinePSMAgent'),
    'fb': ('archive.agents.fb', 'FBAgent'),
}


def archive_agents():
    """The archived agents, keyed like `agents.agents`. Import errors surface here."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    out = {}
    for name, (mod, cls) in _ARCHIVE_AGENTS.items():
        out[name] = getattr(importlib.import_module(mod), cls)
    return out


def all_agents():
    """Live registry first, archived agents layered under it (live wins on a name clash)."""
    from agents import agents as live
    merged = archive_agents()
    merged.update(live)
    return merged


def register_archive_configs():
    """Make `archive/configs/agent/*.yaml` visible to hydra's `agent` config group.

    Must be called BEFORE @hydra.main runs. Uses the documented SearchPathPlugin hook
    rather than a CLI `hydra.searchpath=` so the caller's command line is the ordinary one.
    """
    from hydra.core.config_search_path import ConfigSearchPath
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    class _ArchiveSearchPath(SearchPathPlugin):
        def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
            search_path.append(provider='psmflows-archive', path=f'file://{ARCHIVE_CONFIGS}')

    Plugins.instance().register(_ArchiveSearchPath)
