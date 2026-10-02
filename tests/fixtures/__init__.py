"""Scenes, probes and model doubles that exist only to exercise the instrument offline.

None of this ships. `src/renderprobe/` holds the product - two calibrated scenes, their
probes, the renderers, the analyzers and the runner - and everything here is the rig that
proves the product works with no network and no API key.

These scenes predate the calibration gate and were never taken through it, so a model's
accuracy on them sits wherever it happens to sit and means nothing about the model. They
are kept because deleting them would take the evidence with them: `scene_blocks` backs the
occlusion certificate and the Mitsuba renderer tests, `scene_relational` and
`scene_spatial_config` back the decomposition cascade, the oracle-validity certificate and
the calibration and taxonomy analyzers.

The mock models are deterministic doubles, not models. They read pixels honestly where
they can and are seeded per prompt, so a test can assert an exact bottleneck verdict.

Usage: build a registry the normal way, then add these.

    reg = Registry()
    reg.autodiscover()        # the product
    register(reg)             # the rig
"""
from __future__ import annotations

from typing import Any

MODULES = (
    "scene_blocks",
    "scene_dots",
    "scene_pathfinding",
    "scene_relational",
    "scene_spatial_config",
    "scene_tabletop_occlusion",
    "probe_count",
    "probe_identity_under_occlusion",
    "probe_pathfinding",
    "probe_relational",
    "probe_spatial_relation",
    "mock",
)

# Names these modules register, so a test can assert on them without importing each one.
SCENES = ("blocks", "dots", "pathfinding", "relational", "spatial_config",
          "tabletop_occlusion")
PROBES = ("count", "identity_under_occlusion", "pathfinding", "relational",
          "spatial_relation")
MODELS = ("mock", "mock_arbitration", "mock_noisy")


def register(registry: Any) -> list[str]:
    """Register every fixture plugin onto `registry`, and say what was added.

    Imported normally rather than loaded by path: these are a package, they import each
    other, and `load_external` deliberately imports a lone file with no package around it.
    """
    import importlib

    added: list[str] = []
    for name in MODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        plugins = []
        single = getattr(module, "PLUGIN", None)
        if single is not None:
            plugins.append(single)
        plugins.extend(getattr(module, "PLUGINS", None) or [])
        for plugin in plugins:
            added.append(registry._register_by_family(plugin, module.__file__))
    return added
