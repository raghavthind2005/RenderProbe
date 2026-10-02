"""Renderers - turn a renderer-agnostic ``SceneGraph`` into pixels.

Realism is a pluggable knob: the SAME graph (and therefore the SAME exact ground truth)
can be drawn flat (``pil_2d``), depth-shaded (``pil_25d``), or photorealistically in true
3D. Two photoreal tiers ship, both optional: ``mitsuba_3d`` (a headless CPU path tracer -
no OpenGL/GPU, works everywhere; ``pip install 'renderprobe[render3d-cpu]'``) and
``pyrender_3d`` (GPU/OpenGL, for machines with a GL backend). A scene picks a renderer by
name via ``render_graph``; the choice never changes what the scene means.

Built-in renderers are discovered by scanning this package (``_``-prefixed modules are
skipped). They are also registered as a plugin family in ``core/registry.py`` so the
tooling can enumerate/validate them and load third-party ones.

RESOLVING A SCIENTIST'S OWN RENDERER
    A graph scene resolves a renderer by NAME (``render_graph(graph, name)``) from inside
    its ``generate`` - whose signature is frozen, so it cannot be handed a registry. To let
    a scene reach an out-of-tree ``--plugin-dir`` renderer, the runner and ``validate`` wrap
    scene generation in ``with using_registry(reg): ...``. Inside that scope ``render_graph``
    resolves names through the registry FIRST (so externals and any built-in both work),
    then falls back to the built-ins. The scope is restored on exit - no global leaks
    between runs or across tests.
"""
from __future__ import annotations

import importlib
import pkgutil
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from renderprobe.core.schema import SceneGraph

# Optional, scoped resolver (a Registry, or anything with .get_renderer / .list_renderers)
# installed by the runner / validate around scene generation. None outside such a scope.
_resolver: Any = None


@contextmanager
def using_registry(resolver: Any):
    """Within this context, ``render_graph`` / ``get_renderer`` resolve renderer names
    through ``resolver`` (typically a Registry, including its ``--plugin-dir`` renderers)
    first, then the built-ins. Restored on exit; safe to nest."""
    global _resolver
    prev = _resolver
    _resolver = resolver
    try:
        yield
    finally:
        _resolver = prev


@lru_cache(maxsize=1)
def _builtin() -> dict[str, Any]:
    out: dict[str, Any] = {}
    pkg_dir = Path(__file__).parent
    for mod_info in pkgutil.iter_modules([str(pkg_dir)]):
        if mod_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"renderprobe.renderers.{mod_info.name}")
        plugin = getattr(module, "PLUGIN", None)
        if plugin is not None:
            out[plugin.name] = plugin
    return out


def available() -> list[str]:
    """Names of the resolvable renderers - the built-ins, plus any renderers in the active
    resolver (a scientist's ``--plugin-dir`` ones) when inside a ``using_registry`` scope."""
    names = set(_builtin())
    if _resolver is not None and hasattr(_resolver, "list_renderers"):
        try:
            names |= set(_resolver.list_renderers())
        except Exception:
            pass
    return sorted(names)


def get_renderer(name: str) -> Any:
    """Resolve a renderer by name: the active resolver (external + built-in) first, then
    the built-ins. Raises KeyError naming what IS available if nothing matches."""
    if _resolver is not None:
        try:
            return _resolver.get_renderer(name)
        except KeyError:
            pass   # not in the resolver - try the built-ins below
    table = _builtin()
    if name in table:
        return table[name]
    raise KeyError(f"renderer '{name}' not found; available: {available()}")


def render_graph(graph: SceneGraph, name: str = "pil_2d") -> list[Image.Image]:
    """Render ``graph`` with the named renderer (default: flat 2D)."""
    return get_renderer(name).render(graph)
