"""Plugin registry: auto-discovery + validation on registration."""
from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
from typing import Any

from renderprobe.core.schema import (
    ANSWER_TYPES,
    CONDITIONS,
    Analyzer,
    ModelAdapter,
    Probe,
    Renderer,
    SceneGenerator,
)


class RegistrationError(Exception):
    pass


class Registry:
    def __init__(self) -> None:
        self._scenes: dict[str, SceneGenerator] = {}
        self._probes: dict[str, Probe] = {}
        self._models: dict[str, ModelAdapter] = {}
        self._analyzers: dict[str, Analyzer] = {}
        self._renderers: dict[str, Renderer] = {}

    # Registration

    def register_scene(self, plugin: Any) -> None:
        _require_protocol(plugin, SceneGenerator, "SceneGenerator")
        _require_attr(plugin, "name", str)
        _require_attr(plugin, "params_schema", dict)
        self._scenes[plugin.name] = plugin

    def register_probe(self, plugin: Any) -> None:
        _require_protocol(plugin, Probe, "Probe")
        _require_attr(plugin, "name", str)
        _require_attr(plugin, "answer_type", str)
        if plugin.answer_type not in ANSWER_TYPES:
            raise RegistrationError(
                f"Probe '{plugin.name}': answer_type must be one of {ANSWER_TYPES}, "
                f"got '{plugin.answer_type}'"
            )
        _require_attr(plugin, "requires_gt_fields", list)
        self._probes[plugin.name] = plugin

    def register_model(self, plugin: Any) -> None:
        _require_protocol(plugin, ModelAdapter, "ModelAdapter")
        _require_attr(plugin, "name", str)
        _require_attr(plugin, "is_open", bool)
        self._models[plugin.name] = plugin

    def register_analyzer(self, plugin: Any) -> None:
        _require_protocol(plugin, Analyzer, "Analyzer")
        _require_attr(plugin, "name", str)
        _require_attr(plugin, "requires_conditions", list)
        _require_attr(plugin, "requires_varying_factor", bool)
        for cond in plugin.requires_conditions:
            if cond not in CONDITIONS:
                raise RegistrationError(
                    f"Analyzer '{plugin.name}': unknown condition '{cond}'. "
                    f"Must be one of {CONDITIONS}."
                )
        self._analyzers[plugin.name] = plugin

    def register_renderer(self, plugin: Any) -> None:
        _require_protocol(plugin, Renderer, "Renderer")
        _require_attr(plugin, "name", str)
        _require_attr(plugin, "is_3d", bool)
        self._renderers[plugin.name] = plugin

    # Lookup

    def get_scene(self, name: str) -> SceneGenerator:
        return _get(self._scenes, name, "SceneGenerator")

    def get_probe(self, name: str) -> Probe:
        return _get(self._probes, name, "Probe")

    def get_model(self, name: str) -> ModelAdapter:
        return _get(self._models, name, "ModelAdapter")

    def get_analyzer(self, name: str) -> Analyzer:
        return _get(self._analyzers, name, "Analyzer")

    def get_renderer(self, name: str) -> Renderer:
        return _get(self._renderers, name, "Renderer")

    # Catalog (sorted names; used by the UI to populate selectors)

    def list_scenes(self) -> list[str]:
        return sorted(self._scenes)

    def list_probes(self) -> list[str]:
        return sorted(self._probes)

    def list_models(self) -> list[str]:
        return sorted(self._models)

    def list_analyzers(self) -> list[str]:
        return sorted(self._analyzers)

    def list_renderers(self) -> list[str]:
        return sorted(self._renderers)

    # Auto-discovery

    def autodiscover(self, package_root: Path | None = None) -> None:
        """Import all modules in the four plugin packages and register their
        plugins. A module exposes plugins via a ``PLUGIN`` attribute (single)
        and/or a ``PLUGINS`` attribute (list) - both are honored."""
        if package_root is None:
            # default: the renderprobe package alongside this file
            package_root = Path(__file__).parent.parent

        families = {
            "scenes": self.register_scene,
            "probes": self.register_probe,
            "models": self.register_model,
            "analysis": self.register_analyzer,
            "renderers": self.register_renderer,
        }
        for family, register_fn in families.items():
            pkg_path = package_root / family
            if not pkg_path.exists():
                continue
            pkg_name = f"renderprobe.{family}"
            for mod_info in pkgutil.iter_modules([str(pkg_path)]):
                if mod_info.name.startswith("_"):
                    continue
                module = importlib.import_module(f"{pkg_name}.{mod_info.name}")
                for plugin in _module_plugins(module):
                    try:
                        register_fn(plugin)
                    except RegistrationError as exc:
                        raise RegistrationError(
                            f"Auto-discovery failed for {pkg_name}.{mod_info.name}: {exc}"
                        ) from exc

    # Out-of-tree plugins (a scientist's own scenes/probes/metrics)

    def load_external(self, path: Path | str) -> list[str]:
        """Load third-party plugins from a file or directory OUTSIDE the renderprobe
        source tree, so a scientist keeps their own scenes/probes/metrics in their
        own repo.

        Unlike ``autodiscover`` (which imports by the ``renderprobe.<family>`` dotted
        name), this imports each module by FILE PATH, then registers it to whichever
        family Protocol it satisfies. This is the "drop a ``.py`` in a folder and run"
        path - no editing of the installed package.

        - A directory: every non-underscore ``*.py`` directly inside it (non-recursive).
        - A single file: that file, regardless of its name (so a copied ``_template.py``
          can be validated before it is renamed).

        Returns a list like ``["scene:my_scene", "probe:my_probe"]`` describing what
        registered. Raises ``RegistrationError`` with a plugin-author-friendly message
        on any import, conformance, or family-detection failure.
        """
        path = Path(path)
        if path.is_dir():
            files = sorted(p for p in path.glob("*.py") if not p.name.startswith("_"))
            if not files:
                raise RegistrationError(
                    f"no plugin files in {path} (looked for non-underscore *.py)."
                )
        elif path.is_file():
            files = [path]
        else:
            raise RegistrationError(f"plugin path not found: {path}")

        registered: list[str] = []
        for f in files:
            module = _import_from_file(f)
            plugins = _module_plugins(module)
            if not plugins:
                raise RegistrationError(
                    f"{f.name}: no plugins found - expose a module-level PLUGIN or "
                    f"PLUGINS holding a SceneGenerator / Probe / ModelAdapter / "
                    f"Analyzer instance."
                )
            for plugin in plugins:
                family = self._register_by_family(plugin, f)
                registered.append(f"{family}:{getattr(plugin, 'name', '?')}")
        return registered

    def _register_by_family(self, plugin: Any, source: Path) -> str:
        """Detect which family a plugin belongs to (by the Protocol it satisfies) and
        register it. Requires exactly one match, so a malformed plugin fails loudly
        instead of registering as the wrong kind of thing."""
        families = [
            ("scene", SceneGenerator, self.register_scene),
            ("probe", Probe, self.register_probe),
            ("model", ModelAdapter, self.register_model),
            ("analyzer", Analyzer, self.register_analyzer),
            ("renderer", Renderer, self.register_renderer),
        ]
        matches = [(fam, fn) for fam, proto, fn in families if isinstance(plugin, proto)]
        pname = getattr(plugin, "name", repr(plugin))
        if not matches:
            raise RegistrationError(
                f"{source.name}: plugin '{pname}' satisfies no RenderProbe protocol "
                f"(SceneGenerator / Probe / ModelAdapter / Analyzer). Check it defines "
                f"the required attributes and methods."
            )
        if len(matches) > 1:
            fams = ", ".join(f for f, _ in matches)
            raise RegistrationError(
                f"{source.name}: plugin '{pname}' ambiguously matches multiple families "
                f"({fams}); a plugin must be exactly one."
            )
        family, register_fn = matches[0]
        register_fn(plugin)  # re-validates attrs/enums; may raise RegistrationError
        return family


# Helpers

def _import_from_file(path: Path) -> Any:
    """Import a module from an explicit file path (out-of-tree plugin load)."""
    mod_name = f"renderprobe_ext_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RegistrationError(f"cannot import plugin file: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses / pickling / self-referential imports resolve.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # surface the author's own syntax/import error clearly
        sys.modules.pop(mod_name, None)
        raise RegistrationError(f"error importing {path.name}: {exc}") from exc
    return module

def _module_plugins(module: Any) -> list[Any]:
    """Collect a module's registrable plugins from PLUGIN and/or PLUGINS."""
    plugins: list[Any] = []
    single = getattr(module, "PLUGIN", None)
    if single is not None:
        plugins.append(single)
    many = getattr(module, "PLUGINS", None)
    if many is not None:
        plugins.extend(many)
    return plugins


def _require_protocol(plugin: Any, protocol: type, label: str) -> None:
    if not isinstance(plugin, protocol):
        raise RegistrationError(
            f"Plugin '{getattr(plugin, 'name', repr(plugin))}' does not satisfy "
            f"the {label} protocol. Missing methods: "
            f"{_missing_methods(plugin, protocol)}"
        )


def _missing_methods(plugin: Any, protocol: type) -> list[str]:
    return [
        m for m in vars(protocol) if not m.startswith("_") and not hasattr(plugin, m)
    ]


def _require_attr(plugin: Any, attr: str, expected_type: type) -> None:
    if not hasattr(plugin, attr):
        raise RegistrationError(
            f"Plugin '{getattr(plugin, 'name', repr(plugin))}' is missing attribute '{attr}'."
        )
    val = getattr(plugin, attr)
    if not isinstance(val, expected_type):
        raise RegistrationError(
            f"Plugin '{plugin.name}': '{attr}' must be {expected_type.__name__}, "
            f"got {type(val).__name__}."
        )


def _get(store: dict, name: str, label: str) -> Any:
    if name not in store:
        raise KeyError(f"{label} '{name}' not found. Registered: {list(store)}")
    return store[name]


# Module-level default registry
_default = Registry()


def get_default_registry() -> Registry:
    return _default
