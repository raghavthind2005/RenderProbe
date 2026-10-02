"""Command-line interface: renderprobe run <config.yaml>"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from renderprobe.core.registry import RegistrationError, Registry
from renderprobe.core.runner import ExperimentConfig, render_plan, run_experiment

# family -> (package dir under renderprobe/, the template's placeholder `name` value).
# `renderprobe new <family> <name>` copies the template and renames it.
_TEMPLATES = {
    "scene": ("scenes", "template_scene"),
    "probe": ("probes", "template_probe"),
    "renderer": ("renderers", "template_renderer"),
    "model": ("models", "template_model"),
    "analyzer": ("analysis", "template_metric"),
}


def _load_config(path: Path) -> ExperimentConfig:
    """Read a YAML experiment config from disk into an ExperimentConfig."""
    return _config_from_raw(yaml.safe_load(path.read_text()))


def _config_from_raw(raw: dict) -> ExperimentConfig:
    scene_cfg = raw["scene"]
    params = scene_cfg.get("params", {})
    seeds_raw = scene_cfg.get("seeds", [0])
    # seeds may be an int (single) or list
    seeds = seeds_raw if isinstance(seeds_raw, list) else [seeds_raw]

    return ExperimentConfig(
        scene_generator=scene_cfg["generator"],
        scene_params=params,
        seeds=seeds,
        probe=raw["probe"],
        models=raw["models"],
        analyzers=raw.get("analysis", []),
        # Optional: list of oracle verbalization variant names to run.
        # Omit for canonical-only; "all" expands to every variant the probe offers.
        oracle_variants=_parse_oracle_variants(raw.get("oracle_variants")),
        # Optional: ask models for a 0-100 confidence, for calibration.
        elicit_confidence=bool(raw.get("elicit_confidence", False)),
        # Optional: also ask every question with the images withheld, for visual_gain.
        blind=bool(raw.get("blind", False)),
    )


def _parse_oracle_variants(value) -> list[str] | None:
    """Config oracle_variants -> runner selection.

    None -> None (canonical only); "all" -> ["all"] (every variant); a single name
    -> [name]; a list -> the list.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _print_reports(reports) -> None:
    for report in reports:
        print(f"\n{'='*60}")
        print(f"  {report.name.upper()}")
        print(f"{'='*60}")
        print(json.dumps(report.payload, indent=2))


def _load_plugin_dirs(registry: Registry, dirs: list[str]) -> bool:
    """Load out-of-tree plugins into the registry; return False on any failure."""
    for pd in dirs:
        try:
            added = registry.load_external(pd)
        except RegistrationError as exc:
            print(f"Error loading plugins from {pd}: {exc}", file=sys.stderr)
            return False
        print(f"  loaded external plugins from {pd}: {', '.join(added) or '(none)'}")
    return True


def run_cmd(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    raw = yaml.safe_load(config_path.read_text())
    cfg = _config_from_raw(raw)
    # Out-of-tree plugins: config `plugin_dirs:` plus any --plugin-dir flags.
    plugin_dirs = list(raw.get("plugin_dirs", []) or []) + list(args.plugin_dir or [])

    registry = Registry()
    registry.autodiscover()
    if not _load_plugin_dirs(registry, plugin_dirs):
        return 1

    print(f"Running experiment: {config_path.stem}")
    print(f"  scene={cfg.scene_generator}, probe={cfg.probe}, "
          f"models={cfg.models}, seeds={cfg.seeds}")

    plan = render_plan(cfg)
    print(f"  render plan: {plan['n_scenes']} scene image(s) via {plan['renderers']} "
          f"(each rendered once, reused across models/conditions)")
    if plan["heavy"]:
        print(f"  note: {plan['heavy']} render true 3D (~seconds/image). For fast iteration "
              f"use pil_2d/pil_25d; lower cost with scene params `renderer_opts: {{spp: N}}`.")

    reports = run_experiment(cfg, registry)
    _print_reports(reports)

    print("\nDone.")
    return 0


def power_cmd(args: argparse.Namespace) -> int:
    """Report the paired-n needed to detect a given recovery (McNemar power)."""
    from renderprobe.analysis.stats import mcnemar_power_n

    try:
        n = mcnemar_power_n(
            effect=args.effect,
            alpha=args.alpha,
            power=args.power,
            discordance=args.discordance,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Paired McNemar power analysis\n"
        f"  target recovery |effect| : {args.effect}\n"
        f"  alpha                    : {args.alpha}\n"
        f"  power                    : {args.power}\n"
        f"  assumed discordant rate  : {args.discordance}\n"
        f"  --> required paired n    : {n} scenes per cell\n"
        f"\nThe discordant rate is the assumed fraction of scenes answered "
        f"differently across\nconditions; it is a nuisance parameter you cannot "
        f"know a priori. Lower discordance\nneeds larger n - re-run with a few "
        f"values (e.g. --discordance 0.2, then 0.4) to bracket it."
    )
    return 0


def validate_cmd(args: argparse.Namespace) -> int:
    """Check an out-of-tree plugin: Protocol conformance + the honesty invariants
    (trustable ground truth, reproducibility, answer balance, oracle presence)."""
    from renderprobe.core.validate import all_ok, format_reports, validate_path

    path = Path(args.path)
    if not path.exists():
        print(f"Error: plugin path not found: {path}", file=sys.stderr)
        return 1

    print(f"Validating {path} ...")
    reports = validate_path(path, seeds=args.seeds, probe_name=args.probe,
                            renderer_name=args.renderer)
    print(format_reports(reports))
    ok = all_ok(reports)
    warned = [pv for pv in reports if pv.ok and pv.warned]
    if not ok:
        print("\nVALIDATION FAILED")
    elif warned:
        # A warning means a check could not RUN, not that it succeeded. A scene with no
        # probe warns and then skips every check of the answer - which is most of the
        # battery - so a bare "PASSED" here would be the tool overstating what it knows.
        names = ", ".join(f"{pv.family} '{pv.name}'" for pv in warned)
        print(f"\nVALIDATION PASSED WITH WARNINGS  ({names})")
        print("A warning is a check that could not run, not one that succeeded. The "
              "checks it gated are listed above as skipped.")
    else:
        print("\nVALIDATION PASSED")
    return 0 if ok else 1


def list_cmd(args: argparse.Namespace) -> int:
    """Show the plugin catalog - every scene/probe/renderer/model/analyzer available,
    including any loaded from --plugin-dir, so a scientist can see what they can compose."""
    registry = Registry()
    registry.autodiscover()
    if args.plugin_dir and not _load_plugin_dirs(registry, args.plugin_dir):
        return 1

    scenes = registry.list_scenes()
    print(f"\nscenes ({len(scenes)}):")
    for n in scenes:
        params = list(getattr(registry.get_scene(n), "params_schema", {}) or {})
        print(f"  {n:24s} params: {params}")

    probes = registry.list_probes()
    print(f"\nprobes ({len(probes)}):")
    for n in probes:
        p = registry.get_probe(n)
        print(f"  {n:24s} answer={p.answer_type}, needs_gt={list(p.requires_gt_fields)}")

    renderers_ = registry.list_renderers()
    print(f"\nrenderers ({len(renderers_)}):")
    for n in renderers_:
        r = registry.get_renderer(n)
        ids = "render_ids" if callable(getattr(r, "render_ids", None)) else "no render_ids"
        print(f"  {n:24s} is_3d={r.is_3d}, {ids}")

    # Test fixtures are deliberately listed apart from models: they are test
    # doubles, and presenting them as peers of a real system invites a result
    # produced by a fixture being read as a measurement.
    all_models = registry.list_models()
    fixtures = [n for n in all_models
                if getattr(registry.get_model(n), "is_fixture", False)]
    real = [n for n in all_models if n not in fixtures]
    print(f"\nmodels ({len(real)}): {real}")
    if fixtures:
        print(f"test fixtures (not models, not for experiments): {fixtures}")
    print(f"analyzers ({len(registry.list_analyzers())}): {registry.list_analyzers()}")
    return 0


def new_cmd(args: argparse.Namespace) -> int:
    """Scaffold a new plugin from its family template into the current (or --dir) folder,
    pre-named, ready to edit and `renderprobe validate`."""
    fam = args.family
    if fam not in _TEMPLATES:
        print(f"Error: unknown family '{fam}'. Choose from: {', '.join(_TEMPLATES)}",
              file=sys.stderr)
        return 1
    if not re.match(r"^[a-z][a-z0-9_]*$", args.name):
        print(f"Error: name '{args.name}' must be a lowercase snake_case identifier "
              f"(e.g. my_scene)", file=sys.stderr)
        return 1

    pkg, placeholder = _TEMPLATES[fam]
    template = Path(__file__).parent / pkg / "_template.py"
    if not template.exists():
        print(f"Error: template not found: {template}", file=sys.stderr)
        return 1

    out = Path(args.dir) / f"{args.name}.py"
    if out.exists() and not args.force:
        print(f"Error: {out} already exists (use --force to overwrite)", file=sys.stderr)
        return 1

    text = template.read_text().replace(f'name = "{placeholder}"', f'name = "{args.name}"')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Created {out}  (a {fam} named '{args.name}')")
    # A scene on its own has no question attached to it, so validate can check that it
    # draws and is deterministic but nothing about the answer. Saying so here is what
    # stops somebody reading a warning-only pass as a clean one.
    if fam == "scene":
        print("Next: pair it with a probe, which is what asks a question about it:")
        print(f"  renderprobe new probe {args.name}_probe --dir {out.parent}")
        print("The two templates fit together as generated, so you can validate and run")
        print("them before editing either. Then fill in your own logic and check with:")
    elif fam == "probe":
        print("Next: pair it with a scene, which is what it asks its question about:")
        print(f"  renderprobe new scene {args.name}_scene --dir {out.parent}")
        print("The two templates fit together as generated. Then check them with:")
    else:
        print("Next: open it, rename the class and fill in the logic, then check it with:")
    print(f"  renderprobe validate {out.parent if fam in ('scene', 'probe') else out}")
    return 0


def ui_cmd(args: argparse.Namespace) -> int:
    try:
        from renderprobe.ui.app import launch
    except ImportError as exc:
        print(
            f"UI dependencies missing ({exc}). Install with: pip install 'renderprobe[ui]'",
            file=sys.stderr,
        )
        return 1
    launch(share=args.share, server_port=args.port)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="renderprobe")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run an experiment from a YAML config")
    run_p.add_argument("config", help="Path to experiment YAML")
    run_p.add_argument(
        "--plugin-dir", action="append", metavar="PATH",
        help="Load out-of-tree plugins from a file or directory (repeatable). "
             "Combined with any `plugin_dirs:` in the config.",
    )

    val_p = sub.add_parser(
        "validate",
        help="Check an out-of-tree plugin (conformance + ground-truth/oracle honesty)",
    )
    val_p.add_argument("path", help="Plugin file or directory to validate")
    val_p.add_argument(
        "--seeds", type=int, default=12,
        help="Seeds to sample for the answer-balance check (default 12)",
    )
    val_p.add_argument(
        "--probe", default=None,
        help="Probe name to pair with a scene under test (default: auto-detect)",
    )
    val_p.add_argument(
        "--renderer", default=None,
        help="Certify a graph-based scene is occlusion-safe under this renderer "
             "(e.g. pil_25d, mitsuba_3d). Default: the scene's own renderer, else pil_2d.",
    )

    list_p = sub.add_parser("list", help="List available plugins (scenes/probes/renderers/...)")
    list_p.add_argument(
        "--plugin-dir", action="append", metavar="PATH",
        help="Also list out-of-tree plugins from this file or directory (repeatable).",
    )

    new_p = sub.add_parser("new", help="Scaffold a new plugin from its family template")
    new_p.add_argument("family", choices=list(_TEMPLATES),
                       help="Which kind of plugin to create")
    new_p.add_argument("name", help="Name for the new plugin (lowercase snake_case)")
    new_p.add_argument("--dir", default=".", help="Directory to create it in (default: .)")
    new_p.add_argument("--force", action="store_true", help="Overwrite if the file exists")

    ui_p = sub.add_parser("ui", help="Launch the interactive Gradio UI")
    ui_p.add_argument("--port", type=int, default=7860, help="Server port")
    ui_p.add_argument("--share", action="store_true", help="Create a public share link")

    power_p = sub.add_parser(
        "power",
        help="Paired-n needed to detect a given recovery (McNemar power analysis)",
    )
    power_p.add_argument(
        "--effect", type=float, required=True,
        help="Target recovery |acc_oracle - acc_full| to detect (e.g. 0.1)",
    )
    power_p.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    power_p.add_argument("--power", type=float, default=0.80, help="Desired power")
    power_p.add_argument(
        "--discordance", type=float, default=0.30,
        help="Assumed fraction of scenes answered differently across conditions",
    )

    args = parser.parse_args()
    if args.command == "run":
        sys.exit(run_cmd(args))
    elif args.command == "validate":
        sys.exit(validate_cmd(args))
    elif args.command == "list":
        sys.exit(list_cmd(args))
    elif args.command == "new":
        sys.exit(new_cmd(args))
    elif args.command == "ui":
        sys.exit(ui_cmd(args))
    elif args.command == "power":
        sys.exit(power_cmd(args))


if __name__ == "__main__":
    main()
