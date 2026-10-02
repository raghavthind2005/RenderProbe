"""Exercise every UI feature against every showcase scene, and print what is missing.

The UI is the surface the project is judged on, so a tab that is empty for one scene and
full for another is a defect whether or not anything raised. This runs a small real
experiment per scene with every optional condition switched on, pushes the results
through each function the UI calls, and reports one row per feature per scene:

    OK     the feature produced content
    EMPTY  it ran and produced nothing (a blank tab for that scene)
    NONE   it ran and produced nothing, legitimately (nothing failed, so the Inspect
           tab has nothing to show)
    ERROR  it raised

    set -a && . ./.env && set +a
    python tools/check_features.py                 # all three showcase scenes
    python tools/check_features.py --scene route   # just one

Replies are cached by tools/run_experiment.py's cache, so a second run is free.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import _Cache, wrap_models  # noqa: E402

from renderprobe.core.registry import Registry  # noqa: E402
from renderprobe.core.runner import ExperimentConfig, run_experiment_collect  # noqa: E402
from renderprobe.ui import app as A  # noqa: E402
from renderprobe.ui import data as D  # noqa: E402

# One small, fully-featured run per scene: a swept factor so the sweep tab has something
# to draw, the text-only baseline so visual_gain is defined, and elicited confidences so
# calibration is. Two values and two seeds keeps it to 16 calls a scene.
SCENES = {
    "route": dict(params={"n_towns": 8, "n_hops": [2, 3]}, sweep="n_hops", dirs=[]),
    "polycube": dict(params={"n_pieces": [4, 5], "renderer_opts": {"spp": 24}},
                     sweep="n_pieces", dirs=[]),
    "gear_train": dict(params={"n_stages": [1, 2]}, sweep="n_stages", dirs=["examples"]),
}
MODEL = "gemma-3-27b-or"
ANALYZERS = ["accuracy", "visual_gain", "decomposition", "oracle", "taxonomy",
             "calibration", "sweep_curve"]


def _status(fn, empty_ok: bool = False):
    """Run a UI accessor and say whether a user would see anything.

    `empty_ok` marks a feature whose emptiness is a result, not a defect: the inspect
    tab lists wrong-or-errored cases, so nothing to show means the model got them all
    right on that condition.
    """
    try:
        value = fn()
    except Exception as exc:
        return "ERROR", f"{type(exc).__name__}: {exc}"
    blank = "NONE" if empty_ok else "EMPTY"
    if value is None:
        return blank, "returned None"
    if isinstance(value, tuple) and len(value) == 2:      # (columns, rows)
        _cols, rows = value
        return ("OK", f"{len(rows)} row(s)") if rows else (blank, "no rows")
    if isinstance(value, (list, dict, str)):
        return ("OK", f"{len(value)} item(s)") if len(value) else (blank, "empty")
    return "OK", type(value).__name__


def run_scene(name: str, cache_dir: Path) -> dict:
    spec = SCENES[name]
    reg = Registry()
    reg.autodiscover()
    for d in spec["dirs"]:
        reg.load_external(d)
    cache = _Cache(cache_dir / f"{name}.jsonl")
    wrap_models(reg, [MODEL], cache)

    cfg = ExperimentConfig(
        scene_generator=name, scene_params=spec["params"], seeds=[0, 1],
        probe=D.compatible_probe(reg, name, reg.list_probes()),
        models=[MODEL], analyzers=ANALYZERS,
        elicit_confidence=True, blind=True,
    )
    scenes, results, reports = run_experiment_collect(cfg, reg)
    print(f"  {name}: {len(results)} rows, {len(reports)} reports "
          f"({cache.hits} cached, {cache.misses} called)", flush=True)

    images = {s.id: (s.images[0] if s.images else None) for s in scenes}
    rep = lambda n: D.report_by_name(reports, n)          # noqa: E731
    checks = {
        "preview: question + ground truth": lambda: [
            reg.get_probe(cfg.probe).question(scenes[0]),
            str(reg.get_probe(cfg.probe).ground_truth(scenes[0]))],
        "preview: image": lambda: [images[scenes[0].id]] if images[scenes[0].id] else [],
        "Run: param sliders": lambda: D.param_specs(reg.get_scene(name)),
        "Run: results table": lambda: (D.RESULT_COLUMNS, D.results_table(results)),
        "Run: accuracy": lambda: D.accuracy_rows(rep("accuracy")),
        "Run: visual gain": lambda: D.visual_gain_rows(rep("visual_gain")),
        "Run: decomposition": lambda: D.decomposition_rows(rep("decomposition")),
        "Run: decomposition confidence": lambda: [
            r[D.decomposition_rows(rep("decomposition"))[0].index("confidence")]
            for r in D.decomposition_rows(rep("decomposition"))[1]],
        "Run: oracle": lambda: D.oracle_rows(rep("oracle")),
        "Run: oracle note": lambda: D.oracle_note(rep("oracle")),
        "Run: taxonomy": lambda: D.taxonomy_rows(rep("taxonomy")),
        "Run: calibration": lambda: D.calibration_rows(rep("calibration")),
        "Sweep: curve": lambda: D.sweep_series(rep("sweep_curve")).get("series"),
        "Sweep: plot": lambda: [A._render_sweep(reports)[0]],
        "Calibration: reliability plot": lambda: [A._render_reliability(reports)[0]],
        "Inspect: cases (full)": lambda: D.failure_cases(results, "full"),
        "Inspect: cases (oracle)": lambda: D.failure_cases(results, "oracle"),
        "Inspect: cases (report)": lambda: D.failure_cases(results, "report"),
        "Inspect: case label": lambda: [
            D.case_label(c) for c in D.failure_cases(results, "full")[:1]],
    }
    # Nothing to inspect means nothing failed, which is a finding rather than a hole.
    # The whole Inspect tab, labels included, is empty when nothing failed. That is a
    # finding about the model, not a hole in the UI.
    return {label: _status(fn, empty_ok=label.startswith("Inspect:"))
            for label, fn in checks.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", action="append", choices=sorted(SCENES), default=None)
    ap.add_argument("--cache", default="runs/feature_check")
    args = ap.parse_args()
    names = args.scene or list(SCENES)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"model {MODEL}, conditions full+blind+oracle+report, confidences on\n")
    table: dict[str, dict] = {}
    for name in names:
        try:
            table[name] = run_scene(name, cache_dir)
        except Exception:
            traceback.print_exc()
            table[name] = {}

    labels = sorted({k for row in table.values() for k in row})
    width = max(len(x) for x in labels) + 2
    print("\n" + " " * width + "".join(f"{n:<14}" for n in names))
    problems = []
    for label in labels:
        line = f"{label:<{width}}"
        for name in names:
            status, detail = table.get(name, {}).get(label, ("ERROR", "not run"))
            line += f"{status:<14}"
            if status not in ("OK", "NONE"):
                problems.append((name, label, status, detail))
        print(line)

    print()
    if not problems:
        print("every feature produced content for every scene "
              "(NONE = nothing failed on that condition, so there is nothing to show).")
        return 0
    print(f"{len(problems)} gap(s):")
    for name, label, status, detail in problems:
        print(f"  {name:<12} {label:<34} {status:<6} {detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
