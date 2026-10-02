"""Is a scene measurable at all? Ask six calls, not a hundred and forty four.

The three-way decomposition needs `acc_full` off the floor and off the ceiling. At the
floor there is nothing to decompose: recovery is a lift measured from guessing, and the
bottleneck cascade reads thresholds on numbers that mean nothing. At the ceiling there
is nothing to explain. Either way a full run buys no information, and on a true-3D scene
a full run is hundreds of model calls and several minutes of path tracing.

This runs the `full` condition only, on one model, over a handful of scenes, and says
which of the three a scene is in. It is a development gate, not part of the instrument:
nothing here produces a result you would report.

    python tools/calibrate.py polycube --model gemma-3-27b-or --scenes 6
    python tools/calibrate.py polycube --param n_pieces=4 --chance 0.25
    python tools/calibrate.py pathfinding --model llama-3.2-11b-vision-nv --scenes 8

Reads API keys from the environment, so source your .env first:

    set -a && . ./.env && set +a
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from renderprobe.core.registry import Registry  # noqa: E402
from renderprobe.core.runner import _run_condition  # noqa: E402

# A scene is worth a full run when accuracy sits between these. Below the lower bound
# the decomposition is reading noise; above the upper one the model has simply solved it
# and there is no failure to localize. The band is deliberately generous: calibration
# only has to rule out the two useless ends.
_FLOOR = 0.20
_CEILING = 0.80
# Largest P(accuracy >= observed | pure guessing) still counted as clearing chance. Not
# a significance test being reported as a finding, just a gate: at this sample size a
# scene that cannot beat this is not one to spend a run on.
_MAX_P_CHANCE = 0.10


def poisson_binomial_sf(ps: list[float], k: int) -> float:
    """P(X >= k) where X sums independent Bernoulli trials with different p.

    Chance is per scene, not global: a 4-candidate scene and a 6-candidate one in the
    same sample have different floors, and averaging them would misstate both.
    """
    dist = [1.0]
    for p in ps:
        nxt = [0.0] * (len(dist) + 1)
        for i, v in enumerate(dist):
            nxt[i] += v * (1.0 - p)
            nxt[i + 1] += v * p
        dist = nxt
    return sum(dist[k:])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Holds up at the small n and extreme proportions this tool
    lives at, where the textbook normal interval runs past 0 and 1."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def chance_for(scene, probe, override: float | None) -> float | None:
    """Per-scene probability of being right by guessing.

    An explicit --chance wins. Otherwise the probe is asked, via an optional
    `chance_level(scene)`: only the probe knows how many answers its question admits,
    and for a scene whose answer set varies with a parameter it is the only honest
    source. Absent both, the caller is told rather than given a made-up number.
    """
    if override is not None:
        return override
    fn = getattr(probe, "chance_level", None)
    if callable(fn):
        try:
            value = fn(scene)
            if value:
                return float(value)
        except Exception:
            return None
    return None


def schema_defaults(gen) -> dict:
    """Every parameter the scene declares a default for."""
    out = {}
    for name, spec in (getattr(gen, "params_schema", None) or {}).items():
        if isinstance(spec, dict) and "default" in spec:
            out[name] = spec["default"]
    return out


def parse_params(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            out[key] = int(raw)
        except ValueError:
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scene")
    ap.add_argument("--model", default="gemma-3-27b-or")
    ap.add_argument("--probe", default=None, help="default: the scene's paired probe")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--param", action="append", default=[], metavar="K=V")
    ap.add_argument("--chance", type=float, default=None,
                    help="per-scene guessing rate, if the probe cannot state it")
    ap.add_argument("--plugin-dir", action="append", default=[], metavar="PATH",
                    help="load out-of-tree plugins from this file or directory "
                         "(repeatable), so a scene can be calibrated before it moves "
                         "into a package")
    ap.add_argument("--spp", type=int, default=None,
                    help="samples per pixel for a 3D scene. Match whatever the real run "
                         "will use: a noisier image is a different perceptual task, and "
                         "calibrating on one then running the other measures neither")
    args = ap.parse_args()

    reg = Registry()
    reg.autodiscover()
    for path in args.plugin_dir:
        print(f"loaded   {', '.join(reg.load_external(path))}  from {path}")
    gen = reg.get_scene(args.scene)
    probe_name = args.probe or _paired_probe(reg, args.scene)
    probe = reg.get_probe(probe_name)
    model = reg.get_model(args.model)
    # Start from the scene's own declared defaults so a scene whose generate() requires
    # every parameter still runs with no --param at all, then let --param override.
    params = schema_defaults(gen)
    params.update(parse_params(args.param))
    if args.spp is not None:
        params["renderer_opts"] = {"spp": args.spp}

    print(f"calibrating {args.scene} on {args.model}  (probe: {probe_name})")
    print(f"  {args.scenes} scene(s), full condition only, params {params or '{}'}\n")

    chances: list[float] = []
    rows = []
    unknown_chance = False
    for seed in range(args.scenes):
        scene = gen.generate(dict(params), seed)
        question = probe.question(scene)
        gt = probe.ground_truth(scene)
        r = _run_condition(scene, probe, model, "full", question, gt)
        c = chance_for(scene, probe, args.chance)
        if c is None:
            unknown_chance = True
        else:
            chances.append(c)
        rows.append((scene.id, c, r))
        mark = "OK" if r.correct else ("ERR" if r.error_flag else "x")
        shown = "  n/a" if c is None else f"{c:.2f}"
        print(f"  {scene.id:26} chance={shown:>5}  "
              f"gt={str(gt)[:16]:16} pred={str(r.pred)[:16]:16} {mark}"
              f"{'  ' + r.error_flag if r.error_flag else ''}")

    valid = [r for _, _, r in rows if r.error_flag is None]
    k = sum(bool(r.correct) for r in valid)
    n = len(valid)
    dropped = len(rows) - n
    print()
    if not n:
        print("  every call errored - nothing to calibrate. Check keys and model name.")
        return 2

    acc = k / n
    lo, hi = wilson(k, n)
    print(f"  accuracy            {k}/{n} = {acc:.2f}   95% CI [{lo}, {hi}]")
    if dropped:
        print(f"  dropped             {dropped} row(s) errored or were truncated")

    if unknown_chance or len(chances) != n:
        print("  chance              unknown - the probe exposes no chance_level(scene) "
              "and no --chance was given")
        print("\n  VERDICT: UNKNOWN. Accuracy alone cannot say whether this beats "
              "guessing.\n  Pass --chance, or give the probe a chance_level(scene).")
        return 1

    expected = sum(chances)
    p_chance = poisson_binomial_sf(chances, k)
    print(f"  chance              {expected:.1f}/{n} = {expected / n:.2f}   "
          "(per scene, summed)")
    print(f"  P(>= observed | guessing)   {p_chance:.3f}")

    print()
    if p_chance > _MAX_P_CHANCE or acc < _FLOOR:
        print("  VERDICT: AT FLOOR. Guessing explains this result.")
        print("  A full run would measure nothing: recovery becomes a lift from noise "
              "and the\n  bottleneck cascade reads thresholds on numbers that do not "
              "mean anything. Make\n  the task easier, then calibrate again.")
        return 1
    if acc > _CEILING:
        print("  VERDICT: AT CEILING. The model solves this.")
        print("  There is no failure to localize. Make the task harder, then calibrate "
              "again.")
        return 1
    print(f"  VERDICT: MEASURABLE. Accuracy is clear of chance and inside "
          f"[{_FLOOR}, {_CEILING}].")
    print("  Worth a full run: the decomposition has something to decompose.")
    return 0


def _paired_probe(reg: Registry, scene_name: str) -> str:
    from renderprobe.ui import data as D
    name = D.compatible_probe(reg, scene_name, reg.list_probes())
    if not name:
        raise SystemExit(f"no probe pairs with scene {scene_name!r}; pass --probe")
    return name


if __name__ == "__main__":
    raise SystemExit(main())
