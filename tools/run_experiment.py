"""Run an experiment so that interrupting it costs nothing.

Two caches, because one is not enough.

The REPLY cache keys each answer by the model, the prompt and the exact image bytes.
Hashing the pixels rather than a scene name is what stops a reply being replayed against
a picture that has since changed.

That identity rule has a consequence, and it used to break resumption on exactly the
runs worth resuming. `mitsuba_3d` is a stochastic path tracer, so the same scene at the
same seed MAY come out with different pixels. Not always: measured over 10 pairs of
renders at one seed, 3 pairs differed and the renders landed on 6 distinct images. That
is more than enough to break a cache keyed on pixels, and worse than outright
non-determinism would be, because a resumed run replays some calls and re-pays for
others with no pattern to it. Measured on a three-seed polycube run: of 12 calls, 3
replayed - the text-only oracle ones, which carry no pixels - and the 9 carrying an
image were paid for a second time.

The SCENE cache closes that. A generated scene is stored on first render and replayed
byte for byte afterwards, so a resumed run asks about the identical picture, the reply
keys match, and the 3D re-render (~6 s per polycube scene) is skipped as well. The
ground truth was always deterministic; it is only the pixels that needed pinning.

A scene is replayed only when the generator, its parameters, the seed AND the source of
the generator's own module are unchanged. Editing the scene invalidates it. Editing
something the scene imports, or a renderer it calls, does NOT: pass --fresh to discard
both caches when you have changed anything further out.

    python tools/run_experiment.py configs/route.yaml
    nohup caffeinate -i python tools/run_experiment.py configs/route.yaml > run.log 2>&1 &

`caffeinate -i` stops the machine idling to sleep while it works. It does NOT survive the
lid being closed: macOS sleeps on lid close unless `sudo pmset -a disablesleep 1` is set,
which is a system-wide change to make deliberately and undo afterwards with
`sudo pmset -a disablesleep 0`.

Results and reports land next to the cache. Nothing here is part of the instrument: it is
the same run_experiment_collect the CLI calls, with a cache wrapped around the adapters.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from renderprobe.cli import _config_from_raw  # noqa: E402
from renderprobe.core.registry import Registry  # noqa: E402
from renderprobe.core.runner import run_experiment_collect  # noqa: E402
from renderprobe.core.schema import Response  # noqa: E402


def _key(model_name: str, images, prompt: str) -> str:
    """Identity of a call: the model, the prompt, and the pixels.

    The images are hashed rather than named, so a scene whose rendering changed produces
    a different key and is re-asked instead of replaying a reply to a picture that no
    longer exists.
    """
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(b"\0")
    h.update(prompt.encode())
    for img in images or []:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        h.update(buf.getvalue())
    return h.hexdigest()


class _Cache:
    """Append-only JSONL. Written and flushed per reply, so whatever the run got to
    survives however it stopped."""

    def __init__(self, path: Path):
        self.path = path
        self.hits = 0
        self.misses = 0
        self.entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    try:
                        row = json.loads(line)
                        self.entries[row["key"]] = row
                    except json.JSONDecodeError:
                        continue          # a half-written last line, from a hard kill
        self.fh = path.open("a")

    def get(self, key: str):
        row = self.entries.get(key)
        if row is None:
            return None
        self.hits += 1
        return Response(text=row["text"], confidence=row.get("confidence"),
                        truncated=bool(row.get("truncated")))

    def put(self, key: str, model: str, prompt: str, resp: Response) -> None:
        self.misses += 1
        row = {"key": key, "model": model, "prompt": prompt[:400],
               "text": resp.text, "confidence": resp.confidence,
               "truncated": bool(getattr(resp, "truncated", False)),
               "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.entries[key] = row
        self.fh.write(json.dumps(row) + "\n")
        self.fh.flush()


def _source_fingerprint(gen) -> str:
    """Hash of the file the generator class is defined in.

    A bounded guarantee, and worth stating: it catches an edit to the scene itself,
    which is the common case, and it does not catch an edit to a module the scene
    imports or to the renderer it calls. `--fresh` is the blunt instrument for those.
    """
    try:
        path = inspect.getsourcefile(type(gen))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16] if path else "?"
    except Exception:
        return "?"


def _scene_key(gen_name: str, params: dict, seed: int, fingerprint: str) -> str:
    """Identity of a generated scene: what it is of, at what settings, from what code."""
    payload = json.dumps(
        {"gen": gen_name, "params": params, "seed": seed, "src": fingerprint},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class _SceneCache:
    """One pickle per scene, so a resumed run re-renders nothing and re-asks nothing.

    Pickle rather than PNG-plus-JSON because a Scene carries its SceneGraph too, and a
    scene replayed without its graph would validate differently from the one that was
    probed. A stale or unreadable entry degrades to a miss rather than an error: the
    cost of being wrong here is one re-render.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        f = self.path / f"{key}.pkl"
        if not f.exists():
            return None
        try:
            scene = pickle.loads(f.read_bytes())
        except Exception:
            return None                  # schema moved under it, or a truncated write
        self.hits += 1
        return scene

    def put(self, key: str, scene) -> None:
        self.misses += 1
        tmp = self.path / f"{key}.pkl.part"
        try:
            tmp.write_bytes(pickle.dumps(scene))
            tmp.replace(self.path / f"{key}.pkl")   # atomic: never a half-written scene
        except Exception:
            tmp.unlink(missing_ok=True)             # uncacheable scene still runs


def wrap_scenes(reg: Registry, name: str, cache: _SceneCache) -> None:
    """Put the scene cache in front of the generator's generate(), leaving it untouched."""
    gen = reg.get_scene(name)
    original = gen.generate
    fingerprint = _source_fingerprint(gen)

    def cached(params, seed, _orig=original, _name=name, _fp=fingerprint):
        key = _scene_key(_name, params, seed, _fp)
        hit = cache.get(key)
        if hit is not None:
            return hit
        scene = _orig(params, seed)
        cache.put(key, scene)
        return scene

    gen.generate = cached


def wrap_models(reg: Registry, names, cache: _Cache) -> None:
    """Put the cache in front of each adapter's run(), leaving the adapter untouched."""
    for name in names:
        model = reg.get_model(name)
        original = model.run

        def cached(images, prompt, _model=model, _run=original, _name=name):
            key = _key(_name, images, prompt)
            hit = cache.get(key)
            if hit is not None:
                return hit
            resp = _run(images, prompt)
            cache.put(key, _name, prompt, resp)
            return resp

        model.run = cached


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--out", default="runs", help="directory for cache and results")
    ap.add_argument("--fresh", action="store_true",
                    help="discard both caches first: re-render every scene and re-ask "
                         "every call (use after changing a renderer or anything a scene "
                         "imports, which the scene cache does not track)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = _config_from_raw(raw)
    out = Path(args.out) / cfg_path.stem
    out.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        shutil.rmtree(out / "scenes", ignore_errors=True)
        (out / "responses.jsonl").unlink(missing_ok=True)

    reg = Registry()
    reg.autodiscover()
    # Out-of-tree plugins named by the config. The CLI does this from `raw`; without it
    # a config with `plugin_dirs:` (gear_train's, for one) cannot find its own scene.
    for d in raw.get("plugin_dirs", []) or []:
        reg.load_external(d)
    scene_cache = _SceneCache(out / "scenes")
    wrap_scenes(reg, cfg.scene_generator, scene_cache)
    cache = _Cache(out / "responses.jsonl")
    wrap_models(reg, cfg.models, cache)

    stored = len(list((out / "scenes").glob("*.pkl")))
    print(f"config   {cfg_path}")
    print(f"scenes   {out / 'scenes'}  ({stored} scene(s) already rendered)")
    print(f"replies  {cache.path}  ({len(cache.entries)} reply(ies) already stored)")
    print(f"models   {', '.join(cfg.models)}", flush=True)

    started = time.time()
    scenes, results, reports = run_experiment_collect(cfg, reg)

    with (out / "results.pkl").open("wb") as f:
        pickle.dump([r.__dict__ for r in results], f)
    with (out / "reports.json").open("w") as f:
        json.dump({r.name: r.payload for r in reports}, f, indent=2, default=str)

    print(f"\n{len(scenes)} scene(s), {len(results)} result(s) in "
          f"{time.time() - started:.0f}s")
    print(f"  scenes replayed      {scene_cache.hits}")
    print(f"  scenes rendered      {scene_cache.misses}")
    print(f"  replies replayed     {cache.hits}")
    print(f"  newly called         {cache.misses}")
    print(f"  written to           {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
