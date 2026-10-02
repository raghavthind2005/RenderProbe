"""Pure logic for the compositional relational scene - NO rendering, NO plugin deps.

This module is the correctness-critical core (grammar, solver, question generation).
It is underscore-prefixed so the registry's auto-discovery skips it; `scenes/relational.py`
(the generator), `probes/relational.py`, and the mock import from here.

Design. Objects live on an integer canvas with
DISTINCT x and DISTINCT y (so positional superlatives are tie-free). A question is a
`Chain`: a uniquely-identifiable anchor object, then `hop_depth` steps of "the <shape>
that is <relation> the current referent", ending in "what color is that object?".

The single guarantee this module provides: `resolve(objects, chain)` re-derives the
referenced object over the true scene graph, returning it only when EVERY step is
unique. `build_chain` constructs chains and then *verifies* them with `resolve` before
returning - so a generated question is correct by construction, and any logic error
surfaces as a build failure (retry) rather than a wrong-but-accepted answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random

COLORS = ["red", "blue", "green", "orange", "purple", "yellow"]
SHAPES = ["circle", "square", "triangle"]
SIZES = ["small", "large"]
EXTREMES = ["leftmost", "rightmost", "topmost", "bottommost"]

# Half-plane relations. `pred(o, ref)` is True when object `o` is <relation> `ref`.
RELATIONS: dict[str, "callable"] = {
    "left of": lambda o, ref: o.x < ref.x,
    "right of": lambda o, ref: o.x > ref.x,
    "above": lambda o, ref: o.y < ref.y,
    "below": lambda o, ref: o.y > ref.y,
}
_REL_NAMES = list(RELATIONS)


@dataclass(frozen=True)
class Obj:
    color: str
    shape: str
    size: str
    x: int
    y: int


# An anchor is ("attr", color, shape) or ("extreme", extreme, shape).
# A hop is (relation, shape, quals) where quals is a subset of {"color","size"}.
@dataclass(frozen=True)
class Chain:
    anchor: tuple
    hops: tuple  # tuple of (relation:str, shape:str, quals:tuple[tuple[str,str],...])
    query: str = "color"


# Resolution (the source of truth)

def _match(objects: list[Obj], shape: str, quals: dict) -> list[Obj]:
    out = []
    for o in objects:
        if o.shape != shape:
            continue
        if any(getattr(o, k) != v for k, v in quals.items()):
            continue
        out.append(o)
    return out


def _resolve_anchor(objects: list[Obj], anchor: tuple) -> Obj | None:
    kind = anchor[0]
    if kind == "attr":
        _, color, shape = anchor
        m = [o for o in objects if o.color == color and o.shape == shape]
        return m[0] if len(m) == 1 else None
    _, extreme, shape = anchor
    m = _match(objects, shape, {})
    if not m:
        return None
    keyfn = {
        "leftmost": lambda o: o.x,
        "rightmost": lambda o: -o.x,
        "topmost": lambda o: o.y,
        "bottommost": lambda o: -o.y,
    }[extreme]
    ordered = sorted(m, key=keyfn)
    if len(ordered) >= 2 and keyfn(ordered[0]) == keyfn(ordered[1]):
        return None  # tie -> not a unique extreme
    return ordered[0]


def resolve(objects: list[Obj], chain: Chain) -> Obj | None:
    """Return the object the chain refers to, or None if any step is non-unique."""
    ref = _resolve_anchor(objects, chain.anchor)
    if ref is None:
        return None
    for rel, shape, quals in chain.hops:
        pred = RELATIONS[rel]
        quals_d = dict(quals)
        matches = [o for o in _match(objects, shape, quals_d) if pred(o, ref)]
        if len(matches) != 1:
            return None
        ref = matches[0]
    return ref


def answer_of(objects: list[Obj], chain: Chain) -> str | None:
    obj = resolve(objects, chain)
    return getattr(obj, chain.query) if obj is not None else None


# Descriptor + chain construction

def unique_descriptor(objects: list[Obj], obj: Obj) -> tuple | None:
    """A minimal anchor description that resolves to exactly `obj`, or None."""
    if len([o for o in objects if o.color == obj.color and o.shape == obj.shape]) == 1:
        return ("attr", obj.color, obj.shape)
    for extreme in EXTREMES:
        anc = ("extreme", extreme, obj.shape)
        if _resolve_anchor(objects, anc) is obj:
            return anc
    return None


def _find_predecessor(objects, current, rng, used, tries=40):
    """Find (P, relation, quals) such that `current` is the UNIQUE shape+quals object
    that is <relation> P. Minimal qualifiers (shape -> +color -> +size)."""
    for _ in range(tries):
        P = rng.choice(objects)
        if P is current or id(P) in used:
            continue
        rel = rng.choice(_REL_NAMES)
        if not RELATIONS[rel](current, P):
            continue  # `current` must actually stand in `rel` to P
        base = [o for o in _match(objects, current.shape, {}) if RELATIONS[rel](o, P)]
        if len(base) == 1 and base[0] is current:
            return P, rel, ()
        withc = [o for o in base if o.color == current.color]
        if len(withc) == 1 and withc[0] is current:
            return P, rel, (("color", current.color),)
        withcs = [o for o in withc if o.size == current.size]
        if len(withcs) == 1 and withcs[0] is current:
            return P, rel, (("color", current.color), ("size", current.size))
    return None


def build_chain(objects, rng: Random, depth: int, target_color: str, tries=200) -> Chain | None:
    """Construct a depth-`depth` chain whose answer is an object of `target_color`.

    Correct by construction: the chain is verified with `resolve` before return, so a
    returned chain always resolves (uniquely, every hop) to the intended target."""
    targets = [o for o in objects if o.color == target_color]
    if not targets:
        return None
    for _ in range(tries):
        T = rng.choice(targets)
        current, used, hops_rev, ok = T, {id(T)}, [], True
        for _ in range(depth):
            found = _find_predecessor(objects, current, rng, used)
            if found is None:
                ok = False
                break
            P, rel, quals = found
            hops_rev.append((rel, current.shape, quals))
            current = P
            used.add(id(current))
        if not ok:
            continue
        anchor = unique_descriptor(objects, current)
        if anchor is None:
            continue
        chain = Chain(anchor=anchor, hops=tuple(reversed(hops_rev)))
        if resolve(objects, chain) is T:   # the safety net
            return chain
    return None


# Natural-language rendering + parsing (inverses; round-trip tested)

def _descriptor_phrase(anchor: tuple) -> str:
    kind = anchor[0]
    if kind == "attr":
        _, color, shape = anchor
        return f"the {color} {shape}"
    _, extreme, shape = anchor
    return f"the {extreme} {shape}"


def _qual_shape_phrase(shape: str, quals: tuple) -> str:
    d = dict(quals)
    parts = ["the"]
    if "size" in d:
        parts.append(d["size"])
    if "color" in d:
        parts.append(d["color"])
    parts.append(shape)
    return " ".join(parts)


def render_question(chain: Chain) -> str:
    sents = [f"Start at {_descriptor_phrase(chain.anchor)}."]
    for rel, shape, quals in chain.hops:
        sents.append(f"Then go to {_qual_shape_phrase(shape, quals)} that is {rel} it.")
    sents.append(f"What {chain.query} is that object? Answer with a single word.")
    return " ".join(sents)


def parse_question(text: str) -> Chain | None:
    """Inverse of `render_question` over the controlled grammar. None if unparseable."""
    try:
        anchor_m = re.search(r"Start at the ([a-z]+) ([a-z]+)\.", text)
        if not anchor_m:
            return None
        w1, w2 = anchor_m.group(1), anchor_m.group(2)
        anchor = ("extreme", w1, w2) if w1 in EXTREMES else ("attr", w1, w2)

        hops = []
        hop_re = r"Then go to the ([a-z ]+?) that is (left of|right of|above|below) it\."
        for hm in re.finditer(hop_re, text):
            words = hm.group(1).split()
            rel = hm.group(2)
            shape = words[-1]
            mods = words[:-1]
            quals = {}
            for m in mods:
                if m in SIZES:
                    quals["size"] = m
                elif m in COLORS:
                    quals["color"] = m
            qtuple = tuple((k, quals[k]) for k in ("color", "size") if k in quals)
            hops.append((rel, shape, qtuple))

        qm = re.search(r"What (\w+) is that object", text)
        query = qm.group(1) if qm else "color"
        return Chain(anchor=anchor, hops=tuple(hops), query=query)
    except Exception:
        return None


# Scene-graph text (oracle) + its parser (inverses; round-trip tested)

def scene_graph_to_text(objects: list[Obj]) -> str:
    lines = [
        "The scene contains these objects "
        "(x is pixels from the left, y is pixels from the top):"
    ]
    for o in sorted(objects, key=lambda o: (o.x, o.y)):
        lines.append(f"- a {o.size} {o.color} {o.shape} at ({o.x}, {o.y})")
    return "\n".join(lines)


def parse_scene_graph_text(text: str) -> list[Obj]:
    objs = []
    for m in re.finditer(
        r"- a (\w+) (\w+) (\w+) at \((\d+),\s*(\d+)\)", text
    ):
        size, color, shape, x, y = m.groups()
        objs.append(Obj(color=color, shape=shape, size=size, x=int(x), y=int(y)))
    return objs
