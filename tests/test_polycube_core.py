"""Polycube assembly logic: the rotation group, shape identity, and the two
properties that make the question well posed.

No rendering and no plugin deps - this is the correctness-critical layer, so the
assertions are about mathematics rather than about pixels.
"""
from __future__ import annotations

from random import Random

import pytest

from renderprobe.scenes._polycube_core import (
    ROTATIONS,
    _mul,
    assert_unique_fit,
    bounding_box,
    canonical,
    congruent,
    decompose,
    eligible_answers,
    hidden_cells,
    is_connected,
    normalize,
    orient_for_visibility,
    tiles_cube,
    toward_camera,
)


def _det(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


# ---------------------------------------------------------------------------
# The rotation group
# ---------------------------------------------------------------------------

def test_group_has_exactly_24_elements():
    assert len(ROTATIONS) == 24
    assert len(set(ROTATIONS)) == 24


def test_every_element_is_a_proper_rotation():
    # det +1 only. A reflection would make mirror-image pieces compare equal, and a
    # solid piece cannot be mirrored in the real assembly.
    for m in ROTATIONS:
        assert _det(m) == 1


def test_group_is_closed_under_composition():
    s = set(ROTATIONS)
    for a in ROTATIONS:
        for b in ROTATIONS:
            assert _mul(a, b) in s


def test_group_contains_the_identity():
    assert ((1, 0, 0), (0, 1, 0), (0, 0, 1)) in ROTATIONS


# ---------------------------------------------------------------------------
# Shape identity
# ---------------------------------------------------------------------------

_L = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]          # L-tromino, flat
_I = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]          # straight tromino


def test_normalize_is_translation_invariant():
    shifted = [(x + 7, y - 3, z + 11) for x, y, z in _L]
    assert normalize(_L) == normalize(shifted)


def test_canonical_is_invariant_under_every_rotation():
    base = canonical(_L)
    for m in ROTATIONS:
        rotated = [(m[0][0]*x + m[0][1]*y + m[0][2]*z,
                    m[1][0]*x + m[1][1]*y + m[1][2]*z,
                    m[2][0]*x + m[2][1]*y + m[2][2]*z) for x, y, z in _L]
        assert canonical(rotated) == base


def test_different_shapes_are_not_congruent():
    assert not congruent(_L, _I)


def test_same_shape_rotated_is_congruent():
    turned = [(y, -x, z) for x, y, z in _L]
    assert congruent(_L, turned)


def test_cell_count_alone_does_not_decide_shape():
    # Both are 3 cells; a fingerprint that only counted cells would call them equal.
    assert len(_L) == len(_I)
    assert canonical(_L) != canonical(_I)


def test_a_chiral_piece_is_not_congruent_to_its_mirror():
    # The mirror of a chiral solid cannot be reached by rotation, and cannot be
    # reached by physically turning the piece over either - so treating the two as
    # the same shape would make an unsolvable puzzle look solvable.
    screw = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    mirror = [(-x, y, z) for x, y, z in screw]
    assert not congruent(screw, mirror)


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

def test_connected_and_disconnected():
    assert is_connected(_L)
    assert not is_connected([(0, 0, 0), (5, 5, 5)])
    assert not is_connected([])
    # diagonal touch is NOT connectivity: these share only an edge
    assert not is_connected([(0, 0, 0), (1, 1, 0)])


# ---------------------------------------------------------------------------
# Decomposition: the guarantees the question rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(3, 3), (3, 4), (3, 5), (4, 4), (4, 6), (4, 8)])
def test_decomposition_holds_over_many_seeds(n, k):
    for seed in range(12):
        pieces = decompose(n, k, Random(seed))
        assert len(pieces) == k
        assert tiles_cube(pieces, n), f"n={n} k={k} seed={seed}: not an exact tiling"
        for p in pieces:
            assert is_connected(p), f"n={n} k={k} seed={seed}: disconnected piece"
        shapes = [canonical(p) for p in pieces]
        assert len(set(shapes)) == k, f"n={n} k={k} seed={seed}: two pieces congruent"


def test_every_piece_makes_a_uniquely_answerable_question():
    # Whichever piece is removed, exactly one candidate fills the hole.
    for seed in range(8):
        pieces = decompose(3, 4, Random(seed))
        for i in range(len(pieces)):
            assert_unique_fit(pieces, i)


def test_unique_fit_rejects_an_ambiguous_set():
    # Two congruent pieces: the question would have two correct answers.
    a = [(0, 0, 0), (1, 0, 0)]
    b = [(0, 1, 0), (1, 1, 0)]
    with pytest.raises(AssertionError, match="no single correct answer"):
        assert_unique_fit([a, b], 0)


def test_decomposition_is_deterministic_for_a_seed():
    assert decompose(3, 4, Random(5)) == decompose(3, 4, Random(5))


def test_impossible_request_raises_rather_than_returning_junk():
    with pytest.raises(ValueError, match="could not decompose"):
        decompose(2, 7, Random(0), tries=20)   # 8 cells, 7 non-congruent pieces


def test_tiles_cube_catches_overlap_and_gaps():
    assert not tiles_cube([[(0, 0, 0)], [(0, 0, 0)]], 1)       # overlap
    assert not tiles_cube([[(0, 0, 0)]], 2)                    # gap
    assert not tiles_cube([[(0, 0, 0)], [(9, 9, 9)]], 1)       # outside


# ---------------------------------------------------------------------------
# No counting shortcut
# ---------------------------------------------------------------------------

def test_eligible_answers_excludes_uniquely_sized_pieces():
    # sizes 1, 2, 2 -> only the two of size 2 make a question counting cannot settle
    a = [(0, 0, 0)]
    b = [(1, 0, 0), (2, 0, 0)]
    c = [(0, 1, 0), (0, 2, 0)]
    assert eligible_answers([a, b, c]) == [1, 2]


def test_bounding_box_is_rotation_invariant():
    turned = [(y, -x, z) for x, y, z in _L]
    assert bounding_box(_L) == bounding_box(turned)


def test_eligible_answers_also_requires_a_bounding_box_twin():
    # Same cell count, different boxes: comparing boxes alone would settle it, so
    # neither is safe to use as the answer.
    flat = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]          # box 1x1x3
    ell = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]           # box 1x2x2
    assert eligible_answers([flat, ell]) == []
    # add a second piece matching flat's box and flat becomes usable
    flat2 = [(0, 5, 0), (1, 5, 0), (2, 5, 0)]
    assert eligible_answers([flat, ell, flat2]) == [0, 2]


def test_every_eligible_answer_has_both_twins():
    for seed in range(10):
        pieces = decompose(3, 5, Random(seed))
        for i in eligible_answers(pieces):
            twins = [j for j, p in enumerate(pieces)
                     if j != i and len(p) == len(pieces[i])
                     and bounding_box(p) == bounding_box(pieces[i])]
            assert twins, f"piece {i} is separable by count or box alone"


def test_every_eligible_answer_has_a_size_twin():
    for seed in range(10):
        pieces = decompose(3, 5, Random(seed))
        elig = eligible_answers(pieces)
        assert elig, "a decomposition with no safe answer should have been rejected"
        for i in elig:
            twins = [j for j, p in enumerate(pieces)
                     if j != i and len(p) == len(pieces[i])]
            assert twins, f"piece {i} could be identified by cell count alone"


def test_decompose_rejects_partitions_with_no_safe_answer():
    # require_twin off, then on: the guarantee is the difference between them
    loose = decompose(3, 5, Random(11), require_twin=False)
    assert len(loose) == 5
    strict = decompose(3, 5, Random(11), require_twin=True)
    assert eligible_answers(strict)


# ---------------------------------------------------------------------------
# Visibility: is the question answerable from the image at all?
# ---------------------------------------------------------------------------

_VIEW = toward_camera(32.0, 17.0)


def test_toward_camera_is_a_unit_vector_pointing_up_and_out():
    v = toward_camera(32.0, 17.0)
    assert abs(sum(c * c for c in v) ** 0.5 - 1.0) < 1e-9
    assert v[1] > 0          # camera is above
    assert v[2] < 0          # grid z runs INTO the frame, camera is out of it


def test_a_single_cell_is_never_hidden():
    assert hidden_cells([(0, 0, 0)], _VIEW) == []


def test_two_diagonally_offset_cells_do_not_occlude():
    # One neighbor on the view diagonal blocks the center ray but not the whole cell,
    # and the id pass agrees the far cube is still drawn - so neither is hidden.
    far = (0, 0, 0)
    near = (round(_VIEW[0]), round(_VIEW[1]), round(_VIEW[2]))
    assert hidden_cells([far, near], _VIEW) == []


def test_an_enclosed_cell_is_hidden():
    # the center of a solid 3x3x3 block cannot be seen from anywhere
    block = [(x, y, z) for x in range(3) for y in range(3) for z in range(3)]
    hidden = hidden_cells(block, _VIEW)
    assert (1, 1, 1) in hidden


def test_a_cell_squarely_behind_a_wall_is_hidden():
    # a cell plus a full wall of cells one step toward the camera covering every sample
    step = (round(_VIEW[0]), round(_VIEW[1]), round(_VIEW[2]))
    target = (0, 0, 0)
    wall = [(target[0] + step[0] + dx, target[1] + step[1] + dy, target[2] + step[2] + dz)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    assert target in hidden_cells([target, *wall], _VIEW)


def test_a_flat_plate_facing_the_camera_is_fully_visible():
    plate = [(x, y, 0) for x in range(3) for y in range(3)]
    assert hidden_cells(plate, _VIEW) == []


def test_every_piece_can_be_oriented_fully_visible():
    # This is what makes the question answerable, so it is asserted, not assumed.
    for seed in range(10):
        for piece in decompose(3, 5, Random(8000 + seed)):
            oriented = orient_for_visibility(piece, _VIEW, Random(seed))
            assert oriented is not None, "no orientation shows every cube"
            assert hidden_cells(oriented, _VIEW) == []
            assert congruent(oriented, piece), "orienting must not change the shape"
