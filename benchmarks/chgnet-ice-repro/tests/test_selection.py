import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_subset import choose_frame, rank


def test_selection_deterministic_and_seeded():
    assert rank(42, "mp-1") == rank(42, "mp-1")
    assert rank(43, "mp-1") != rank(42, "mp-1")
    assert rank(42, "mp-2") != rank(42, "mp-1")


def test_choose_frame_rejects_missing_labels():
    valid = {"structure": {"sites": [{}, {}]},
             "energy_per_atom": -2.0, "force": [[0,0,0], [0,0,0]],
             "stress": [[0,0,0], [0,0,0], [0,0,0]], "magmom": None}
    frames = {"bad": {"structure": {"sites": [{}]}}, "good": valid}
    assert choose_frame(frames, 2, 16) == ("good", valid)
