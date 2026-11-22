"""Convert verified binned counts into explicitly approximate POYO input events."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

from inspect_achilles import DATASET_ID, SOURCE, load_achilles


def prepare(data_root, timing_file, output, accept_bin_centers):
    _, neural, behavior = load_achilles(data_root)
    if not accept_bin_centers:
        raise ValueError("Explicitly accept bin-center event approximation; exact spike times are unavailable.")
    if np.any(neural < 0) or np.any(neural > 20_000_000) or not np.allclose(neural, np.rint(neural), rtol=0, atol=1e-6):
        raise ValueError("Neural values must be verified nonnegative integer counts; rates cannot be expanded.")
    timing = json.loads(timing_file.read_text())
    if timing.get("representation") != "spike_counts":
        raise ValueError("Timing evidence must confirm spike_counts representation.")
    # Require a public provenance reference, not an undocumented sampling-rate guess.
    for key in ("timing_source", "counts_source", "unit_mapping_source", "behavior_alignment_source"):
        if not str(timing.get(key, "")).startswith("https://"):
            raise ValueError(f"Supply public source evidence in {key}.")
    n = len(neural)
    starts = np.asarray(timing["bin_start_seconds"], dtype=np.float64)
    ends = np.asarray(timing["bin_end_seconds"], dtype=np.float64)
    target_times = np.asarray(timing["behavior_timestamps_seconds"], dtype=np.float64)
    for array in (starts, ends, target_times):
        if array.shape != (n,) or not np.isfinite(array).all():
            raise ValueError("Timing arrays must contain one finite coordinate per original row.")
    if starts[0] < 0 or np.any(ends <= starts) or np.any(starts[1:] < ends[:-1] - 1e-9):
        raise ValueError("Bins must be ordered, positive-duration, nonoverlapping, and nonnegative.")
    if np.any(np.diff(target_times) <= 0) or np.any(target_times < starts) or np.any(target_times >= ends):
        raise ValueError("Behavior timestamps must increase strictly and lie in their corresponding bins.")
    # This adapter targets the source-documented 25 ms counts; other representations need review.
    if not np.allclose(ends - starts, 0.025, rtol=0, atol=1e-9):
        raise ValueError("Bin widths disagree with the CEBRA loader's documented 25 ms count bins.")
    duration = float(timing["recording_end_seconds"])
    if not np.isfinite(duration) or not np.isclose(duration, ends[-1], rtol=0, atol=1e-9):
        raise ValueError("Recording end must match the final bin end.")
    # Remove only a documented common origin; preserve all real gaps.
    origin = starts[0]
    starts, ends, target_times = starts - origin, ends - origin, target_times - origin
    duration -= origin
    counts = np.rint(neural).astype(np.int64)
    event_count = sum(int(x) for x in counts.ravel())
    if event_count <= 0 or event_count > 20_000_000:
        raise ValueError("Empty or unexpectedly large event stream; inspect counts before allocation.")
    rows, units = np.nonzero(counts)
    multiplicity = counts[rows, units]
    timestamps = np.repeat((starts[rows] + ends[rows]) / 2, multiplicity)
    unit_index = np.repeat(units, multiplicity).astype(np.int64)
    if np.any(np.diff(timestamps) < 0) or np.any(timestamps < 0) or np.any(timestamps >= duration):
        raise ValueError("Invalid event ordering or domain.")
    if len(timestamps) != event_count or unit_index.min() < 0 or unit_index.max() >= 120:
        raise ValueError("Invalid event multiplicity or unit indices.")
    # Split by original row order (70/15/15), and split sampling domains at every gap.
    boundaries = (0, int(n * 0.70), int(n * 0.85), n)
    split = np.empty(n, dtype=np.int64)
    domains = {}
    for label, (name, left, right) in enumerate(zip(("train", "valid", "test"), boundaries, boundaries[1:])):
        split[left:right] = label
        cuts = np.flatnonzero(~np.isclose(starts[left + 1:right], ends[left:right - 1], rtol=0, atol=1e-9)) + left + 1
        first = np.r_[left, cuts]
        last = np.r_[cuts - 1, right - 1]
        domains[name + "_domain"] = Interval(start=starts[first], end=ends[last])
        if not np.any(ends[last] - starts[first] >= 1.0):
            raise ValueError(f"No {name} segment fits the official one-second context window.")
        if name == "train" and np.floor((ends[last] - starts[first]) / 1.0).sum() < 8:
            raise ValueError("Insufficient continuous training windows for the configured batch of eight.")
    domain = Interval(0.0, duration)
    data = Data(
        domain=domain,
        session=Data(id="achilles"),
        units=ArrayDict(id=np.array([f"achilles/unit_{i:03d}" for i in range(120)])),
        spikes=IrregularTimeSeries(timestamps=timestamps, unit_index=unit_index, domain=domain),
        behavior=IrregularTimeSeries(timestamps=target_times, values=behavior.astype(np.float32),
                                     row_index=np.arange(n), split=split, domain=domain),
        **domains,
    )
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(output / "achilles.h5", "w") as handle:
        data.to_hdf5(handle)
    manifest = {
        "dataset": DATASET_ID, "loader_source": SOURCE,
        "neural_shape": list(neural.shape), "behavior_shape": list(behavior.shape),
        "event_count": event_count, "duration_seconds": duration,
        "event_representation": "Repeated bin-center tokens; NOT recovered biological spike times",
        "unit_identity": "Stable input column IDs; source mapping requires external evidence",
        "target_columns": ["position", "right", "left"],
        "split_policy": "chronological 70/15/15 by original rows; gaps retained",
        "timing_evidence": timing,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-bin-center-approximation", action="store_true")
    args = parser.parse_args()
    prepare(args.data_root, args.timing, args.output, args.accept_bin_center_approximation)


if __name__ == "__main__":
    main()
