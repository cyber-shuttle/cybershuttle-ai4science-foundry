# Provenance

The original source revision, checkpoint identity, reference outputs, and measured CPU baseline were recorded in this task on September 27, 2026. The model-selection issue is #21. The new workflow keeps that source revision instead of silently selecting a newer release.

The full original remote environment lock and JSON/log files remain to be retrieved from `/home/hice1/ppatel686/chgnet-repro`. Local historical evidence is a recorded summary and screenshot, not a complete archival record. The summary's statement that the GitHub issue awaits approval is stale: issue #21 was subsequently posted.

New runs generate SHA-256 hashes for the input CIF, source test, upstream license, and checkpoint, plus script/reference hashes. They record actual Python/packages, GPU, Slurm identifiers, and tolerances. This manifest enables artifact-integrity checks on subsequent runs; hashes do not independently certify scientific correctness.

The main model's energy/force calculation uses automatic differentiation. Do not wrap force/stress inference in `torch.no_grad()` or `torch.inference_mode()`. `model.eval()` is used to select evaluation behavior.

Source identity correction (October 11, 2026): the September summary transcribed the prefix as `b9a6c158603...`; `git ls-remote` confirms `b9a6c158602...`. The workflow uses the verified ID. Recover the old raw results.json to independently check that run’s recorded ID.
