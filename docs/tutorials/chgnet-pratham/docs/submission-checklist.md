# Submission audit

| Criterion | Evidence |
|---|---|
| Environment (15) | Fresh CUDA setup log; Python/package records; exact portable dependency lock; setup instructions |
| Traceability (15) | Verified source commit; checkpoint 0.3.0; input/test/license/checkpoint hashes; reference source |
| Workflow (25) | Completed CUDA run; OnDemand resource record; runner, Slurm script, submission parameters |
| Evidence (15) | Raw execution/setup logs; predictions; exit code 0; measured runtime/RAM/VRAM; recovered CPU history |
| Validation (20) | Smoke checks; four reference comparisons; three rotations, translation, permutation; negative unit tests |
| Documentation (10) | Setup/run/check/archive/troubleshoot/cleanup guide; run summary |

## Independent reproduction

- [PR #39](https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/pull/39) links this workflow to model-tracking issue #21.
- [Sri Ravela's fresh-checkout H200 run](https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/pull/39#issuecomment-6407596978) passed setup, inference, unit tests, and a second run through Slurm.
- Follow the guide to verify a new run from its own `runs/<run-id>/results.json` and exit status.
