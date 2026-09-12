# Recovery collection metadata snapshot

Captured for the 2026-09-12 pod rebuild: **252 complete, 48 remaining**.
See [DATA_CREATION.md](../../DATA_CREATION.md) for the full handoff.

These files preserve plans, job ledgers, checks, and cleanup provenance in Git.
They contain paths and metadata, **not HDF5 recordings, videos, or checkpoints**.
The active copies under `v4/trajectories/robustness300/recovery300/` are authoritative.
Do not overwrite newer active ledgers with this snapshot.

- `recovery_plan.json`: frozen stronger-dataset `plan.json`, including source hashes.
- `legacy_robustness_plan.json`: the retired parent dataset's `plan.json`; still used for seed exclusions.
- `diverse100_split.json`: original `v4/splits/diverse100.json`.
- `collection_state.json`: all 252 job records plus the 48 remaining slot IDs.
- `recovery_summary.json`: partial category/transition summary.
- `handoff_validation.json`: preservation and video checks, with explicit audit limits.
- `storage_recovery.json`: SHA-256 provenance for 111 recovered HDF5s and discarded temporary files.
- `storage_cleanup.json` / `retirement.json`: superseded mild-data deletion record.
- `focused-tests.txt`: twelve focused tests passed.

A local Git bundle of the handoff commit is saved outside the repository at
`/workspace/robotic-grasp-data-handoff.bundle`. It contains the new commit and
requires the repository history through `1b7d391` (the pre-handoff HEAD).
Keep/download it if the commit has not been pushed and the old workspace will
be discarded. The data directories must be preserved separately.
