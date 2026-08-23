# Reproducibility Records

The release uses three machine-readable registries:

- `experiments.json` pins the upstream commit, public entry point, command, and
  status of all four order policies for each of the nine hosts.
- `release_artifacts.json` records the SHA-256 digest and byte size of the
  selected checkpoint/controller and raw evaluation records for each host.
- `hf_checkpoint_policy.json` names the single model checkpoint or controller
  retained for each host in the companion Hugging Face release.

Run the public consistency checks from the repository root:

```bash
python scripts/verify_release.py
python scripts/verify_artifact_manifest.py --artifact-root "$DPRM_ARTIFACT_ROOT"
python scripts/audit_artifact_semantics.py --artifact-root "$DPRM_ARTIFACT_ROOT"
```

The paper configuration is the configuration declared in each integration
README and in `experiments.json`. Development sweeps and held-out confirmation
sets are disjoint where a controller or scalarization preference is selected.
Canonical paper rows are written only after the corresponding raw-record and
artifact checks pass.
