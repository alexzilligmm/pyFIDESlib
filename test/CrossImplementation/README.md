# Cross-Implementation Differential Scaffold

This directory contains the Phase-1 scaffold for validating pyFIDESlib GPU outputs against Cachemir reference fixtures.

## Current Files

- `cachemir_fixture_loader.py`: Loads and validates fixture schema, computes error metrics.
- `cachemir_packing.py`: Cachemir-oriented slot/rotation packing adapter utilities used by differential checks.

`_pyfideslib` now also exposes a native helper:

```python
import _pyfideslib as fd
eng = fd.CachemirPackingEngine(max_slots=2048, hid_dim=256, exp_dim=1024, num_heads=32, seq_len=512)
plan = eng.compute_plan(rotation_step=5, auto_sign_check=True)
indices = eng.build_rotation_indices(rotation_step=5, auto_sign_check=True, include_stride=True)
```

## Intended Workflow

1. Generate fixture from Cachemir:

```bash
cd cachemir
go run ./cmd/exportfixtures -output=../pyFIDESlib/test/CrossImplementation/fixtures_ckks_v1.json -fixtureSlots=128 -level=5
```

2. Validate fixture structure:

```bash
cd pyFIDESlib/test/CrossImplementation
python3 cachemir_fixture_loader.py fixtures_ckks_v1.json
```

3. Run pyFIDESlib differential checks (GPU path by default):

```bash
python3 run_pyfideslib_differential.py fixtures_ckks_v1.json
```

Optional flags:

```bash
python3 run_pyfideslib_differential.py fixtures_ckks_v1.json --max-abs 1e-4 --rmse 1e-5
python3 run_pyfideslib_differential.py fixtures_ckks_v1.json --level-tol 0 --log-scale-tol 0.5
python3 run_pyfideslib_differential.py fixtures_ckks_v1.json --strict-metadata
python3 run_pyfideslib_differential.py fixtures_ckks_v1.json --no-auto-rotate-sign-check
python3 run_pyfideslib_differential.py fixtures_ckks_v1.json --no-matmul-kernel-checks
python3 run_pyfideslib_differential.py fixtures_ckks_v1.json --cpu
python3 run_pyfideslib_differential.py fixtures_ckks_v1.json --with-bootstrap
```

4. Integrate this loader into pyFIDESlib tests to compare decrypted GPU outputs against fixture op outputs.

## SLURM Workflow

From repository root, submit build and test separately:

```bash
sbatch build_bindings.slurm
sbatch test_differential.slurm
```

Or submit test automatically after successful build:

```bash
bash submit_build_then_test.sh
```

Monitor running jobs without hanging after completion:

```bash
bash monitor_slurm_jobs.sh <build_job_id>,<test_job_id>
```

Override differential tolerances at submission time:

```bash
sbatch --export=ALL,DIFF_MAX_ABS=1e-4,DIFF_RMSE=1e-5,DIFF_LEVEL_TOL=0,DIFF_LOG_SCALE_TOL=0.5 test_differential.slurm
```

## Notes

- This scaffold validates fixture integrity and runs direct pyFIDESlib operation checks against fixture references.
- The differential runner now derives slot padding and rotation candidate conventions through `cachemir_packing.py`.
- `cachemir_packing.py` now uses native `CachemirPackingEngine`/typed objects as the primary contract and falls back to pure Python only when native APIs are unavailable.
- The runner also performs optional 1x1 `pcmm_gpu` and `ccmm_gpu` kernel parity checks when fixture references (`pcmm_1x1`, `ccmm_1x1`) are present.
