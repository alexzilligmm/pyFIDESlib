#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REQUIRED_OPS = {
    "scalar_add",
    "scalar_mul",
    "ct_ct_mul",
    "ct_pt_mul",
    "rotate",
    "bootstrap",
}

def _rmse(values: List[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def compute_metrics(reference: List[float], got: List[float]) -> Dict[str, float]:
    if len(reference) != len(got):
        raise ValueError(f"size mismatch: ref={len(reference)} got={len(got)}")
    errs = [g - r for r, g in zip(reference, got)]
    return {
        "max_abs_err": max(abs(e) for e in errs) if errs else 0.0,
        "rmse": _rmse(errs),
    }


def load_fixture(path: Path) -> Dict:
    payload = json.loads(path.read_text())

    if payload.get("schema_version") != "ckks-fixture.v1":
        raise ValueError("unsupported schema_version")

    used_slots = payload.get("used_slots")
    if not isinstance(used_slots, int) or used_slots <= 0:
        raise ValueError("invalid used_slots")

    ops = payload.get("ops")
    if not isinstance(ops, dict):
        raise ValueError("missing ops object")

    missing = REQUIRED_OPS.difference(ops.keys())
    if missing:
        raise ValueError(f"missing required ops: {sorted(missing)}")

    for op_name, op in ops.items():
        if "error" in op and op["error"]:
            continue
        real = op.get("real")
        imag = op.get("imag")
        if not isinstance(real, list) or not isinstance(imag, list):
            raise ValueError(f"{op_name}: missing real/imag arrays")
        if len(real) != used_slots or len(imag) != used_slots:
            raise ValueError(f"{op_name}: expected {used_slots} entries")

    return payload


def summarize_fixture(payload: Dict) -> str:
    used_slots = payload["used_slots"]
    params = payload.get("params", {})
    lines = [
        f"schema={payload.get('schema_version')}",
        f"used_slots={used_slots}",
        f"logN={params.get('logN')}",
        f"logDefaultScale={params.get('logDefaultScale')}",
    ]

    for op_name in sorted(REQUIRED_OPS):
        op = payload["ops"][op_name]
        if op.get("error"):
            lines.append(f"{op_name}: error={op['error']}")
        else:
            lines.append(
                f"{op_name}: level={op.get('level')} log_scale={op.get('log_scale'):.4f}"
            )
    return "\n".join(lines)


def compare_op(op_name: str, payload: Dict, got_real: List[float], max_abs: float, rmse: float) -> Tuple[bool, Dict[str, float]]:
    op = payload["ops"][op_name]
    if op.get("error"):
        raise ValueError(f"operation {op_name} is unavailable in fixture: {op['error']}")

    metrics = compute_metrics(op["real"], got_real)
    ok = metrics["max_abs_err"] <= max_abs and metrics["rmse"] <= rmse
    return ok, metrics


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python cachemir_fixture_loader.py <fixture.json>")
        return 2

    fixture_path = Path(sys.argv[1])
    payload = load_fixture(fixture_path)
    print(summarize_fixture(payload))
    print("fixture validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
