#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from cachemir_fixture_loader import load_fixture
from cachemir_packing import from_fixture, padded_input_slots

try:
    import _pyfideslib as fd
except ImportError as exc:
    raise SystemExit(f"failed to import _pyfideslib: {exc}")


def _configure_context(payload, force_cpu=False):
    params = payload["params"]
    config = from_fixture(payload)

    ccparams = fd.CCParams()
    ccparams.set_ring_dim(1 << int(params.get("logN", 12)))
    ccparams.set_batch_size(config.max_slots)
    ccparams.set_multiplicative_depth(12)
    ccparams.set_scaling_mod_size(int(params.get("logDefaultScale", 40)))
    ccparams.set_first_mod_size(52)
    ccparams.set_scaling_technique(fd.FIXEDAUTO)
    ccparams.set_security_level(fd.HEStd_NotSet)
    if force_cpu:
        ccparams.set_devices([])
    else:
        ccparams.set_devices([0])

    cc = fd.gen_crypto_context(ccparams)
    cc.enable(fd.PKE)
    cc.enable(fd.KEYSWITCH)
    cc.enable(fd.LEVELEDSHE)
    cc.enable(fd.ADVANCEDSHE)
    return cc


def _log(event, **kwargs):
    msg = {"event": event, **kwargs}
    print(json.dumps(msg, sort_keys=True), flush=True)


def run(payload, row_size=2, bstep=1, force_cpu=False):
    params = payload["params"]
    config = from_fixture(payload)

    cc = _configure_context(payload, force_cpu=force_cpu)
    keys = cc.key_gen()
    cc.eval_mult_key_gen(keys.secret_key)

    rot_steps = sorted(set(int(i) for i in fd.generate_matmul_rotation_indices_gpu(row_size, bstep)))
    _log("matmul_rotation_indices", row_size=row_size, bstep=bstep, count=len(rot_steps), indices=rot_steps)
    if rot_steps:
        cc.eval_rotate_key_gen(keys.secret_key, rot_steps)
    cc.load_context(keys.public_key)

    input_vec = padded_input_slots(payload["input"], config.max_slots)
    pt = cc.make_ckks_packed_plaintext(input_vec)
    ct = cc.encrypt(keys.public_key, pt)
    _log("ciphertext_created", ct_level=int(ct.get_level()), ct_noise_scale_deg=int(ct.get_noise_scale_deg()))

    configured_depth = 12
    precomp_level = max(1, min(configured_depth, int(ct.get_level())))
    _log("precomp_level", requested_level=precomp_level)

    precomp = fd.get_matmul_precomp_single_level_gpu(
        cc,
        row_size=row_size,
        bstep=bstep,
        level=precomp_level,
        fuse_boot_prescale=False,
        slots=config.max_slots,
    )
    _log("precomp_created", precomp_row_size=int(precomp.row_size), precomp_bstep=int(precomp.bstep))

    ct_mat = fd.CiphertextMatrixGPU(cc, [[ct]])
    _log("ciphertext_matrix_created", rows=1, cols=1)

    _log("ccmm_start")
    ccmm_out = fd.ccmm_gpu(cc, ct_mat, ct_mat, row_size, precomp)
    ccmm_ct = ccmm_out.to_list(cc)[0][0]
    _log(
        "ccmm_done",
        out_ct_level=int(ccmm_ct.get_level()),
        out_ct_noise_scale_deg=int(ccmm_ct.get_noise_scale_deg()),
    )

    # Attempt a decrypt to force full sync and make failures deterministic in logs.
    pt_out = cc.decrypt(keys.secret_key, ccmm_ct)
    out_vals = pt_out.get_real_packed_value()[: config.used_slots]
    _log("decrypt_done", used_slots=int(config.used_slots), sample=out_vals[:8])


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal matmul backend repro with level/key logging")
    parser.add_argument("fixture", type=Path, help="path to Cachemir JSON fixture")
    parser.add_argument("--row-size", type=int, default=2)
    parser.add_argument("--bstep", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    payload = load_fixture(args.fixture)
    run(payload, row_size=args.row_size, bstep=args.bstep, force_cpu=args.cpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
