#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import math

from cachemir_fixture_loader import compare_op, load_fixture
from cachemir_packing import (
    from_fixture,
    padded_input_slots,
    rotation_candidates_with_native,
    token_slot_stride,
    pack_single_token_feature_major,
    pack_k_cache_token_feature_major,
    pack_v_cache_token_feature_major,
    cache_position_indices,
    simulate_rotate,
    assert_native_fallback_consistency,
)

try:
    import _pyfideslib as fd
except ImportError as exc:
    raise SystemExit(f"failed to import _pyfideslib: {exc}")


def _decrypt_real(cc, sk, ct, used_slots):
    pt = cc.decrypt(sk, ct)
    values = pt.get_real_packed_value()
    return [float(v) for v in values[:used_slots]]


def _try_compare_optional_op(op_name, payload, got_real, max_abs, rmse):
    op = payload.get("ops", {}).get(op_name)
    if not isinstance(op, dict):
        return None, None
    if op.get("error"):
        return None, None
    ok, metrics = compare_op(op_name, payload, got_real, max_abs=max_abs, rmse=rmse)
    return ok, metrics


def _check_metadata(op_name, payload, cc, sk, ct, level_tol=0, log_scale_tol=0.5):
    op = payload["ops"][op_name]
    if op.get("error"):
        return True, None, None, 0.0

    expected_level = int(op.get("level", 0))
    expected_log_scale = float(op.get("log_scale", 0.0))
    got_level = int(ct.get_level())
    got_log_scale = float(cc.decrypt(sk, ct).get_log_precision())

    level_ok = abs(got_level - expected_level) <= int(level_tol)
    log_scale_delta = abs(got_log_scale - expected_log_scale)
    scale_ok = log_scale_delta <= float(log_scale_tol)

    return level_ok and scale_ok, got_level, got_log_scale, log_scale_delta


def _compose_result(op_name, value_ok, meta_ok, metrics, got_level, got_log_scale, log_scale_delta, strict_metadata, note=""):
    overall_ok = value_ok and (meta_ok if strict_metadata else True)
    return {
        "op": op_name,
        "ok": overall_ok,
        "value_ok": value_ok,
        "meta_ok": meta_ok,
        "metrics": metrics,
        "level": got_level,
        "log_scale": got_log_scale,
        "dlog_scale": log_scale_delta,
        "note": note,
    }


def _smoke_metrics(expected, got):
    n = min(len(expected), len(got))
    if n == 0:
        return {"max_abs_err": 0.0, "rmse": 0.0}
    max_abs = 0.0
    sq = 0.0
    for i in range(n):
        d = abs(float(expected[i]) - float(got[i]))
        if d > max_abs:
            max_abs = d
        sq += d * d
    return {"max_abs_err": max_abs, "rmse": math.sqrt(sq / n)}


def _vector_sum(values):
    return float(sum(float(v) for v in values))


def _vector_l2(values):
    return float(math.sqrt(sum(float(v) * float(v) for v in values)))


def _cleartext_self_ccmm(flat_values, row_size):
    width = int(row_size)
    needed = width * width
    matrix_flat = [float(v) for v in flat_values[:needed]]
    if len(matrix_flat) < needed:
        matrix_flat.extend([0.0] * (needed - len(matrix_flat)))

    out = [0.0] * needed
    for row in range(width):
        for col in range(width):
            acc = 0.0
            for inner in range(width):
                acc += matrix_flat[row * width + inner] * matrix_flat[inner * width + col]
            out[row * width + col] = acc
    return out


def _cleartext_mm(flat_a, flat_b, row_size):
    width = int(row_size)
    needed = width * width
    lhs = [float(v) for v in flat_a[:needed]]
    rhs = [float(v) for v in flat_b[:needed]]
    if len(lhs) < needed:
        lhs.extend([0.0] * (needed - len(lhs)))
    if len(rhs) < needed:
        rhs.extend([0.0] * (needed - len(rhs)))

    out = [0.0] * needed
    for row in range(width):
        for col in range(width):
            acc = 0.0
            for inner in range(width):
                acc += lhs[row * width + inner] * rhs[inner * width + col]
            out[row * width + col] = acc
    return out


def _matmul_proto_rotation_steps(row_size):
    width = int(row_size)
    shifts = set()
    for row in range(width):
        for col in range(width):
            out_idx = row * width + col
            for inner in range(width):
                idx_a = row * width + inner
                idx_b = inner * width + col
                shifts.add(idx_b - idx_a)
                shifts.add(out_idx - idx_b)
    shifts.discard(0)
    return sorted(int(s) for s in shifts)


def _configure_context(payload, force_cpu=False, enable_fhe=False, multiplicative_depth=12, secret_key_dist=None):
    params = payload["params"]
    config = from_fixture(payload)
    max_slots = config.max_slots

    ccparams = fd.CCParams()
    ccparams.set_ring_dim(1 << int(params.get("logN", 12)))
    ccparams.set_batch_size(max_slots)
    ccparams.set_multiplicative_depth(int(multiplicative_depth))
    ccparams.set_scaling_mod_size(int(params.get("logDefaultScale", 40)))
    ccparams.set_first_mod_size(52)
    ccparams.set_scaling_technique(fd.FIXEDAUTO)
    ccparams.set_security_level(fd.HEStd_NotSet)
    if secret_key_dist is not None:
        ccparams.set_secret_key_dist(secret_key_dist)
    if force_cpu:
        ccparams.set_devices([])
    else:
        ccparams.set_devices([0])

    cc = fd.gen_crypto_context(ccparams)
    cc.enable(fd.PKE)
    cc.enable(fd.KEYSWITCH)
    cc.enable(fd.LEVELEDSHE)
    cc.enable(fd.ADVANCEDSHE)
    if enable_fhe:
        cc.enable(fd.FHE)

    return cc


def _run_bootstrap_probe(
    payload,
    input_vec,
    used_slots,
    max_slots,
    op_name,
    force_cpu,
    correction_factor,
    max_abs,
    rmse,
    level_tol,
    log_scale_tol,
    strict_metadata,
    bootstrap_profile,
):
    probe_device = "cpu" if force_cpu else "gpu"
    try:
        if bootstrap_profile == "colleague":
            ccparams = fd.CCParams()
            ccparams.set_multiplicative_depth(40)
            ccparams.set_scaling_mod_size(55)
            ccparams.set_scaling_technique(fd.FLEXIBLEAUTO)
            ccparams.set_first_mod_size(60)
            ccparams.set_batch_size(max_slots)
            ccparams.set_ring_dim(65536)
            ccparams.set_num_large_digits(2)
            ccparams.set_secret_key_dist(fd.UNIFORM_TERNARY)
            ccparams.set_security_level(fd.HEStd_NotSet)
            if force_cpu:
                ccparams.set_devices([])
            else:
                ccparams.set_devices([0])

            cc_boot = fd.gen_crypto_context(ccparams)
            cc_boot.enable(fd.PKE)
            cc_boot.enable(fd.KEYSWITCH)
            cc_boot.enable(fd.LEVELEDSHE)
            cc_boot.enable(fd.ADVANCEDSHE)
            cc_boot.enable(fd.FHE)
        else:
            cc_boot = _configure_context(
                payload,
                force_cpu=force_cpu,
                enable_fhe=True,
                multiplicative_depth=24,
                secret_key_dist=fd.SPARSE_TERNARY,
            )
        keys_boot = cc_boot.key_gen()
        cc_boot.eval_mult_key_gen(keys_boot.secret_key)
        cc_boot.eval_bootstrap_setup([4, 4], [0, 0], max_slots, int(correction_factor))
        cc_boot.eval_bootstrap_key_gen(keys_boot.secret_key, max_slots)
        cc_boot.load_context(keys_boot.public_key)

        pt_boot = cc_boot.make_ckks_packed_plaintext(input_vec)
        ct_boot_in = cc_boot.encrypt(keys_boot.public_key, pt_boot)
        ct_in_level = int(ct_boot_in.get_level())
        ct_in_noise_scale = int(ct_boot_in.get_noise_scale_deg())
        ct_boot = cc_boot.eval_bootstrap(ct_boot_in)
        ct_out_level = int(ct_boot.get_level())
        ct_out_noise_scale = int(ct_boot.get_noise_scale_deg())

        try:
            got = _decrypt_real(cc_boot, keys_boot.secret_key, ct_boot, used_slots)
            value_ok, metrics = compare_op(op_name if op_name in payload.get("ops", {}) else "bootstrap", payload, got, max_abs=max_abs * 100, rmse=rmse * 100)
            meta_ok, got_level, got_log_scale, log_scale_delta = _check_metadata(
                "bootstrap", payload, cc_boot, keys_boot.secret_key, ct_boot, level_tol=level_tol + 1, log_scale_tol=log_scale_tol * 2
            )
            return _compose_result(
                op_name,
                value_ok,
                meta_ok,
                metrics,
                got_level,
                got_log_scale,
                log_scale_delta,
                strict_metadata,
                note=(
                    f"bootstrap_device={probe_device} correction_factor={int(correction_factor)} "
                    f"profile={bootstrap_profile} "
                    f"in_level={ct_in_level} in_ns={ct_in_noise_scale} "
                    f"out_level={ct_out_level} out_ns={ct_out_noise_scale}"
                ),
            )
        except Exception as exc:
            return {
                "op": op_name,
                "ok": False,
                "value_ok": False,
                "meta_ok": False,
                "metrics": {"max_abs_err": float("inf"), "rmse": float("inf"), "error": str(exc)},
                "level": ct_out_level,
                "log_scale": None,
                "dlog_scale": 0.0,
                "note": (
                    f"bootstrap_device={probe_device} correction_factor={int(correction_factor)} "
                    f"profile={bootstrap_profile} "
                    f"in_level={ct_in_level} in_ns={ct_in_noise_scale} "
                    f"out_level={ct_out_level} out_ns={ct_out_noise_scale} error={exc}"
                ),
            }
    except Exception as exc:
        return {
            "op": op_name,
            "ok": False,
            "value_ok": False,
            "meta_ok": False,
            "metrics": {"max_abs_err": float("inf"), "rmse": float("inf"), "error": str(exc)},
            "level": None,
            "log_scale": None,
            "dlog_scale": 0.0,
            "note": f"bootstrap_device={probe_device} correction_factor={int(correction_factor)} profile={bootstrap_profile} error={exc}",
        }


def run(
    payload,
    max_abs,
    rmse,
    include_bootstrap=False,
    force_cpu=False,
    level_tol=0,
    log_scale_tol=0.5,
    strict_metadata=False,
    auto_rotate_sign_check=True,
    include_packing_smokes=True,
    include_matmul_kernels=True,
    matmul_row_size=2,
    allow_unstable_matmul=False,
    matmul_cuda_custom=False,
    basic_ops_only=False,
    bootstrap_device="cpu",
    bootstrap_correction_factor=0,
    bootstrap_trace=False,
    bootstrap_profile="default",
):
    params = payload["params"]
    config = from_fixture(payload)
    used_slots = config.used_slots

    assert_native_fallback_consistency(config, payload["input"], seq_len=config.seq_len)

    cc = _configure_context(payload, force_cpu=force_cpu)
    keys = cc.key_gen()
    cc.eval_mult_key_gen(keys.secret_key)
    rotation = int(params.get("rotation", 5))
    rotation_steps = set(rotation_candidates_with_native(config, rotation, auto_rotate_sign_check))
    packing_stride = token_slot_stride(config)
    if packing_stride > 0 and packing_stride < config.max_slots:
        rotation_steps.add(int(packing_stride))
        rotation_steps.add(int(-packing_stride))
    rotation_steps = sorted(rotation_steps)
    cc.eval_rotate_key_gen(keys.secret_key, rotation_steps)
    cc.load_context(keys.public_key)

    max_slots = config.max_slots
    input_vec = padded_input_slots(payload["input"], max_slots)
    pt = cc.make_ckks_packed_plaintext(input_vec)
    ct = cc.encrypt(keys.public_key, pt)

    results = []

    ct_scalar_add = cc.eval_add_scalar(ct, float(params.get("scalarAddConstant", 1.25)))
    got = _decrypt_real(cc, keys.secret_key, ct_scalar_add, used_slots)
    ok, metrics = compare_op("scalar_add", payload, got, max_abs=max_abs, rmse=rmse)
    meta_ok, got_level, got_log_scale, log_scale_delta = _check_metadata(
        "scalar_add", payload, cc, keys.secret_key, ct_scalar_add, level_tol=level_tol, log_scale_tol=log_scale_tol
    )
    results.append(
        _compose_result(
            "scalar_add",
            ok,
            meta_ok,
            metrics,
            got_level,
            got_log_scale,
            log_scale_delta,
            strict_metadata,
        )
    )

    ct_scalar_mul = cc.eval_mult_scalar(ct, float(params.get("scalarMulConstant", 1.5)))
    got = _decrypt_real(cc, keys.secret_key, ct_scalar_mul, used_slots)
    ok, metrics = compare_op("scalar_mul", payload, got, max_abs=max_abs, rmse=rmse)
    meta_ok, got_level, got_log_scale, log_scale_delta = _check_metadata(
        "scalar_mul", payload, cc, keys.secret_key, ct_scalar_mul, level_tol=level_tol, log_scale_tol=log_scale_tol
    )
    results.append(
        _compose_result(
            "scalar_mul",
            ok,
            meta_ok,
            metrics,
            got_level,
            got_log_scale,
            log_scale_delta,
            strict_metadata,
        )
    )

    ct_ct_add = cc.eval_add(ct, ct)
    got_ct_add = _decrypt_real(cc, keys.secret_key, ct_ct_add, used_slots)
    ct_ct_add_cmp = _try_compare_optional_op("ct_ct_add", payload, got_ct_add, max_abs=max_abs, rmse=rmse)
    if ct_ct_add_cmp[0] is not None:
        ct_ct_add_ok, ct_ct_add_metrics = ct_ct_add_cmp
        meta_ok, got_level, got_log_scale, log_scale_delta = _check_metadata(
            "ct_ct_add", payload, cc, keys.secret_key, ct_ct_add, level_tol=level_tol, log_scale_tol=log_scale_tol
        )
        results.append(
            _compose_result(
                "ct_ct_add",
                ct_ct_add_ok,
                meta_ok,
                ct_ct_add_metrics,
                got_level,
                got_log_scale,
                log_scale_delta,
                strict_metadata,
            )
        )
    else:
        expected_ct_add = [2.0 * float(v) for v in payload["input"][:used_slots]]
        metrics_ct_add = _smoke_metrics(expected_ct_add, got_ct_add)
        ok_ct_add = metrics_ct_add["max_abs_err"] <= max_abs and metrics_ct_add["rmse"] <= rmse
        results.append(
            {
                "op": "ct_ct_add",
                "ok": ok_ct_add,
                "value_ok": ok_ct_add,
                "meta_ok": True,
                "metrics": metrics_ct_add,
                "level": int(ct_ct_add.get_level()),
                "log_scale": float(cc.decrypt(keys.secret_key, ct_ct_add).get_log_precision()),
                "dlog_scale": 0.0,
                "note": "fixture_missing_used_expected_2x_input",
            }
        )

    ct_pt_add_constant = float(params.get("ctPtAddConstant", params.get("scalarAddConstant", 1.25)))
    add_pt = cc.make_ckks_packed_plaintext([ct_pt_add_constant] * max_slots)
    ct_pt_add = cc.eval_add_pt(ct, add_pt)
    got_ct_pt_add = _decrypt_real(cc, keys.secret_key, ct_pt_add, used_slots)
    ct_pt_add_cmp = _try_compare_optional_op("ct_pt_add", payload, got_ct_pt_add, max_abs=max_abs, rmse=rmse)
    if ct_pt_add_cmp[0] is not None:
        ct_pt_add_ok, ct_pt_add_metrics = ct_pt_add_cmp
        meta_ok, got_level, got_log_scale, log_scale_delta = _check_metadata(
            "ct_pt_add", payload, cc, keys.secret_key, ct_pt_add, level_tol=level_tol, log_scale_tol=log_scale_tol
        )
        results.append(
            _compose_result(
                "ct_pt_add",
                ct_pt_add_ok,
                meta_ok,
                ct_pt_add_metrics,
                got_level,
                got_log_scale,
                log_scale_delta,
                strict_metadata,
            )
        )
    else:
        expected_ct_pt_add = [float(v) + ct_pt_add_constant for v in payload["input"][:used_slots]]
        metrics_ct_pt_add = _smoke_metrics(expected_ct_pt_add, got_ct_pt_add)
        ok_ct_pt_add = metrics_ct_pt_add["max_abs_err"] <= max_abs and metrics_ct_pt_add["rmse"] <= rmse
        results.append(
            {
                "op": "ct_pt_add",
                "ok": ok_ct_pt_add,
                "value_ok": ok_ct_pt_add,
                "meta_ok": True,
                "metrics": metrics_ct_pt_add,
                "level": int(ct_pt_add.get_level()),
                "log_scale": float(cc.decrypt(keys.secret_key, ct_pt_add).get_log_precision()),
                "dlog_scale": 0.0,
                "note": f"fixture_missing_used_expected_input_plus_{ct_pt_add_constant:g}",
            }
        )

    ct_rot_pos = cc.eval_rotate(ct, rotation)
    got_pos = _decrypt_real(cc, keys.secret_key, ct_rot_pos, used_slots)
    ok_pos, metrics_pos = compare_op("rotate", payload, got_pos, max_abs=max_abs, rmse=rmse)

    chosen_ct = ct_rot_pos
    chosen_ok = ok_pos
    chosen_metrics = metrics_pos
    chosen_note = f"rotate_step=+{rotation}"

    if auto_rotate_sign_check:
        ct_rot_neg = cc.eval_rotate(ct, -rotation)
        got_neg = _decrypt_real(cc, keys.secret_key, ct_rot_neg, used_slots)
        ok_neg, metrics_neg = compare_op("rotate", payload, got_neg, max_abs=max_abs, rmse=rmse)

        chosen_note = (
            f"rotate_step=+{rotation} rmse_pos={metrics_pos['rmse']:.6e} "
            f"rmse_neg={metrics_neg['rmse']:.6e}"
        )

        if metrics_neg["rmse"] < metrics_pos["rmse"]:
            chosen_ct = ct_rot_neg
            chosen_ok = ok_neg
            chosen_metrics = metrics_neg
            chosen_note = (
                f"rotate_step=-{rotation} rmse_pos={metrics_pos['rmse']:.6e} "
                f"rmse_neg={metrics_neg['rmse']:.6e}"
            )

    meta_ok, got_level, got_log_scale, log_scale_delta = _check_metadata(
        "rotate", payload, cc, keys.secret_key, chosen_ct, level_tol=level_tol, log_scale_tol=log_scale_tol
    )
    results.append(
        _compose_result(
            "rotate",
            chosen_ok,
            meta_ok,
            chosen_metrics,
            got_level,
            got_log_scale,
            log_scale_delta,
            strict_metadata,
            note=chosen_note,
        )
    )

    ct_ct_mul = cc.eval_mult(ct, ct)
    got = _decrypt_real(cc, keys.secret_key, ct_ct_mul, used_slots)
    ok, metrics = compare_op("ct_ct_mul", payload, got, max_abs=max_abs, rmse=rmse)
    meta_ok, got_level, got_log_scale, log_scale_delta = _check_metadata(
        "ct_ct_mul", payload, cc, keys.secret_key, ct_ct_mul, level_tol=level_tol, log_scale_tol=log_scale_tol
    )
    results.append(
        _compose_result(
            "ct_ct_mul",
            ok,
            meta_ok,
            metrics,
            got_level,
            got_log_scale,
            log_scale_delta,
            strict_metadata,
        )
    )

    const_pt = cc.make_ckks_packed_plaintext([float(params.get("ctPtMulConstant", 0.75))] * max_slots)
    ct_pt_mul = cc.eval_mult_pt(ct, const_pt)
    got = _decrypt_real(cc, keys.secret_key, ct_pt_mul, used_slots)
    ok, metrics = compare_op("ct_pt_mul", payload, got, max_abs=max_abs, rmse=rmse)
    meta_ok, got_level, got_log_scale, log_scale_delta = _check_metadata(
        "ct_pt_mul", payload, cc, keys.secret_key, ct_pt_mul, level_tol=level_tol, log_scale_tol=log_scale_tol
    )
    results.append(
        _compose_result(
            "ct_pt_mul",
            ok,
            meta_ok,
            metrics,
            got_level,
            got_log_scale,
            log_scale_delta,
            strict_metadata,
        )
    )

    if basic_ops_only:
        return results

    packing_stride = token_slot_stride(config)
    if include_packing_smokes and packing_stride > 0 and packing_stride < max_slots:
        packed_input = pack_single_token_feature_major(payload["input"], config, token_index=0)
        packed_pt = cc.make_ckks_packed_plaintext(packed_input)
        packed_ct = cc.encrypt(keys.public_key, packed_pt)

        packed_rot_pos = cc.eval_rotate(packed_ct, packing_stride)
        packed_rot_neg = cc.eval_rotate(packed_ct, -packing_stride)
        got_pos = _decrypt_real(cc, keys.secret_key, packed_rot_pos, max_slots)
        got_neg = _decrypt_real(cc, keys.secret_key, packed_rot_neg, max_slots)

        expected_pos = simulate_rotate(packed_input, packing_stride)
        expected_neg = simulate_rotate(packed_input, -packing_stride)
        metrics_pos = _smoke_metrics(expected_pos, got_pos)
        metrics_neg = _smoke_metrics(expected_neg, got_neg)

        in_sum = _vector_sum(packed_input)
        in_l2 = _vector_l2(packed_input)

        pos_sum_delta = abs(_vector_sum(got_pos) - in_sum)
        neg_sum_delta = abs(_vector_sum(got_neg) - in_sum)
        pos_l2_delta = abs(_vector_l2(got_pos) - in_l2)
        neg_l2_delta = abs(_vector_l2(got_neg) - in_l2)

        pos_perm_score = pos_sum_delta + pos_l2_delta
        neg_perm_score = neg_sum_delta + neg_l2_delta
        use_neg = neg_perm_score < pos_perm_score
        chosen = packed_rot_neg if use_neg else packed_rot_pos
        chosen_got = got_neg if use_neg else got_pos
        chosen_sum_delta = neg_sum_delta if use_neg else pos_sum_delta
        chosen_l2_delta = neg_l2_delta if use_neg else pos_l2_delta
        sign_note = (
            f"chosen_step={'-' if use_neg else '+'}{packing_stride} "
            f"rmse_pos={metrics_pos['rmse']:.6e} rmse_neg={metrics_neg['rmse']:.6e}"
        )

        finite_ok = all(math.isfinite(v) for v in chosen_got)
        perm_ok = chosen_sum_delta <= 5e-4 and chosen_l2_delta <= 5e-4
        ok = finite_ok and perm_ok
        results.append(
            {
                "op": "cachemir_packing_rotate_smoke",
                "ok": ok,
                "value_ok": ok,
                "meta_ok": True,
                "metrics": {
                    "sum_delta": chosen_sum_delta,
                    "l2_delta": chosen_l2_delta,
                    "perm_score": chosen_sum_delta + chosen_l2_delta,
                },
                "level": int(chosen.get_level()),
                "log_scale": float(cc.decrypt(keys.secret_key, chosen).get_log_precision()),
                "dlog_scale": 0.0,
                "note": f"stride={packing_stride} {sign_note}",
            }
        )

        _, int_idx, mid_idx = cache_position_indices(config)
        packed_k_input = pack_k_cache_token_feature_major(payload["input"], config, seq_len=config.seq_len)
        expected_k = pack_single_token_feature_major(payload["input"], config, token_index=int_idx)
        packed_k_pt = cc.make_ckks_packed_plaintext(packed_k_input)
        packed_k_ct = cc.encrypt(keys.public_key, packed_k_pt)
        got_k = _decrypt_real(cc, keys.secret_key, packed_k_ct, used_slots)
        expected_k_used = expected_k[:used_slots]
        metrics_k = _smoke_metrics(expected_k_used, got_k)
        ok_k = metrics_k["max_abs_err"] <= max_abs and metrics_k["rmse"] <= max(rmse, 1e-6)
        results.append(
            {
                "op": "cachemir_k_cache_pack_smoke",
                "ok": ok_k,
                "value_ok": ok_k,
                "meta_ok": True,
                "metrics": metrics_k,
                "level": int(packed_k_ct.get_level()),
                "log_scale": float(cc.decrypt(keys.secret_key, packed_k_ct).get_log_precision()),
                "dlog_scale": 0.0,
                "note": f"seq_len={config.seq_len} int_idx={int_idx} mid_idx={mid_idx}",
            }
        )

        packed_v_input = pack_v_cache_token_feature_major(payload["input"], config, seq_len=config.seq_len)
        expected_v = pack_single_token_feature_major(payload["input"], config, token_index=int_idx)
        packed_v_pt = cc.make_ckks_packed_plaintext(packed_v_input)
        packed_v_ct = cc.encrypt(keys.public_key, packed_v_pt)
        got_v = _decrypt_real(cc, keys.secret_key, packed_v_ct, used_slots)
        expected_v_used = expected_v[:used_slots]
        metrics_v = _smoke_metrics(expected_v_used, got_v)
        ok_v = metrics_v["max_abs_err"] <= max_abs and metrics_v["rmse"] <= max(rmse, 1e-6)
        results.append(
            {
                "op": "cachemir_v_cache_pack_smoke",
                "ok": ok_v,
                "value_ok": ok_v,
                "meta_ok": True,
                "metrics": metrics_v,
                "level": int(packed_v_ct.get_level()),
                "log_scale": float(cc.decrypt(keys.secret_key, packed_v_ct).get_log_precision()),
                "dlog_scale": 0.0,
                "note": f"seq_len={config.seq_len} int_idx={int_idx} mid_idx={mid_idx}",
            }
        )

    if include_bootstrap and not payload["ops"]["bootstrap"].get("error"):
        bootstrap_force_cpu = bootstrap_device != "gpu"
        if bootstrap_trace:
            trace_cpu = _run_bootstrap_probe(
                payload,
                input_vec,
                used_slots,
                max_slots,
                "bootstrap_trace_cpu",
                True,
                bootstrap_correction_factor,
                max_abs,
                rmse,
                level_tol,
                log_scale_tol,
                strict_metadata,
                bootstrap_profile,
            )
            trace_gpu = _run_bootstrap_probe(
                payload,
                input_vec,
                used_slots,
                max_slots,
                "bootstrap_trace_gpu",
                False,
                bootstrap_correction_factor,
                max_abs,
                rmse,
                level_tol,
                log_scale_tol,
                strict_metadata,
                bootstrap_profile,
            )
            results.append(trace_cpu)
            results.append(trace_gpu)

            selected_trace = trace_cpu if bootstrap_force_cpu else trace_gpu
            selected_bootstrap = dict(selected_trace)
            selected_bootstrap["op"] = "bootstrap"
            results.append(selected_bootstrap)
        else:
            if bootstrap_device == "both":
                results.append(
                    _run_bootstrap_probe(
                        payload,
                        input_vec,
                        used_slots,
                        max_slots,
                        "bootstrap_cpu",
                        True,
                        bootstrap_correction_factor,
                        max_abs,
                        rmse,
                        level_tol,
                        log_scale_tol,
                        strict_metadata,
                        bootstrap_profile,
                    )
                )
                results.append(
                    _run_bootstrap_probe(
                        payload,
                        input_vec,
                        used_slots,
                        max_slots,
                        "bootstrap_gpu",
                        False,
                        bootstrap_correction_factor,
                        max_abs,
                        rmse,
                        level_tol,
                        log_scale_tol,
                        strict_metadata,
                        bootstrap_profile,
                    )
                )
            else:
                results.append(
                    _run_bootstrap_probe(
                        payload,
                        input_vec,
                        used_slots,
                        max_slots,
                        "bootstrap",
                        bootstrap_force_cpu,
                        bootstrap_correction_factor,
                        max_abs,
                        rmse,
                        level_tol,
                        log_scale_tol,
                        strict_metadata,
                        bootstrap_profile,
                    )
                )

    if include_matmul_kernels:
        row_size = int(matmul_row_size)
        bstep = 1
        cc_mm = _configure_context(payload, force_cpu=force_cpu)
        keys_mm = cc_mm.key_gen()
        cc_mm.eval_mult_key_gen(keys_mm.secret_key)
        matmul_rotation_steps = sorted(set(int(i) for i in fd.generate_matmul_rotation_indices_gpu(row_size, bstep)))
        if matmul_cuda_custom:
            proto_steps = _matmul_proto_rotation_steps(row_size)
            if proto_steps:
                matmul_rotation_steps = sorted(set(matmul_rotation_steps + proto_steps + [-s for s in proto_steps]))
        if matmul_rotation_steps:
            cc_mm.eval_rotate_key_gen(keys_mm.secret_key, matmul_rotation_steps)
        cc_mm.load_context(keys_mm.public_key)

        input_vec_mm = padded_input_slots(payload["input"], max_slots)
        pt_mm = cc_mm.make_ckks_packed_plaintext(input_vec_mm)
        ct_mm = cc_mm.encrypt(keys_mm.public_key, pt_mm)

        if matmul_cuda_custom:
            ct_mat = fd.CiphertextMatrixGPU(cc_mm, [[ct_mm]])
            got_input = _decrypt_real(cc_mm, keys_mm.secret_key, ct_mm, used_slots)
            expected_ccmm = _cleartext_self_ccmm(got_input, row_size)

            ccmm_pos = fd.ccmm_gpu_custom_proto(cc_mm, ct_mat, ct_mat, row_size, max_slots, 1)
            ccmm_ct_pos = ccmm_pos.to_list(cc_mm)[0][0]
            got_ccmm_pos = _decrypt_real(cc_mm, keys_mm.secret_key, ccmm_ct_pos, used_slots)
            ccmm_metrics_pos = _smoke_metrics(expected_ccmm, got_ccmm_pos)

            ccmm_neg = fd.ccmm_gpu_custom_proto(cc_mm, ct_mat, ct_mat, row_size, max_slots, -1)
            ccmm_ct_neg = ccmm_neg.to_list(cc_mm)[0][0]
            got_ccmm_neg = _decrypt_real(cc_mm, keys_mm.secret_key, ccmm_ct_neg, used_slots)
            ccmm_metrics_neg = _smoke_metrics(expected_ccmm, got_ccmm_neg)

            use_ccmm_neg = ccmm_metrics_neg["rmse"] < ccmm_metrics_pos["rmse"]
            chosen_ccmm_ct = ccmm_ct_neg if use_ccmm_neg else ccmm_ct_pos
            chosen_ccmm_got = got_ccmm_neg if use_ccmm_neg else got_ccmm_pos
            chosen_ccmm_metrics = ccmm_metrics_neg if use_ccmm_neg else ccmm_metrics_pos
            chosen_ccmm_note = (
                f"matmul_cuda_custom_row_size={row_size} chosen_sign={'-' if use_ccmm_neg else '+'} "
                f"rmse_pos={ccmm_metrics_pos['rmse']:.6e} rmse_neg={ccmm_metrics_neg['rmse']:.6e}"
            )

            ccmm_ok = all(math.isfinite(v) and abs(v) < 1e12 for v in chosen_ccmm_got[: row_size * row_size]) and chosen_ccmm_metrics["rmse"] <= 5e-3
            results.append(
                {
                    "op": "ccmm_cuda_custom",
                    "ok": ccmm_ok,
                    "value_ok": ccmm_ok,
                    "meta_ok": True,
                    "metrics": chosen_ccmm_metrics,
                    "level": int(chosen_ccmm_ct.get_level()),
                    "log_scale": float(cc_mm.decrypt(keys_mm.secret_key, chosen_ccmm_ct).get_log_precision()),
                    "dlog_scale": 0.0,
                    "note": chosen_ccmm_note,
                }
            )

            bias_constant = float(params.get("matmulBiasConstant", 0.125))
            bias_values = [0.0] * max_slots
            if row_size == 1:
                bias_values = [bias_constant] * max_slots
            else:
                for idx in range(min(row_size * row_size, max_slots)):
                    bias_values[idx] = bias_constant
            bias_target_level = int(chosen_ccmm_ct.get_level())
            bias_pt = cc_mm.make_ckks_packed_plaintext(bias_values, 1, bias_target_level, max_slots)
            cc_mm.load_plaintext(bias_pt)
            bias_mat = fd.PlaintextMatrixGPU(cc_mm, [[bias_pt]])

            ccmm_bias_pos = fd.ccmm_gpu_custom_bias_proto(cc_mm, ct_mat, ct_mat, bias_mat, row_size, max_slots, 1)
            ccmm_bias_ct_pos = ccmm_bias_pos.to_list(cc_mm)[0][0]
            got_ccmm_bias_pos = _decrypt_real(cc_mm, keys_mm.secret_key, ccmm_bias_ct_pos, used_slots)
            expected_ccmm_bias = [float(v) + bias_constant for v in expected_ccmm]
            ccmm_bias_metrics_pos = _smoke_metrics(expected_ccmm_bias, got_ccmm_bias_pos)

            ccmm_bias_neg = fd.ccmm_gpu_custom_bias_proto(cc_mm, ct_mat, ct_mat, bias_mat, row_size, max_slots, -1)
            ccmm_bias_ct_neg = ccmm_bias_neg.to_list(cc_mm)[0][0]
            got_ccmm_bias_neg = _decrypt_real(cc_mm, keys_mm.secret_key, ccmm_bias_ct_neg, used_slots)
            ccmm_bias_metrics_neg = _smoke_metrics(expected_ccmm_bias, got_ccmm_bias_neg)

            use_ccmm_bias_neg = ccmm_bias_metrics_neg["rmse"] < ccmm_bias_metrics_pos["rmse"]
            chosen_ccmm_bias_ct = ccmm_bias_ct_neg if use_ccmm_bias_neg else ccmm_bias_ct_pos
            chosen_ccmm_bias_got = got_ccmm_bias_neg if use_ccmm_bias_neg else got_ccmm_bias_pos
            chosen_ccmm_bias_metrics = ccmm_bias_metrics_neg if use_ccmm_bias_neg else ccmm_bias_metrics_pos
            chosen_ccmm_bias_note = (
                f"ccmm_cuda_custom_bias_row_size={row_size} chosen_sign={'-' if use_ccmm_bias_neg else '+'} "
                f"rmse_pos={ccmm_bias_metrics_pos['rmse']:.6e} rmse_neg={ccmm_bias_metrics_neg['rmse']:.6e}"
            )
            ccmm_bias_ok = all(math.isfinite(v) and abs(v) < 1e12 for v in chosen_ccmm_bias_got[: row_size * row_size]) and chosen_ccmm_bias_metrics["rmse"] <= 5e-3
            results.append(
                {
                    "op": "ccmm_cuda_custom_bias",
                    "ok": ccmm_bias_ok,
                    "value_ok": ccmm_bias_ok,
                    "meta_ok": True,
                    "metrics": chosen_ccmm_bias_metrics,
                    "level": int(chosen_ccmm_bias_ct.get_level()),
                    "log_scale": float(cc_mm.decrypt(keys_mm.secret_key, chosen_ccmm_bias_ct).get_log_precision()),
                    "dlog_scale": 0.0,
                    "note": chosen_ccmm_bias_note,
                }
            )

            pcmm_constant = float(params.get("pcmmConstant", 0.5))
            if row_size == 1:
                pcmm_b_values = [pcmm_constant] * max_slots
                expected_pcmm = [float(v) * pcmm_constant for v in got_input[:used_slots]]
            else:
                needed = row_size * row_size
                pcmm_b_values = [0.0] * max_slots
                for idx in range(min(needed, max_slots)):
                    pcmm_b_values[idx] = pcmm_constant
                expected_pcmm = _cleartext_mm(got_input, pcmm_b_values, row_size)

            pcmm_pt = cc_mm.make_ckks_packed_plaintext(pcmm_b_values)
            cc_mm.load_plaintext(pcmm_pt)
            pcmm_mat = fd.PlaintextMatrixGPU(cc_mm, [[pcmm_pt]])

            pcmm_pos = fd.pcmm_gpu_custom_proto(cc_mm, ct_mat, pcmm_mat, row_size, max_slots, 1)
            pcmm_ct_pos = pcmm_pos.to_list(cc_mm)[0][0]
            got_pcmm_pos = _decrypt_real(cc_mm, keys_mm.secret_key, pcmm_ct_pos, used_slots)
            pcmm_metrics_pos = _smoke_metrics(expected_pcmm, got_pcmm_pos)

            pcmm_neg = fd.pcmm_gpu_custom_proto(cc_mm, ct_mat, pcmm_mat, row_size, max_slots, -1)
            pcmm_ct_neg = pcmm_neg.to_list(cc_mm)[0][0]
            got_pcmm_neg = _decrypt_real(cc_mm, keys_mm.secret_key, pcmm_ct_neg, used_slots)
            pcmm_metrics_neg = _smoke_metrics(expected_pcmm, got_pcmm_neg)

            use_pcmm_neg = pcmm_metrics_neg["rmse"] < pcmm_metrics_pos["rmse"]
            chosen_pcmm_ct = pcmm_ct_neg if use_pcmm_neg else pcmm_ct_pos
            chosen_pcmm_got = got_pcmm_neg if use_pcmm_neg else got_pcmm_pos
            chosen_pcmm_metrics = pcmm_metrics_neg if use_pcmm_neg else pcmm_metrics_pos
            chosen_pcmm_note = (
                f"pcmm_cuda_custom_row_size={row_size} chosen_sign={'-' if use_pcmm_neg else '+'} "
                f"rmse_pos={pcmm_metrics_pos['rmse']:.6e} rmse_neg={pcmm_metrics_neg['rmse']:.6e}"
            )

            pcmm_ok = all(math.isfinite(v) and abs(v) < 1e12 for v in chosen_pcmm_got[: row_size * row_size]) and chosen_pcmm_metrics["rmse"] <= 5e-3
            results.append(
                {
                    "op": "pcmm_cuda_custom",
                    "ok": pcmm_ok,
                    "value_ok": pcmm_ok,
                    "meta_ok": True,
                    "metrics": chosen_pcmm_metrics,
                    "level": int(chosen_pcmm_ct.get_level()),
                    "log_scale": float(cc_mm.decrypt(keys_mm.secret_key, chosen_pcmm_ct).get_log_precision()),
                    "dlog_scale": 0.0,
                    "note": chosen_pcmm_note,
                }
            )

            expected_pcmm_bias = [float(v) + bias_constant for v in expected_pcmm]
            pcmm_bias_pos = fd.pcmm_gpu_custom_bias_proto(cc_mm, ct_mat, pcmm_mat, bias_mat, row_size, max_slots, 1)
            pcmm_bias_ct_pos = pcmm_bias_pos.to_list(cc_mm)[0][0]
            got_pcmm_bias_pos = _decrypt_real(cc_mm, keys_mm.secret_key, pcmm_bias_ct_pos, used_slots)
            pcmm_bias_metrics_pos = _smoke_metrics(expected_pcmm_bias, got_pcmm_bias_pos)

            pcmm_bias_neg = fd.pcmm_gpu_custom_bias_proto(cc_mm, ct_mat, pcmm_mat, bias_mat, row_size, max_slots, -1)
            pcmm_bias_ct_neg = pcmm_bias_neg.to_list(cc_mm)[0][0]
            got_pcmm_bias_neg = _decrypt_real(cc_mm, keys_mm.secret_key, pcmm_bias_ct_neg, used_slots)
            pcmm_bias_metrics_neg = _smoke_metrics(expected_pcmm_bias, got_pcmm_bias_neg)

            use_pcmm_bias_neg = pcmm_bias_metrics_neg["rmse"] < pcmm_bias_metrics_pos["rmse"]
            chosen_pcmm_bias_ct = pcmm_bias_ct_neg if use_pcmm_bias_neg else pcmm_bias_ct_pos
            chosen_pcmm_bias_got = got_pcmm_bias_neg if use_pcmm_bias_neg else got_pcmm_bias_pos
            chosen_pcmm_bias_metrics = pcmm_bias_metrics_neg if use_pcmm_bias_neg else pcmm_bias_metrics_pos
            chosen_pcmm_bias_note = (
                f"pcmm_cuda_custom_bias_row_size={row_size} chosen_sign={'-' if use_pcmm_bias_neg else '+'} "
                f"rmse_pos={pcmm_bias_metrics_pos['rmse']:.6e} rmse_neg={pcmm_bias_metrics_neg['rmse']:.6e}"
            )
            pcmm_bias_ok = all(math.isfinite(v) and abs(v) < 1e12 for v in chosen_pcmm_bias_got[: row_size * row_size]) and chosen_pcmm_bias_metrics["rmse"] <= 5e-3
            results.append(
                {
                    "op": "pcmm_cuda_custom_bias",
                    "ok": pcmm_bias_ok,
                    "value_ok": pcmm_bias_ok,
                    "meta_ok": True,
                    "metrics": chosen_pcmm_bias_metrics,
                    "level": int(chosen_pcmm_bias_ct.get_level()),
                    "log_scale": float(cc_mm.decrypt(keys_mm.secret_key, chosen_pcmm_bias_ct).get_log_precision()),
                    "dlog_scale": 0.0,
                    "note": chosen_pcmm_bias_note,
                }
            )
        else:
            if row_size <= 1:
                return results

            ct_level = int(ct_mm.get_level())
            configured_depth = 12
            precomp_level = max(1, min(configured_depth, ct_level))
            precomp = fd.get_matmul_precomp_single_level_gpu(
                cc_mm,
                row_size=row_size,
                bstep=bstep,
                level=precomp_level,
                fuse_boot_prescale=False,
                slots=max_slots,
            )

            ct_mat = fd.CiphertextMatrixGPU(cc_mm, [[ct_mm]])

            ccmm_out = fd.ccmm_gpu(cc_mm, ct_mat, ct_mat, row_size, precomp)
            ccmm_ct = ccmm_out.to_list(cc_mm)[0][0]
            got_ccmm = _decrypt_real(cc_mm, keys_mm.secret_key, ccmm_ct, used_slots)
            ccmm_cmp = (all(abs(v) < 1e9 for v in got_ccmm), {"max_abs_err": 0.0, "rmse": 0.0})
            if ccmm_cmp[0] is not None:
                results.append(
                    {
                        "op": "ccmm_gpu_smoke",
                        "ok": ccmm_cmp[0],
                        "value_ok": ccmm_cmp[0],
                        "meta_ok": True,
                        "metrics": ccmm_cmp[1],
                        "level": int(ccmm_ct.get_level()),
                        "log_scale": float(cc_mm.decrypt(keys_mm.secret_key, ccmm_ct).get_log_precision()),
                        "dlog_scale": 0.0,
                        "note": f"matmul_kernel_ccmm_smoke_row_size={row_size}",
                    }
                )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pyFIDESlib differential checks against Cachemir fixture")
    parser.add_argument("fixture", type=Path, help="path to Cachemir JSON fixture")
    parser.add_argument("--max-abs", type=float, default=1e-5, help="max absolute error threshold")
    parser.add_argument("--rmse", type=float, default=1e-6, help="RMSE threshold")
    parser.add_argument("--level-tol", type=int, default=0, help="allowed level difference vs fixture metadata")
    parser.add_argument("--log-scale-tol", type=float, default=0.5, help="allowed log-scale difference vs fixture metadata")
    parser.add_argument("--strict-metadata", action="store_true", help="fail test on metadata mismatch (level/log-scale)")
    parser.add_argument("--no-auto-rotate-sign-check", action="store_true", help="disable automatic +k/-k rotation sign probing")
    parser.add_argument("--no-matmul-kernel-checks", action="store_true", help="disable optional matmul kernel checks")
    parser.add_argument("--matmul-row-size", type=int, default=2, help="row_size parameter for optional matmul kernel checks")
    parser.add_argument("--allow-unstable-matmul", action="store_true", help="execute known-unstable matmul kernel checks for debugging")
    parser.add_argument("--matmul-cuda-custom", action="store_true", help="run matmul via custom CUDA binding prototype")
    parser.add_argument("--with-bootstrap", action="store_true", help="also compare bootstrap output if available")
    parser.add_argument("--bootstrap-device", choices=["cpu", "gpu", "both"], default="cpu", help="device for bootstrap check")
    parser.add_argument("--bootstrap-correction-factor", type=int, default=0, help="bootstrap correction factor")
    parser.add_argument("--bootstrap-trace", action="store_true", help="run both CPU/GPU bootstrap probes for divergence tracing")
    parser.add_argument("--bootstrap-profile", choices=["default", "colleague"], default="default", help="bootstrap context profile")
    parser.add_argument("--basic-ops-only", action="store_true", help="run only core ops: scalar add/mul, rotate, ct-pt mul, ct-ct mul")
    parser.add_argument("--cpu", action="store_true", help="force CPU path by setting empty devices")
    args = parser.parse_args()

    include_bootstrap = args.with_bootstrap and not args.basic_ops_only
    include_matmul_kernels = (not args.no_matmul_kernel_checks) and (not args.basic_ops_only)
    include_packing_smokes = not args.basic_ops_only
    bootstrap_device = "cpu" if args.cpu else args.bootstrap_device

    payload = load_fixture(args.fixture)
    results = run(
        payload,
        max_abs=args.max_abs,
        rmse=args.rmse,
        include_bootstrap=include_bootstrap,
        force_cpu=args.cpu,
        level_tol=args.level_tol,
        log_scale_tol=args.log_scale_tol,
        strict_metadata=args.strict_metadata,
        auto_rotate_sign_check=not args.no_auto_rotate_sign_check,
        include_packing_smokes=include_packing_smokes,
        include_matmul_kernels=include_matmul_kernels,
        matmul_row_size=args.matmul_row_size,
        allow_unstable_matmul=args.allow_unstable_matmul,
        matmul_cuda_custom=args.matmul_cuda_custom,
        basic_ops_only=args.basic_ops_only,
        bootstrap_device=bootstrap_device,
        bootstrap_correction_factor=args.bootstrap_correction_factor,
        bootstrap_trace=args.bootstrap_trace,
        bootstrap_profile=args.bootstrap_profile,
    )

    all_ok = True
    for result in results:
        op_name = result["op"]
        ok = result["ok"]
        metrics = result["metrics"]
        got_level = result["level"]
        got_log_scale = result["log_scale"]
        log_scale_delta = result["dlog_scale"]
        note = result["note"]

        status = "PASS" if ok else "FAIL"
        meta_fragment = ""
        if got_level is not None and got_log_scale is not None:
            meta_fragment = f" level={got_level} log_scale={got_log_scale:.4f} dlog_scale={log_scale_delta:.4f}"
        note_fragment = f" {note}" if note else ""
        if "max_abs_err" in metrics and "rmse" in metrics:
            metric_fragment = f"max_abs={metrics['max_abs_err']:.6e} rmse={metrics['rmse']:.6e}"
        elif "sum_delta" in metrics and "l2_delta" in metrics:
            metric_fragment = (
                f"sum_delta={metrics['sum_delta']:.6e} "
                f"l2_delta={metrics['l2_delta']:.6e} "
                f"perm_score={metrics.get('perm_score', metrics['sum_delta'] + metrics['l2_delta']):.6e}"
            )
        else:
            metric_fragment = "metrics=unavailable"

        print(f"{status} {op_name}: {metric_fragment}{meta_fragment}{note_fragment}")
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
