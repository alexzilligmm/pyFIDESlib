from dataclasses import dataclass
from math import isqrt
from typing import Dict, List, Tuple


def _get_fd_module():
    try:
        import _pyfideslib as fd
        return fd
    except Exception:
        return None


def _field(obj, name):
    if obj is None:
        raise AttributeError(name)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    raise AttributeError(name)


def _build_native_engine(config: "CachemirPackingConfig"):
    fd = _get_fd_module()
    if fd is None or not hasattr(fd, "CachemirPackingEngine"):
        return None
    try:
        return fd.CachemirPackingEngine(
            int(config.max_slots),
            int(config.hid_dim),
            int(config.exp_dim),
            int(config.num_heads),
            int(config.seq_len),
        )
    except Exception:
        return None


def native_packing_plan(
    max_slots: int,
    hid_dim: int,
    exp_dim: int,
    num_heads: int,
    seq_len: int,
    rotation_step: int,
    auto_sign_check: bool,
):
    fd = _get_fd_module()
    if fd is None:
        return None

    config = CachemirPackingConfig(
        log_n=12,
        max_slots=int(max_slots),
        used_slots=int(max_slots),
        hid_dim=int(hid_dim),
        exp_dim=int(exp_dim),
        num_heads=int(num_heads),
        seq_len=int(seq_len),
    )
    engine = _build_native_engine(config)
    if engine is not None:
        try:
            return engine.compute_plan(
                rotation_step=int(rotation_step),
                auto_sign_check=bool(auto_sign_check),
            )
        except Exception:
            pass

    if not hasattr(fd, "cachemir_packing_plan"):
        return None

    try:
        return fd.cachemir_packing_plan(
            max_slots=max_slots,
            hid_dim=hid_dim,
            exp_dim=exp_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            rotation_step=rotation_step,
            auto_sign_check=auto_sign_check,
        )
    except Exception:
        return None


@dataclass(frozen=True)
class CachemirPackingConfig:
    log_n: int
    max_slots: int
    used_slots: int
    hid_dim: int
    exp_dim: int
    num_heads: int
    seq_len: int



def from_fixture(payload: Dict) -> CachemirPackingConfig:
    params = payload.get("params", {})
    used_slots = int(payload.get("used_slots", 0))
    max_slots = int(params.get("maxSlots", used_slots))
    return CachemirPackingConfig(
        log_n=int(params.get("logN", 12)),
        max_slots=max_slots,
        used_slots=used_slots,
        hid_dim=int(params.get("hidDim", 256)),
        exp_dim=int(params.get("expDim", 1024)),
        num_heads=int(params.get("numHeads", 32)),
        seq_len=int(params.get("seqLen", 512)),
    )



def padded_input_slots(input_values: List[float], max_slots: int) -> List[float]:
    slots = [0.0] * max_slots
    for idx, value in enumerate(input_values):
        if idx >= max_slots:
            break
        slots[idx] = float(value)
    return slots


def _token_slot_stride_fallback(config: CachemirPackingConfig) -> int:
    if config.hid_dim <= 0:
        return 1
    return max(1, config.max_slots // config.hid_dim)


def token_slot_stride(config: CachemirPackingConfig) -> int:
    engine = _build_native_engine(config)
    if engine is not None:
        try:
            return int(engine.token_slot_stride())
        except Exception:
            pass

    fd = _get_fd_module()
    if fd is not None:
        try:
            if hasattr(fd, "cachemir_token_slot_stride"):
                return int(
                    fd.cachemir_token_slot_stride(
                        int(config.max_slots),
                        int(config.hid_dim),
                    )
                )
        except Exception:
            pass

    return _token_slot_stride_fallback(config)


def _pack_single_token_feature_major_fallback(
    input_values: List[float],
    config: CachemirPackingConfig,
    token_index: int,
) -> List[float]:
    slots = [0.0] * config.max_slots
    stride = _token_slot_stride_fallback(config)
    token_slot = int(token_index) % stride

    width = min(len(input_values), config.hid_dim)
    for feat in range(width):
        slot_idx = feat * stride + token_slot
        if slot_idx >= config.max_slots:
            break
        slots[slot_idx] = float(input_values[feat])
    return slots


def pack_single_token_feature_major(
    input_values: List[float],
    config: CachemirPackingConfig,
    token_index: int = 0,
) -> List[float]:
    engine = _build_native_engine(config)
    if engine is not None:
        try:
            return [
                float(v)
                for v in engine.pack_single_token_feature_major(
                    input_values=[float(v) for v in input_values],
                    token_index=int(token_index),
                )
            ]
        except Exception:
            pass

    fd = _get_fd_module()
    if fd is not None:
        try:
            if hasattr(fd, "cachemir_pack_single_token_feature_major"):
                return [
                    float(v)
                    for v in fd.cachemir_pack_single_token_feature_major(
                        input_values=[float(v) for v in input_values],
                        max_slots=int(config.max_slots),
                        hid_dim=int(config.hid_dim),
                        token_index=int(token_index),
                    )
                ]
        except Exception:
            pass

    return _pack_single_token_feature_major_fallback(input_values, config, token_index)


def simulate_rotate(values: List[float], rotation: int) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    k = rotation % n
    if k == 0:
        return list(values)
    return list(values[-k:]) + list(values[:-k])



def rotation_candidates(rotation_step: int, auto_sign_check: bool) -> List[int]:
    if not auto_sign_check:
        return [rotation_step]
    if rotation_step == 0:
        return [0]
    return [rotation_step, -rotation_step]


def rotation_candidates_with_native(config: CachemirPackingConfig, rotation_step: int, auto_sign_check: bool) -> List[int]:
    fd = _get_fd_module()
    if fd is not None and hasattr(fd, "cachemir_rotation_indices"):
        try:
            return [
                int(v)
                for v in fd.cachemir_rotation_indices(
                    max_slots=int(config.max_slots),
                    hid_dim=int(config.hid_dim),
                    exp_dim=int(config.exp_dim),
                    num_heads=int(config.num_heads),
                    seq_len=int(config.seq_len),
                    rotation_step=int(rotation_step),
                    auto_sign_check=bool(auto_sign_check),
                    include_stride=False,
                )
            ]
        except Exception:
            pass

    plan = native_packing_plan(
        max_slots=config.max_slots,
        hid_dim=config.hid_dim,
        exp_dim=config.exp_dim,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        rotation_step=rotation_step,
        auto_sign_check=auto_sign_check,
    )
    try:
        if plan is not None:
            return [int(v) for v in _field(plan, "rotation_candidates")]
    except Exception:
        pass
    return rotation_candidates(rotation_step, auto_sign_check)



def linear_rotation_factors(config: CachemirPackingConfig, expand: bool = False) -> Tuple[int, int]:
    if not expand:
        in_rot = isqrt((config.hid_dim * config.hid_dim) // (2 * config.max_slots))
        out_rot = isqrt((2 * config.hid_dim * config.hid_dim) // config.max_slots)
        return max(in_rot, 1), max(out_rot, 1)

    in_rot = isqrt((config.hid_dim * config.exp_dim) // (2 * config.max_slots))
    in_rot = max(in_rot, 1)
    out_rot = (config.hid_dim * config.exp_dim) // (config.max_slots * in_rot)
    return in_rot, max(out_rot, 1)


def _cache_position_indices_fallback(config: CachemirPackingConfig, seq_len: int) -> Tuple[int, int, int]:
    int_rot = _token_slot_stride_fallback(config)
    int_idx = int(seq_len) % int_rot
    mid_idx = int(seq_len) // int_rot
    return int_rot, int_idx, mid_idx



def cache_position_indices(config: CachemirPackingConfig) -> Tuple[int, int, int]:
    engine = _build_native_engine(config)
    if engine is not None:
        try:
            out = engine.cache_position(int(config.seq_len))
            return int(_field(out, "int_rot")), int(_field(out, "int_idx")), int(_field(out, "mid_idx"))
        except Exception:
            pass

    fd = _get_fd_module()
    if fd is not None:
        try:
            if hasattr(fd, "cachemir_cache_position"):
                out = fd.cachemir_cache_position(
                    max_slots=int(config.max_slots),
                    hid_dim=int(config.hid_dim),
                    seq_len=int(config.seq_len),
                )
                return int(_field(out, "int_rot")), int(_field(out, "int_idx")), int(_field(out, "mid_idx"))
        except Exception:
            pass

    return _cache_position_indices_fallback(config, config.seq_len)


def _pack_k_cache_token_feature_major_fallback(
    input_values: List[float],
    config: CachemirPackingConfig,
    seq_len: int,
) -> List[float]:
    int_rot = _token_slot_stride_fallback(config)
    token_index = int(seq_len) % int_rot
    return _pack_single_token_feature_major_fallback(input_values, config, token_index=token_index)


def pack_k_cache_token_feature_major(
    input_values: List[float],
    config: CachemirPackingConfig,
    seq_len: int,
) -> List[float]:
    engine = _build_native_engine(config)
    if engine is not None:
        try:
            return [
                float(v)
                for v in engine.pack_k_cache_token_feature_major(
                    input_values=[float(v) for v in input_values],
                    seq_len=int(seq_len),
                )
            ]
        except Exception:
            pass

    fd = _get_fd_module()
    if fd is not None:
        try:
            if hasattr(fd, "cachemir_pack_k_cache_token_feature_major"):
                return [
                    float(v)
                    for v in fd.cachemir_pack_k_cache_token_feature_major(
                        input_values=[float(v) for v in input_values],
                        max_slots=int(config.max_slots),
                        hid_dim=int(config.hid_dim),
                        seq_len=int(seq_len),
                    )
                ]
        except Exception:
            pass

    return _pack_k_cache_token_feature_major_fallback(input_values, config, seq_len)


def _pack_v_cache_token_feature_major_fallback(
    input_values: List[float],
    config: CachemirPackingConfig,
    seq_len: int,
) -> List[float]:
    int_rot = _token_slot_stride_fallback(config)
    token_index = int(seq_len) % int_rot
    return _pack_single_token_feature_major_fallback(input_values, config, token_index=token_index)


def pack_v_cache_token_feature_major(
    input_values: List[float],
    config: CachemirPackingConfig,
    seq_len: int,
) -> List[float]:
    engine = _build_native_engine(config)
    if engine is not None:
        try:
            return [
                float(v)
                for v in engine.pack_v_cache_token_feature_major(
                    input_values=[float(v) for v in input_values],
                    seq_len=int(seq_len),
                )
            ]
        except Exception:
            pass

    fd = _get_fd_module()
    if fd is not None:
        try:
            if hasattr(fd, "cachemir_pack_v_cache_token_feature_major"):
                return [
                    float(v)
                    for v in fd.cachemir_pack_v_cache_token_feature_major(
                        input_values=[float(v) for v in input_values],
                        max_slots=int(config.max_slots),
                        hid_dim=int(config.hid_dim),
                        seq_len=int(seq_len),
                    )
                ]
        except Exception:
            pass

    return _pack_v_cache_token_feature_major_fallback(input_values, config, seq_len)


def assert_native_fallback_consistency(
    config: CachemirPackingConfig,
    input_values: List[float],
    seq_len: int,
    atol: float = 1e-12,
) -> None:
    fd = _get_fd_module()
    if fd is None:
        return

    engine = _build_native_engine(config)
    if engine is None:
        return

    native_stride = int(engine.token_slot_stride())
    fallback_stride = _token_slot_stride_fallback(config)
    if native_stride != fallback_stride:
        raise AssertionError(
            f"native stride mismatch: native={native_stride} fallback={fallback_stride}"
        )

    native_pos = engine.cache_position(int(seq_len))
    fallback_pos = _cache_position_indices_fallback(config, seq_len)
    if (
        int(_field(native_pos, "int_rot")) != int(fallback_pos[0])
        or int(_field(native_pos, "int_idx")) != int(fallback_pos[1])
        or int(_field(native_pos, "mid_idx")) != int(fallback_pos[2])
    ):
        raise AssertionError(
            "native cache_position mismatch: "
            f"native={{'int_rot': {_field(native_pos, 'int_rot')}, 'int_idx': {_field(native_pos, 'int_idx')}, 'mid_idx': {_field(native_pos, 'mid_idx')}}} "
            f"fallback={{'int_rot': {fallback_pos[0]}, 'int_idx': {fallback_pos[1]}, 'mid_idx': {fallback_pos[2]}}}"
        )

    token_index = int(seq_len) % native_stride
    native_single = [
        float(v)
        for v in engine.pack_single_token_feature_major(
            input_values=[float(v) for v in input_values],
            token_index=int(token_index),
        )
    ]
    fallback_single = _pack_single_token_feature_major_fallback(input_values, config, token_index)
    single_max_abs = max(abs(a - b) for a, b in zip(native_single, fallback_single)) if native_single else 0.0
    if single_max_abs > float(atol):
        raise AssertionError(
            f"native single-token pack mismatch: max_abs={single_max_abs:.3e} atol={atol:.3e}"
        )

    native_k_cache = [
        float(v)
        for v in engine.pack_k_cache_token_feature_major(
            input_values=[float(v) for v in input_values],
            seq_len=int(seq_len),
        )
    ]
    fallback_k_cache = _pack_k_cache_token_feature_major_fallback(input_values, config, seq_len)
    k_max_abs = max(abs(a - b) for a, b in zip(native_k_cache, fallback_k_cache)) if native_k_cache else 0.0
    if k_max_abs > float(atol):
        raise AssertionError(
            f"native k-cache pack mismatch: max_abs={k_max_abs:.3e} atol={atol:.3e}"
        )

    native_v_cache = [
        float(v)
        for v in engine.pack_v_cache_token_feature_major(
            input_values=[float(v) for v in input_values],
            seq_len=int(seq_len),
        )
    ]
    fallback_v_cache = _pack_v_cache_token_feature_major_fallback(input_values, config, seq_len)
    v_max_abs = max(abs(a - b) for a, b in zip(native_v_cache, fallback_v_cache)) if native_v_cache else 0.0
    if v_max_abs > float(atol):
        raise AssertionError(
            f"native v-cache pack mismatch: max_abs={v_max_abs:.3e} atol={atol:.3e}"
        )
