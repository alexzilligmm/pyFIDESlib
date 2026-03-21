#include "cachemir_packing_native.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace {

uint32_t isqrt_u64(uint64_t x) {
    return static_cast<uint32_t>(std::sqrt(static_cast<long double>(x)));
}

CachemirPackingLayoutNative MakeLayout(
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t exp_dim,
    uint32_t num_heads,
    uint32_t seq_len) {
    if (max_slots == 0 || hid_dim == 0 || exp_dim == 0 || num_heads == 0) {
        throw std::invalid_argument("max_slots, hid_dim, exp_dim and num_heads must be > 0");
    }
    CachemirPackingLayoutNative layout{};
    layout.max_slots = max_slots;
    layout.hid_dim = hid_dim;
    layout.exp_dim = exp_dim;
    layout.num_heads = num_heads;
    layout.seq_len = seq_len;
    layout.token_slot_stride = std::max<uint32_t>(1, max_slots / hid_dim);
    return layout;
}

}  // namespace

CachemirPackingEngineNative::CachemirPackingEngineNative(
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t exp_dim,
    uint32_t num_heads,
    uint32_t seq_len)
    : layout_(MakeLayout(max_slots, hid_dim, exp_dim, num_heads, seq_len)) {}

CachemirPackingLayoutNative CachemirPackingEngineNative::Layout() const {
    return layout_;
}

uint32_t CachemirPackingEngineNative::TokenSlotStride() const {
    return layout_.token_slot_stride;
}

CachemirPackingPlanNative CachemirPackingEngineNative::ComputePlan(
    int32_t rotation_step,
    bool auto_sign_check) const {
    const uint64_t hid2 = static_cast<uint64_t>(layout_.hid_dim) * static_cast<uint64_t>(layout_.hid_dim);
    const uint64_t hid_exp =
        static_cast<uint64_t>(layout_.hid_dim) * static_cast<uint64_t>(layout_.exp_dim);

    CachemirPackingPlanNative plan{};
    plan.max_slots = layout_.max_slots;
    plan.hid_dim = layout_.hid_dim;
    plan.exp_dim = layout_.exp_dim;
    plan.num_heads = layout_.num_heads;
    plan.seq_len = layout_.seq_len;
    plan.in_rot = std::max<uint32_t>(1, isqrt_u64(hid2 / (2 * layout_.max_slots)));
    plan.out_rot = std::max<uint32_t>(1, isqrt_u64((2 * hid2) / layout_.max_slots));
    plan.in_rot_expand = std::max<uint32_t>(1, isqrt_u64(hid_exp / (2 * layout_.max_slots)));
    plan.out_rot_expand = std::max<uint32_t>(
        1, static_cast<uint32_t>(hid_exp / (layout_.max_slots * plan.in_rot_expand)));
    plan.int_rot = layout_.token_slot_stride;
    plan.int_idx = layout_.seq_len % plan.int_rot;
    plan.mid_idx = layout_.seq_len / plan.int_rot;

    if (!auto_sign_check || rotation_step == 0) {
        plan.rotation_candidates.push_back(rotation_step);
    } else {
        plan.rotation_candidates.push_back(rotation_step);
        plan.rotation_candidates.push_back(-rotation_step);
    }
    return plan;
}

std::vector<int32_t> CachemirPackingEngineNative::BuildRotationIndices(
    int32_t rotation_step,
    bool auto_sign_check,
    bool include_stride) const {
    std::set<int32_t> uniq;
    const auto plan = ComputePlan(rotation_step, auto_sign_check);
    for (const auto step : plan.rotation_candidates) {
        uniq.insert(step);
    }
    if (include_stride && layout_.token_slot_stride < layout_.max_slots) {
        uniq.insert(static_cast<int32_t>(layout_.token_slot_stride));
        uniq.insert(-static_cast<int32_t>(layout_.token_slot_stride));
    }
    return std::vector<int32_t>(uniq.begin(), uniq.end());
}

CachemirCachePositionNative CachemirPackingEngineNative::CachePosition(uint32_t seq_len) const {
    CachemirCachePositionNative pos{};
    pos.int_rot = layout_.token_slot_stride;
    pos.int_idx = seq_len % pos.int_rot;
    pos.mid_idx = seq_len / pos.int_rot;
    return pos;
}

std::vector<double> CachemirPackingEngineNative::PackSingleTokenFeatureMajor(
    const std::vector<double>& input_values,
    uint32_t token_index) const {
    const uint32_t token_slot = token_index % layout_.token_slot_stride;

    std::vector<double> slots(layout_.max_slots, 0.0);
    const uint32_t width = std::min<uint32_t>(static_cast<uint32_t>(input_values.size()), layout_.hid_dim);
    for (uint32_t feat = 0; feat < width; ++feat) {
        const uint64_t slot_idx =
            static_cast<uint64_t>(feat) * static_cast<uint64_t>(layout_.token_slot_stride) + token_slot;
        if (slot_idx >= layout_.max_slots) {
            break;
        }
        slots[static_cast<size_t>(slot_idx)] = input_values[feat];
    }
    return slots;
}

std::vector<double> CachemirPackingEngineNative::PackKCacheTokenFeatureMajor(
    const std::vector<double>& input_values,
    uint32_t seq_len) const {
    const auto pos = CachePosition(seq_len);
    return PackSingleTokenFeatureMajor(input_values, pos.int_idx);
}

std::vector<double> CachemirPackingEngineNative::PackVCacheTokenFeatureMajor(
    const std::vector<double>& input_values,
    uint32_t seq_len) const {
    const auto pos = CachePosition(seq_len);
    return PackSingleTokenFeatureMajor(input_values, pos.int_idx);
}

CachemirPackingPlanNative ComputeCachemirPackingPlan(
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t exp_dim,
    uint32_t num_heads,
    uint32_t seq_len,
    int32_t rotation_step,
    bool auto_sign_check) {
    return CachemirPackingEngineNative(max_slots, hid_dim, exp_dim, num_heads, seq_len)
        .ComputePlan(rotation_step, auto_sign_check);
}

uint32_t ComputeCachemirTokenSlotStride(uint32_t max_slots, uint32_t hid_dim) {
    return CachemirPackingEngineNative(max_slots, hid_dim, 1, 1, 0).TokenSlotStride();
}

std::vector<double> PackSingleTokenFeatureMajor(
    const std::vector<double>& input_values,
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t token_index) {
    return CachemirPackingEngineNative(max_slots, hid_dim, 1, 1, 0)
        .PackSingleTokenFeatureMajor(input_values, token_index);
}

CachemirCachePositionNative ComputeCachemirCachePosition(
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t seq_len) {
    return CachemirPackingEngineNative(max_slots, hid_dim, 1, 1, seq_len)
        .CachePosition(seq_len);
}

std::vector<double> PackKCacheTokenFeatureMajor(
    const std::vector<double>& input_values,
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t seq_len) {
    return CachemirPackingEngineNative(max_slots, hid_dim, 1, 1, seq_len)
        .PackKCacheTokenFeatureMajor(input_values, seq_len);
}

std::vector<double> PackVCacheTokenFeatureMajor(
    const std::vector<double>& input_values,
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t seq_len) {
    return CachemirPackingEngineNative(max_slots, hid_dim, 1, 1, seq_len)
        .PackVCacheTokenFeatureMajor(input_values, seq_len);
}

std::vector<int32_t> ComputeCachemirRotationIndices(
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t exp_dim,
    uint32_t num_heads,
    uint32_t seq_len,
    int32_t rotation_step,
    bool auto_sign_check,
    bool include_stride) {
    return CachemirPackingEngineNative(max_slots, hid_dim, exp_dim, num_heads, seq_len)
        .BuildRotationIndices(rotation_step, auto_sign_check, include_stride);
}
