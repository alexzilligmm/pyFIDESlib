#pragma once

#include <cstdint>
#include <vector>

struct CachemirPackingPlanNative {
    uint32_t max_slots;
    uint32_t hid_dim;
    uint32_t exp_dim;
    uint32_t num_heads;
    uint32_t seq_len;
    uint32_t in_rot;
    uint32_t out_rot;
    uint32_t in_rot_expand;
    uint32_t out_rot_expand;
    uint32_t int_rot;
    uint32_t int_idx;
    uint32_t mid_idx;
    std::vector<int32_t> rotation_candidates;
};

struct CachemirCachePositionNative {
    uint32_t int_rot;
    uint32_t int_idx;
    uint32_t mid_idx;
};

struct CachemirPackingLayoutNative {
    uint32_t max_slots;
    uint32_t hid_dim;
    uint32_t exp_dim;
    uint32_t num_heads;
    uint32_t seq_len;
    uint32_t token_slot_stride;
};

class CachemirPackingEngineNative {
public:
    CachemirPackingEngineNative(
        uint32_t max_slots,
        uint32_t hid_dim,
        uint32_t exp_dim,
        uint32_t num_heads,
        uint32_t seq_len);

    CachemirPackingLayoutNative Layout() const;
    uint32_t TokenSlotStride() const;

    CachemirPackingPlanNative ComputePlan(int32_t rotation_step, bool auto_sign_check) const;
    std::vector<int32_t> BuildRotationIndices(
        int32_t rotation_step,
        bool auto_sign_check,
        bool include_stride) const;

    CachemirCachePositionNative CachePosition(uint32_t seq_len) const;
    std::vector<double> PackSingleTokenFeatureMajor(
        const std::vector<double>& input_values,
        uint32_t token_index) const;
    std::vector<double> PackKCacheTokenFeatureMajor(
        const std::vector<double>& input_values,
        uint32_t seq_len) const;
    std::vector<double> PackVCacheTokenFeatureMajor(
        const std::vector<double>& input_values,
        uint32_t seq_len) const;

private:
    CachemirPackingLayoutNative layout_;
};

CachemirPackingPlanNative ComputeCachemirPackingPlan(
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t exp_dim,
    uint32_t num_heads,
    uint32_t seq_len,
    int32_t rotation_step,
    bool auto_sign_check);

uint32_t ComputeCachemirTokenSlotStride(uint32_t max_slots, uint32_t hid_dim);

std::vector<double> PackSingleTokenFeatureMajor(
    const std::vector<double>& input_values,
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t token_index);

CachemirCachePositionNative ComputeCachemirCachePosition(
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t seq_len);

std::vector<double> PackKCacheTokenFeatureMajor(
    const std::vector<double>& input_values,
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t seq_len);

std::vector<double> PackVCacheTokenFeatureMajor(
    const std::vector<double>& input_values,
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t seq_len);

std::vector<int32_t> ComputeCachemirRotationIndices(
    uint32_t max_slots,
    uint32_t hid_dim,
    uint32_t exp_dim,
    uint32_t num_heads,
    uint32_t seq_len,
    int32_t rotation_step,
    bool auto_sign_check,
    bool include_stride);
