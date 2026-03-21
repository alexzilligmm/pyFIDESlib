#include <any>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fideslib.hpp>

#include "MatMul.cuh"
#include "Inputs.cuh"
#include "Transpose.cuh"
#include "matmul_bindings.h"

// Inlined from Transformer.cu 
namespace FIDESlib::CKKS {
static void dropMatrixLevel(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& in, int level) {
    for (auto& row : in)
        for (auto& ct : row) {
            if (ct.NoiseLevel == 2) ct.rescale();
            if (ct.getLevel() > level) {
                ct.dropToLevel(level);
            }
        }
}
} // namespace FIDESlib::CKKS

namespace py = pybind11;

using FidelCC    = fideslib::CryptoContextImpl<fideslib::DCRTPoly>;
using FidelCCPtr = fideslib::CryptoContext<fideslib::DCRTPoly>;
using FidelCT    = fideslib::CiphertextImpl<fideslib::DCRTPoly>;
using FidelCTPtr = fideslib::Ciphertext<fideslib::DCRTPoly>;
using FidelPTPtr = fideslib::Plaintext;
using FidelPKPtr = fideslib::PublicKey<fideslib::DCRTPoly>;


struct CiphertextMatrixGPU {
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> data;
    std::any tmpl_cpu;
    CiphertextMatrixGPU() = default;
    CiphertextMatrixGPU(CiphertextMatrixGPU&&) = default;
    CiphertextMatrixGPU& operator=(CiphertextMatrixGPU&&) = default;
    CiphertextMatrixGPU(const CiphertextMatrixGPU&) = delete;
    CiphertextMatrixGPU& operator=(const CiphertextMatrixGPU&) = delete;
};

struct PlaintextMatrixGPU {
    std::vector<std::vector<FIDESlib::CKKS::Plaintext>> data;
    PlaintextMatrixGPU() = default;
    PlaintextMatrixGPU(PlaintextMatrixGPU&&) = default;
    PlaintextMatrixGPU& operator=(PlaintextMatrixGPU&&) = default;
    PlaintextMatrixGPU(const PlaintextMatrixGPU&) = delete;
    PlaintextMatrixGPU& operator=(const PlaintextMatrixGPU&) = delete;
};

namespace {

inline void require_non_empty_ct_matrix(
        const std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix,
        const std::string& name) {
    if (matrix.empty() || matrix[0].empty()) {
        throw std::runtime_error(name + " must be non-empty");
    }
}

inline int min_ct_level(const std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix) {
    int min_level = std::numeric_limits<int>::max();
    for (const auto& row : matrix) {
        for (const auto& ct : row) {
            min_level = std::min(min_level, ct.getLevel());
        }
    }
    return min_level;
}

inline int precomp_min_level(const FIDESlib::CKKS::MatrixMatrixProductPrecomputations_GPU& precomp) {
    if (precomp.pts_1.empty() || precomp.pts_1[0] == nullptr) {
        throw std::runtime_error("invalid matmul precomp: pts_1 is empty");
    }
    if (precomp.pts_2.empty() || precomp.pts_2[0] == nullptr) {
        throw std::runtime_error("invalid matmul precomp: pts_2 is empty");
    }
    if (precomp.pts_3_1.empty() || precomp.pts_3_1[0] == nullptr) {
        throw std::runtime_error("invalid matmul precomp: pts_3_1 is empty");
    }
    const int lvl1 = precomp.pts_1[0]->c0.getLevel();
    const int lvl2 = precomp.pts_2[0]->c0.getLevel();
    const int lvl3 = precomp.pts_3_1[0]->c0.getLevel();
    return std::min({lvl1, lvl2, lvl3});
}

inline void validate_ccmm_preconditions(
        const std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
        const std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2,
        int row_size,
        const FIDESlib::CKKS::MatrixMatrixProductPrecomputations_GPU& precomp) {
    if (row_size <= 0) {
        throw std::runtime_error("row_size must be > 0");
    }
    require_non_empty_ct_matrix(matrix1, "mat1");
    require_non_empty_ct_matrix(matrix2, "mat2");

    const int min_lvl_m1 = min_ct_level(matrix1);
    const int min_lvl_m2 = min_ct_level(matrix2);
    (void)precomp_min_level(precomp);

    if (std::abs(min_lvl_m1 - min_lvl_m2) > 3) {
        std::ostringstream oss;
        oss << "mat1/mat2 level gap too large for stable CCMM: "
            << "min_level(mat1)=" << min_lvl_m1
            << ", min_level(mat2)=" << min_lvl_m2;
        throw std::runtime_error(oss.str());
    }
}

inline void validate_pcmm_preconditions(
        const std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
        const FIDESlib::CKKS::MatrixMatrixProductPrecomputations_GPU& precomp,
        int row_size) {
    if (row_size <= 0) {
        throw std::runtime_error("row_size must be > 0");
    }
    require_non_empty_ct_matrix(matrix1, "mat1");

    (void)precomp_min_level(precomp);
}

inline FIDESlib::CKKS::Context& gpu_ctx(const FidelCCPtr& cc) {
    return std::any_cast<FIDESlib::CKKS::Context&>(cc->gpu);
}

inline lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cpu_ctx(const FidelCCPtr& cc) {
    return std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
}

inline lbcrypto::PublicKey<lbcrypto::DCRTPoly>& cpu_pk(const FidelPKPtr& pk) {
    return std::any_cast<lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(pk->pimpl);
}

inline FIDESlib::CKKS::Ciphertext& raw_gct(const FidelCCPtr& cc, const FidelCTPtr& ct) {
    return *static_cast<FIDESlib::CKKS::Ciphertext*>(
        cc->GetDeviceCiphertext(ct->gpu).get());
}

inline FIDESlib::CKKS::Plaintext& raw_gpt(const FidelCCPtr& cc, const FidelPTPtr& pt) {
    return *static_cast<FIDESlib::CKKS::Plaintext*>(
        cc->GetDevicePlaintext(pt->gpu).get());
}

CiphertextMatrixGPU wrap_product(
        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>&& product,
        const std::any& tmpl_cpu) {
    CiphertextMatrixGPU out;
    out.data     = std::move(product);
    out.tmpl_cpu = tmpl_cpu;
    return out;
}

FIDESlib::CKKS::Plaintext make_one_hot_plaintext(
        FidelCCPtr cc,
        int slot_index,
        int max_slots,
        int level) {
    auto& context = cpu_ctx(cc);
    FIDESlib::CKKS::Context& gctx = gpu_ctx(cc);

    std::vector<double> values(static_cast<size_t>(max_slots), 0.0);
    values[static_cast<size_t>(slot_index)] = 1.0;

    const int safe_level = std::max(1, level);
    auto cpu_pt = context->MakeCKKSPackedPlaintext(values, 1, gctx->L - safe_level);
    auto raw_pt = FIDESlib::CKKS::GetRawPlainText(context, cpu_pt);
    return FIDESlib::CKKS::Plaintext(gctx, raw_pt);
}

FIDESlib::CKKS::Plaintext make_block_mask_plaintext(
        FidelCCPtr cc,
        int active_slots,
        int max_slots,
        int level) {
    auto& context = cpu_ctx(cc);
    FIDESlib::CKKS::Context& gctx = gpu_ctx(cc);

    std::vector<double> values(static_cast<size_t>(max_slots), 0.0);
    const int capped_active = std::max(0, std::min(active_slots, max_slots));
    for (int idx = 0; idx < capped_active; ++idx) {
        values[static_cast<size_t>(idx)] = 1.0;
    }

    const int safe_level = std::max(1, level);
    auto cpu_pt = context->MakeCKKSPackedPlaintext(values, 1, gctx->L - safe_level);
    auto raw_pt = FIDESlib::CKKS::GetRawPlainText(context, cpu_pt);
    return FIDESlib::CKKS::Plaintext(gctx, raw_pt);
}

CiphertextMatrixGPU ccmm_custom_proto_gpu(
        FidelCCPtr cc,
        CiphertextMatrixGPU& mat1,
        CiphertextMatrixGPU& mat2,
        int row_size,
        int max_slots,
        int rotate_sign) {
    if (row_size <= 0) {
        throw std::runtime_error("row_size must be > 0");
    }
    if (max_slots <= 0) {
        throw std::runtime_error("max_slots must be > 0");
    }
    if (mat1.data.size() != 1 || mat1.data[0].size() != 1 || mat2.data.size() != 1 || mat2.data[0].size() != 1) {
        throw std::runtime_error("ccmm_gpu_custom_proto currently supports only 1x1 ciphertext matrices");
    }

    const int needed_slots = row_size * row_size;
    if (needed_slots > max_slots) {
        throw std::runtime_error("row_size^2 exceeds max_slots");
    }

    FIDESlib::CKKS::Context& gctx = gpu_ctx(cc);
    const auto& src_a = mat1.data[0][0];
    const auto& src_b = mat2.data[0][0];
    const int mask_level = std::max(1, std::min(src_a.getLevel(), src_b.getLevel()));

    std::unordered_map<int, std::unique_ptr<FIDESlib::CKKS::Plaintext>> mask_cache;
    mask_cache.reserve(static_cast<size_t>(needed_slots));

    auto get_mask = [&](int slot_idx) -> FIDESlib::CKKS::Plaintext& {
        auto it = mask_cache.find(slot_idx);
        if (it == mask_cache.end()) {
            auto pt = std::make_unique<FIDESlib::CKKS::Plaintext>(make_one_hot_plaintext(cc, slot_idx, max_slots, mask_level));
            it = mask_cache.emplace(slot_idx, std::move(pt)).first;
        }
        return *(it->second);
    };

    FIDESlib::CKKS::Ciphertext out_ct(gctx);
    bool out_init = false;

    for (int row = 0; row < row_size; ++row) {
        for (int col = 0; col < row_size; ++col) {
            const int out_idx = row * row_size + col;

            FIDESlib::CKKS::Ciphertext acc(gctx);
            bool acc_init = false;

            for (int inner = 0; inner < row_size; ++inner) {
                const int idx_a = row * row_size + inner;
                const int idx_b = inner * row_size + col;

                FIDESlib::CKKS::Ciphertext ct_a(gctx);
                ct_a.multPt(src_a, get_mask(idx_a), false);
                const int shift_align = (idx_b - idx_a) * rotate_sign;
                if (shift_align != 0) {
                    ct_a.rotate(shift_align);
                }

                FIDESlib::CKKS::Ciphertext ct_b(gctx);
                ct_b.multPt(src_b, get_mask(idx_b), false);

                FIDESlib::CKKS::Ciphertext term(gctx);
                term.mult(ct_a, ct_b, false);

                const int shift_out = (out_idx - idx_b) * rotate_sign;
                if (shift_out != 0) {
                    term.rotate(shift_out);
                }

                if (!acc_init) {
                    acc.copy(term);
                    acc_init = true;
                } else {
                    acc.add(term);
                }
            }

            if (!out_init) {
                out_ct.copy(acc);
                out_init = true;
            } else {
                out_ct.add(acc);
            }
        }
    }

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> product(1);
    product[0].push_back(std::move(out_ct));
    return wrap_product(std::move(product), mat1.tmpl_cpu);
}

CiphertextMatrixGPU cpmm_custom_proto_gpu(
        FidelCCPtr cc,
        CiphertextMatrixGPU& mat1,
        PlaintextMatrixGPU& mat2,
        int row_size,
        int max_slots,
        int rotate_sign) {
    if (row_size <= 0) {
        throw std::runtime_error("row_size must be > 0");
    }
    if (max_slots <= 0) {
        throw std::runtime_error("max_slots must be > 0");
    }
    if (mat1.data.size() != 1 || mat1.data[0].size() != 1 || mat2.data.size() != 1 || mat2.data[0].size() != 1) {
        throw std::runtime_error("cpmm_gpu_custom_proto currently supports only 1x1 matrix containers");
    }

    FIDESlib::CKKS::Context& gctx = gpu_ctx(cc);
    const auto& src_a = mat1.data[0][0];
    const auto& src_b = mat2.data[0][0];

    if (row_size == 1) {
        FIDESlib::CKKS::Ciphertext out_ct(gctx);
        out_ct.multPt(src_a, src_b, false);
        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> product(1);
        product[0].push_back(std::move(out_ct));
        return wrap_product(std::move(product), mat1.tmpl_cpu);
    }

    const int needed_slots = row_size * row_size;
    if (needed_slots > max_slots) {
        throw std::runtime_error("row_size^2 exceeds max_slots");
    }

    const int mask_level = std::max(1, src_a.getLevel());

    std::unordered_map<int, std::unique_ptr<FIDESlib::CKKS::Plaintext>> mask_cache;
    mask_cache.reserve(static_cast<size_t>(needed_slots));

    auto get_mask = [&](int slot_idx) -> FIDESlib::CKKS::Plaintext& {
        auto it = mask_cache.find(slot_idx);
        if (it == mask_cache.end()) {
            auto pt = std::make_unique<FIDESlib::CKKS::Plaintext>(make_one_hot_plaintext(cc, slot_idx, max_slots, mask_level));
            it = mask_cache.emplace(slot_idx, std::move(pt)).first;
        }
        return *(it->second);
    };

    FIDESlib::CKKS::Ciphertext out_ct(gctx);
    bool out_init = false;

    for (int row = 0; row < row_size; ++row) {
        for (int col = 0; col < row_size; ++col) {
            const int out_idx = row * row_size + col;

            FIDESlib::CKKS::Ciphertext acc(gctx);
            bool acc_init = false;

            for (int inner = 0; inner < row_size; ++inner) {
                const int idx_a = row * row_size + inner;
                const int idx_b = inner * row_size + col;

                FIDESlib::CKKS::Ciphertext ct_a(gctx);
                ct_a.multPt(src_a, get_mask(idx_a), false);
                const int shift_align = (idx_b - idx_a) * rotate_sign;
                if (shift_align != 0) {
                    ct_a.rotate(shift_align);
                }

                FIDESlib::CKKS::Plaintext pt_b(gctx);
                pt_b.copy(src_b);
                pt_b.multPt(get_mask(idx_b), false);

                FIDESlib::CKKS::Ciphertext term(gctx);
                term.multPt(ct_a, pt_b, false);

                const int shift_out = (out_idx - idx_b) * rotate_sign;
                if (shift_out != 0) {
                    term.rotate(shift_out);
                }

                if (!acc_init) {
                    acc.copy(term);
                    acc_init = true;
                } else {
                    acc.add(term);
                }
            }

            if (!out_init) {
                out_ct.copy(acc);
                out_init = true;
            } else {
                out_ct.add(acc);
            }
        }
    }

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> product(1);
    product[0].push_back(std::move(out_ct));
    return wrap_product(std::move(product), mat1.tmpl_cpu);
}

CiphertextMatrixGPU ccmm_custom_bias_proto_gpu(
        FidelCCPtr cc,
        CiphertextMatrixGPU& mat1,
        CiphertextMatrixGPU& mat2,
        PlaintextMatrixGPU& bias,
        int row_size,
        int max_slots,
        int rotate_sign) {
    if (bias.data.size() != 1 || bias.data[0].size() != 1) {
        throw std::runtime_error("ccmm_gpu_custom_bias_proto currently supports only 1x1 bias matrix container");
    }
    auto out = ccmm_custom_proto_gpu(cc, mat1, mat2, row_size, max_slots, rotate_sign);
    FIDESlib::CKKS::Plaintext bias_copy(gpu_ctx(cc));
    bias_copy.copy(bias.data[0][0]);
    bias_copy.dropToLevel(out.data[0][0].getLevel());
    FIDESlib::CKKS::Plaintext mask = make_block_mask_plaintext(
        cc,
        row_size * row_size,
        max_slots,
        out.data[0][0].getLevel());
    bias_copy.multPt(mask, false);
    out.data[0][0].addPt(bias_copy);
    return out;
}

CiphertextMatrixGPU cpmm_custom_bias_proto_gpu(
        FidelCCPtr cc,
        CiphertextMatrixGPU& mat1,
        PlaintextMatrixGPU& mat2,
        PlaintextMatrixGPU& bias,
        int row_size,
        int max_slots,
        int rotate_sign) {
    if (bias.data.size() != 1 || bias.data[0].size() != 1) {
        throw std::runtime_error("cpmm_gpu_custom_bias_proto currently supports only 1x1 bias matrix container");
    }
    auto out = cpmm_custom_proto_gpu(cc, mat1, mat2, row_size, max_slots, rotate_sign);
    FIDESlib::CKKS::Plaintext bias_copy(gpu_ctx(cc));
    bias_copy.copy(bias.data[0][0]);
    bias_copy.dropToLevel(out.data[0][0].getLevel());
    FIDESlib::CKKS::Plaintext mask = make_block_mask_plaintext(
        cc,
        row_size * row_size,
        max_slots,
        out.data[0][0].getLevel());
    bias_copy.multPt(mask, false);
    out.data[0][0].addPt(bias_copy);
    return out;
}

}

PlaintextMatrixGPU encodeMatrixGPU_fromMatrix(
        const std::vector<std::vector<double>>& inputs,
        FidelPKPtr pk,
        FidelCCPtr cc,
        int numSlots, int blockSize,
        int level, bool if_repeat, int colSize = 0)
{
    if (colSize == 0) colSize = blockSize;

    FIDESlib::CKKS::Context& gctx = gpu_ctx(cc);
    auto& context = cpu_ctx(cc);

    auto inputs_temp = FIDESlib::CKKS::extractAndLinearizeMatrix(
            inputs, numSlots, blockSize, colSize);

    if (!if_repeat) {
        for (auto& i : inputs_temp)
            for (auto& j : i)
                j = FIDESlib::CKKS::getPCMM_bMatrix(j, blockSize);
    }

    auto pt_inputs = FIDESlib::CKKS::EncodeMatrix(
            inputs_temp, cpu_pk(pk), gctx->L - level);

    PlaintextMatrixGPU result;
    result.data.resize(pt_inputs.size());
    for (size_t i = 0; i < pt_inputs.size(); ++i) {
        result.data[i].reserve(pt_inputs[0].size());
        for (size_t j = 0; j < pt_inputs[0].size(); ++j) {
            auto raw_pt = FIDESlib::CKKS::GetRawPlainText(context, pt_inputs[i][j]);
            result.data[i].emplace_back(gctx, raw_pt);
        }
    }
    return result;
}

void init_matmul_bindings(py::module_& m) {
    using GPrecomp  = FIDESlib::CKKS::MatrixMatrixProductPrecomputations_GPU;
    using TPrecomp  = FIDESlib::CKKS::TransposePrecomputations_GPU;

    m.def("encode_matrix_gpu",
      [](FidelCCPtr cc, FidelPKPtr pk, const std::vector<std::vector<double>>& mat,
         int numSlots, int blockSize, int level, bool if_repeat, int colSize) {
          return encodeMatrixGPU_fromMatrix(mat, pk, cc, numSlots, blockSize,
                                           level, if_repeat, colSize);
      },
      py::arg("cc"), py::arg("pk"), py::arg("mat"), py::arg("num_slots"),
      py::arg("block_size"), py::arg("level"),
      py::arg("if_repeat") = false, py::arg("col_size") = 0,
      py::return_value_policy::move);

    py::class_<TPrecomp>(m, "TransposePrecompGPU")
        .def_readonly("row_size", &TPrecomp::rowSize)
        .def_readonly("bstep",    &TPrecomp::bStep);

    py::class_<FIDESlib::CKKS::PtMasks_GPU>(m, "PtMasksGPU")
        .def("get_row_mask", [](FIDESlib::CKKS::PtMasks_GPU& masks, FidelCCPtr cc, size_t i) -> FidelPTPtr {
            FIDESlib::CKKS::Context& gctx = gpu_ctx(cc);
            auto sp = std::make_shared<FIDESlib::CKKS::Plaintext>(gctx);
            sp->copy(masks.row_masks.at(i));
            uint32_t handle = cc->RegisterDevicePlaintext(std::static_pointer_cast<void>(std::move(sp)));
            auto wrapper = std::make_shared<fideslib::PlaintextImpl>();
            wrapper->gpu    = handle;
            wrapper->loaded = true;
            return FidelPTPtr(wrapper);
        }, py::arg("cc"), py::arg("i"))
        .def("__len__", [](const FIDESlib::CKKS::PtMasks_GPU& masks) {
            return masks.row_masks.size();
        });

    py::class_<GPrecomp>(m, "MatMulPrecompGPU")
        .def_readonly("row_size", &GPrecomp::rowSize)
        .def_readonly("bstep",    &GPrecomp::bStep);

    py::class_<CiphertextMatrixGPU>(m, "CiphertextMatrixGPU")
        .def(py::init([](FidelCCPtr cc, py::list rows) {
            CiphertextMatrixGPU m;
            FIDESlib::CKKS::Context& gctx = gpu_ctx(cc);
            m.tmpl_cpu = rows[0].cast<py::list>()[0].cast<FidelCTPtr>()->cpu;
            m.data.reserve(rows.size());
            for (auto& row_obj : rows) {
                py::list row = row_obj.cast<py::list>();
                std::vector<FIDESlib::CKKS::Ciphertext> row_vec;
                row_vec.reserve(row.size());
                for (auto& ct_obj : row) {
                    FidelCTPtr ct = ct_obj.cast<FidelCTPtr>();
                    FIDESlib::CKKS::Ciphertext new_ct(gctx);
                    new_ct.copy(raw_gct(cc, ct));
                    row_vec.push_back(std::move(new_ct));
                }
                m.data.push_back(std::move(row_vec));
            }
            return m;
        }), py::arg("cc"), py::arg("rows"))

        .def("to_list", [](CiphertextMatrixGPU& self, FidelCCPtr cc) -> py::list {
            FIDESlib::CKKS::Context& gctx = gpu_ctx(cc);
            py::list outer;
            for (auto& row : self.data) {
                py::list inner;
                for (auto& ct : row) {
                    FIDESlib::CKKS::Ciphertext copy_ct(gctx);
                    copy_ct.copy(ct);
                    auto sp = std::make_shared<FIDESlib::CKKS::Ciphertext>(std::move(copy_ct));
                    uint32_t handle = cc->RegisterDeviceCiphertext(
                        std::static_pointer_cast<void>(std::move(sp)));
                    auto wrapper = std::make_shared<FidelCT>(FidelCCPtr(cc));
                    wrapper->gpu    = handle;
                    wrapper->loaded = true;
                    wrapper->cpu    = self.tmpl_cpu;
                    inner.append(FidelCTPtr(wrapper));
                }
                outer.append(inner);
            }
            return outer;
        }, py::arg("cc"))
        .def("__len__", [](const CiphertextMatrixGPU& self) {
            return self.data.size();
        });

    py::class_<PlaintextMatrixGPU>(m, "PlaintextMatrixGPU")
        .def(py::init([](FidelCCPtr cc, py::list rows) {
            PlaintextMatrixGPU m;
            FIDESlib::CKKS::Context& gctx = gpu_ctx(cc);
            m.data.reserve(rows.size());
            for (auto& row_obj : rows) {
                py::list row = row_obj.cast<py::list>();
                std::vector<FIDESlib::CKKS::Plaintext> row_vec;
                row_vec.reserve(row.size());
                for (auto& pt_obj : row) {
                    FidelPTPtr pt = pt_obj.cast<FidelPTPtr>();
                    FIDESlib::CKKS::Plaintext new_pt(gctx);
                    new_pt.copy(raw_gpt(cc, pt));
                    row_vec.push_back(std::move(new_pt));
                }
                m.data.push_back(std::move(row_vec));
            }
            return m;
        }), py::arg("cc"), py::arg("rows"))
        .def("__len__", [](const PlaintextMatrixGPU& self) {
            return self.data.size();
        });


    m.def("generate_transpose_rotation_indices_gpu",
          [](int row_size, int bstep) {
              return FIDESlib::CKKS::GenerateTransposeRotationIndices_GPU(row_size, bstep);
          },
          py::arg("row_size"), py::arg("bstep"));

    m.def("get_transpose_precomp_gpu",
          [](FidelCCPtr cc, int row_size, int bstep, int level) -> TPrecomp {
              return FIDESlib::CKKS::getMatrixTransposePrecomputations_GPU(
                  gpu_ctx(cc), cpu_ctx(cc), row_size, bstep, level);
          },
          py::arg("cc"), py::arg("row_size"), py::arg("bstep"), py::arg("level"),
          py::return_value_policy::move);

    m.def("matrix_transpose_gpu",
          [](FidelCCPtr cc, CiphertextMatrixGPU& mat, int row_size,
             const TPrecomp& precomp) -> CiphertextMatrixGPU {
              CiphertextMatrixGPU out;
              out.tmpl_cpu = mat.tmpl_cpu;
              out.data = FIDESlib::CKKS::MatrixTranspose_GPU(
                  std::move(mat.data), (uint32_t)row_size, precomp);
              return out;
          },
          py::arg("cc"), py::arg("mat"), py::arg("row_size"), py::arg("precomp"),
          py::return_value_policy::move);

    m.def("drop_matrix_level",
          [](CiphertextMatrixGPU& mat, int level) {
              FIDESlib::CKKS::dropMatrixLevel(mat.data, level);
          },
          py::arg("mat"), py::arg("level"));

    m.def("get_pt_masks_gpu",
          [](FidelCCPtr cc, int num_slots, int block_size, int level) -> FIDESlib::CKKS::PtMasks_GPU {
              return FIDESlib::CKKS::GetPtMasks_GPU(gpu_ctx(cc), cpu_ctx(cc), num_slots, block_size, level);
          },
          py::arg("cc"),
          py::arg("num_slots"),
          py::arg("block_size"),
          py::arg("level") = 1,
          py::return_value_policy::move);

    m.def("extract_and_linearize_matrix",
          [](const std::vector<std::vector<double>>& matrix,
             size_t num_slots,
             size_t row_size,
             size_t col_size) {
              return FIDESlib::CKKS::extractAndLinearizeMatrix(matrix, num_slots, row_size, col_size);
          },
          py::arg("matrix"),
          py::arg("num_slots"),
          py::arg("row_size"),
          py::arg("col_size") = 0);

    m.def("get_pcmm_b_matrix",
          [](const std::vector<double>& weights, int row_size) {
              return FIDESlib::CKKS::getPCMM_bMatrix(weights, row_size);
          },
          py::arg("weights"),
          py::arg("row_size"));

    m.def("generate_matmul_rotation_indices_gpu",
          [](int row_size, int bstep, int col_size) {
              return FIDESlib::CKKS::GenerateMatMulRotationIndices_GPU(
                  row_size, bstep, col_size);
          },
          py::arg("row_size"), py::arg("bstep"), py::arg("col_size") = 0);

    m.def("get_matmul_precomp_gpu",
          [](FidelCCPtr cc, int row_size, int bstep,
             int level_cp, int level_cc,
             bool fuse_boot_prescale, int slots) -> GPrecomp {
              return FIDESlib::CKKS::getMatrixMatrixProductPrecomputations_GPU(
                  gpu_ctx(cc), cpu_ctx(cc),
                  row_size, bstep, level_cp, level_cc, fuse_boot_prescale, slots);
          },
          py::arg("cc"),
          py::arg("row_size"),
          py::arg("bstep"),
          py::arg("level_cp"),
          py::arg("level_cc"),
          py::arg("fuse_boot_prescale") = false,
          py::arg("slots") = 0,
          py::return_value_policy::move);

    m.def("get_matmul_precomp_single_level_gpu",
          [](FidelCCPtr cc, int row_size, int bstep,
             int level,
             bool fuse_boot_prescale, int slots) -> GPrecomp {
              return FIDESlib::CKKS::getMatrixMatrixProductPrecomputations_GPU(
                  gpu_ctx(cc), cpu_ctx(cc),
                  row_size, bstep, level, level, fuse_boot_prescale, slots);
          },
          py::arg("cc"),
          py::arg("row_size"),
          py::arg("bstep"),
          py::arg("level"),
          py::arg("fuse_boot_prescale") = false,
          py::arg("slots") = 0,
          py::return_value_policy::move);


    m.def("ccmm_gpu",
          [](FidelCCPtr cc,
             CiphertextMatrixGPU& mat1,
             CiphertextMatrixGPU& mat2,
             int row_size,
             const GPrecomp& precomp) -> CiphertextMatrixGPU {
              validate_ccmm_preconditions(mat1.data, mat2.data, row_size, precomp);
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> product;
              FIDESlib::CKKS::CCMM_GPU(mat1.data, mat2.data,
                                        (uint32_t)row_size, product, precomp);
              return wrap_product(std::move(product), mat1.tmpl_cpu);
          },
          py::arg("cc"), py::arg("mat1"), py::arg("mat2"),
          py::arg("row_size"), py::arg("precomp"),
          py::return_value_policy::move);

    m.def("ccmm_gpu_custom_proto",
          [](FidelCCPtr cc,
             CiphertextMatrixGPU& mat1,
             CiphertextMatrixGPU& mat2,
             int row_size,
             int max_slots,
             int rotate_sign) -> CiphertextMatrixGPU {
              return ccmm_custom_proto_gpu(cc, mat1, mat2, row_size, max_slots, rotate_sign);
          },
          py::arg("cc"), py::arg("mat1"), py::arg("mat2"),
          py::arg("row_size"), py::arg("max_slots"), py::arg("rotate_sign") = 1,
          py::return_value_policy::move);

        m.def("ccmm_gpu_custom_bias_proto",
            [](FidelCCPtr cc,
             CiphertextMatrixGPU& mat1,
             CiphertextMatrixGPU& mat2,
             PlaintextMatrixGPU& bias,
             int row_size,
             int max_slots,
             int rotate_sign) -> CiphertextMatrixGPU {
              return ccmm_custom_bias_proto_gpu(cc, mat1, mat2, bias, row_size, max_slots, rotate_sign);
            },
            py::arg("cc"), py::arg("mat1"), py::arg("mat2"), py::arg("bias"),
            py::arg("row_size"), py::arg("max_slots"), py::arg("rotate_sign") = 1,
            py::return_value_policy::move);

        m.def("pcmm_gpu_custom_proto",
            [](FidelCCPtr cc,
             CiphertextMatrixGPU& mat1,
             PlaintextMatrixGPU& mat2,
             int row_size,
             int max_slots,
             int rotate_sign) -> CiphertextMatrixGPU {
              return cpmm_custom_proto_gpu(cc, mat1, mat2, row_size, max_slots, rotate_sign);
            },
            py::arg("cc"), py::arg("mat1"), py::arg("mat2"),
            py::arg("row_size"), py::arg("max_slots"), py::arg("rotate_sign") = 1,
            py::return_value_policy::move);

        m.def("pcmm_gpu_custom_bias_proto",
            [](FidelCCPtr cc,
             CiphertextMatrixGPU& mat1,
             PlaintextMatrixGPU& mat2,
             PlaintextMatrixGPU& bias,
             int row_size,
             int max_slots,
             int rotate_sign) -> CiphertextMatrixGPU {
              return cpmm_custom_bias_proto_gpu(cc, mat1, mat2, bias, row_size, max_slots, rotate_sign);
            },
            py::arg("cc"), py::arg("mat1"), py::arg("mat2"), py::arg("bias"),
            py::arg("row_size"), py::arg("max_slots"), py::arg("rotate_sign") = 1,
            py::return_value_policy::move);

        m.def("cpmm_gpu_custom_proto",
              [](FidelCCPtr cc,
                 CiphertextMatrixGPU& mat1,
                 PlaintextMatrixGPU& mat2,
                 int row_size,
                 int max_slots,
                 int rotate_sign) -> CiphertextMatrixGPU {
                  return cpmm_custom_proto_gpu(cc, mat1, mat2, row_size, max_slots, rotate_sign);
              },
              py::arg("cc"), py::arg("mat1"), py::arg("mat2"),
              py::arg("row_size"), py::arg("max_slots"), py::arg("rotate_sign") = 1,
              py::return_value_policy::move);

        m.def("cpmm_gpu_custom_bias_proto",
              [](FidelCCPtr cc,
                 CiphertextMatrixGPU& mat1,
                 PlaintextMatrixGPU& mat2,
                 PlaintextMatrixGPU& bias,
                 int row_size,
                 int max_slots,
                 int rotate_sign) -> CiphertextMatrixGPU {
                  return cpmm_custom_bias_proto_gpu(cc, mat1, mat2, bias, row_size, max_slots, rotate_sign);
              },
              py::arg("cc"), py::arg("mat1"), py::arg("mat2"), py::arg("bias"),
              py::arg("row_size"), py::arg("max_slots"), py::arg("rotate_sign") = 1,
              py::return_value_policy::move);

    m.def("pcmm_gpu",
          [](FidelCCPtr cc,
             CiphertextMatrixGPU& mat1,
             PlaintextMatrixGPU& mat2,
             int row_size,
             const GPrecomp& precomp,
             PlaintextMatrixGPU& bias) -> CiphertextMatrixGPU {
              validate_pcmm_preconditions(mat1.data, precomp, row_size);
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> product;
              FIDESlib::CKKS::PCMM_GPU(mat1.data, mat2.data,
                                        (uint32_t)row_size, product, precomp, bias.data);
              return wrap_product(std::move(product), mat1.tmpl_cpu);
          },
          py::arg("cc"), py::arg("mat1"), py::arg("mat2"),
          py::arg("row_size"), py::arg("precomp"), py::arg("bias"),
          py::return_value_policy::move);

    m.def("pcmm_gpu_masked",
          [](FidelCCPtr cc,
             CiphertextMatrixGPU& mat1,
             PlaintextMatrixGPU& mat2,
             int row_size,
             const GPrecomp& precomp,
             PlaintextMatrixGPU& bias,
             FidelPTPtr mask_row) -> CiphertextMatrixGPU {
              validate_pcmm_preconditions(mat1.data, precomp, row_size);
              FIDESlib::CKKS::Plaintext mask_copy(gpu_ctx(cc));
              mask_copy.copy(raw_gpt(cc, mask_row));
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> product;
              FIDESlib::CKKS::PCMM_GPU(mat1.data, mat2.data,
                                        (uint32_t)row_size, product, precomp,
                                        bias.data, mask_copy);
              return wrap_product(std::move(product), mat1.tmpl_cpu);
          },
          py::arg("cc"), py::arg("mat1"), py::arg("mat2"),
          py::arg("row_size"), py::arg("precomp"),
          py::arg("bias"), py::arg("mask_row"),
          py::return_value_policy::move);
}
