#include <any>
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fideslib.hpp>

#include "MatMul.cuh"

namespace py = pybind11;

using FidelCC    = fideslib::CryptoContextImpl<fideslib::DCRTPoly>;
using FidelCCPtr = fideslib::CryptoContext<fideslib::DCRTPoly>;
using FidelCT    = fideslib::CiphertextImpl<fideslib::DCRTPoly>;
using FidelCTPtr = fideslib::Ciphertext<fideslib::DCRTPoly>;
using FidelPTPtr = fideslib::Plaintext;


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

inline FIDESlib::CKKS::Context& gpu_ctx(const FidelCCPtr& cc) {
    return std::any_cast<FIDESlib::CKKS::Context&>(cc->gpu);
}

inline lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cpu_ctx(const FidelCCPtr& cc) {
    return std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
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

} 

void init_matmul_bindings(py::module_& m) {
    using GPrecomp = FIDESlib::CKKS::MatrixMatrixProductPrecomputations_GPU;

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


    m.def("ccmm_gpu",
          [](FidelCCPtr cc,
             CiphertextMatrixGPU& mat1,
             CiphertextMatrixGPU& mat2,
             int row_size,
             const GPrecomp& precomp) -> CiphertextMatrixGPU {
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> product;
              FIDESlib::CKKS::CCMM_GPU(mat1.data, mat2.data,
                                        (uint32_t)row_size, product, precomp);
              return wrap_product(std::move(product), mat1.tmpl_cpu);
          },
          py::arg("cc"), py::arg("mat1"), py::arg("mat2"),
          py::arg("row_size"), py::arg("precomp"),
          py::return_value_policy::move);

    m.def("pcmm_gpu",
          [](FidelCCPtr cc,
             CiphertextMatrixGPU& mat1,
             PlaintextMatrixGPU& mat2,
             int row_size,
             const GPrecomp& precomp,
             PlaintextMatrixGPU& bias) -> CiphertextMatrixGPU {
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
