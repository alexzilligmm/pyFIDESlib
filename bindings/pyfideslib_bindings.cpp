// pybind11 bindings for pyFIDESlib (fideslib 2.0 CPU+GPU CKKS API).
//
// Module name: _pyfideslib
// Exposes: CCParams, CryptoContext, Plaintext, Ciphertext, PublicKey, PrivateKey,
//          KeyPair, enums, and GenCryptoContext factory.
//
// CPU-only workflow (no GPU):
//   params.set_ciphertext_autoload(False)  <- avoids LoadCiphertext() call which
//   cc = gen_crypto_context(params)           requires LoadContext() to have been called
//
// GPU workflow:
//   cc.load_context(keys.public_key)       <- sets cc.loaded=True, registers GPU context
//   cc.load_ciphertext(ct)                 <- pushes ciphertext to GPU

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include <fideslib.hpp>

namespace py = pybind11;
using namespace fideslib;

// Defined in matmul_bindings.cu — GPU matrix multiplication bindings.
void init_matmul_bindings(py::module_& m);

// ---------------------------------------------------------------------------
// Type aliases for readability
// ---------------------------------------------------------------------------
using CC    = CryptoContextImpl<DCRTPoly>;
using CCPtr = CryptoContext<DCRTPoly>;
using CT    = CiphertextImpl<DCRTPoly>;
using CTPtr = Ciphertext<DCRTPoly>;
using PT    = PlaintextImpl;
using PTPtr = Plaintext;
using PK    = PublicKeyImpl<DCRTPoly>;
using PKPtr = PublicKey<DCRTPoly>;
using SK    = PrivateKeyImpl<DCRTPoly>;
using SKPtr = PrivateKey<DCRTPoly>;
using KP    = KeyPair<DCRTPoly>;
using Params = CCParams<CryptoContextCKKSRNS>;

PYBIND11_MODULE(_pyfideslib, m) {
    m.doc() = "Python bindings for pyFIDESlib (fideslib 2.0 CPU+GPU CKKS API)";

    py::enum_<PKESchemeFeature>(m, "PKESchemeFeature", py::arithmetic())
        .value("PKE",          PKE)
        .value("KEYSWITCH",    KEYSWITCH)
        .value("PRE",          PRE)
        .value("LEVELEDSHE",   LEVELEDSHE)
        .value("ADVANCEDSHE",  ADVANCEDSHE)
        .value("MULTIPARTY",   MULTIPARTY)
        .value("FHE",          FHE)
        .value("SCHEMESWITCH", SCHEMESWITCH)
        .export_values();

    py::enum_<ScalingTechnique>(m, "ScalingTechnique")
        .value("FIXEDMANUAL",     FIXEDMANUAL)
        .value("FIXEDAUTO",       FIXEDAUTO)
        .value("FLEXIBLEAUTO",    FLEXIBLEAUTO)
        .value("FLEXIBLEAUTOEXT", FLEXIBLEAUTOEXT)
        .export_values();

    py::enum_<KeySwitchTechnique>(m, "KeySwitchTechnique")
        .value("HYBRID", HYBRID)
        .export_values();

    py::enum_<SecretKeyDist>(m, "SecretKeyDist")
        .value("GAUSSIAN",            GAUSSIAN)
        .value("UNIFORM_TERNARY",     UNIFORM_TERNARY)
        .value("SPARSE_TERNARY",      SPARSE_TERNARY)
        .value("SPARSE_ENCAPSULATED", SPARSE_ENCAPSULATED)
        .export_values();

    py::enum_<SecurityLevel>(m, "SecurityLevel")
        .value("HEStd_128_classic", HEStd_128_classic)
        .value("HEStd_192_classic", HEStd_192_classic)
        .value("HEStd_256_classic", HEStd_256_classic)
        .value("HEStd_128_quantum", HEStd_128_quantum)
        .value("HEStd_192_quantum", HEStd_192_quantum)
        .value("HEStd_256_quantum", HEStd_256_quantum)
        .value("HEStd_NotSet",      HEStd_NotSet)
        .export_values();

    py::enum_<SerType>(m, "SerType")
        .value("BINARY", BINARY)
        .value("JSON",   JSON)
        .export_values();


    py::class_<DecryptResult>(m, "DecryptResult")
        .def_readonly("is_valid",       &DecryptResult::isValid)
        .def_readonly("message_length", &DecryptResult::messageLength);


    py::class_<Params>(m, "CCParams")
        .def(py::init<>())
        .def("set_multiplicative_depth", &Params::SetMultiplicativeDepth, py::arg("depth"))
        .def("set_scaling_mod_size",     &Params::SetScalingModSize,      py::arg("size"))
        .def("set_batch_size",           &Params::SetBatchSize,           py::arg("size"))
        .def("set_ring_dim",             &Params::SetRingDim,             py::arg("dim"))
        .def("set_scaling_technique",    &Params::SetScalingTechnique,    py::arg("tech"))
        .def("set_first_mod_size",       &Params::SetFirstModSize,        py::arg("size"))
        .def("set_num_large_digits",     &Params::SetNumLargeDigits,      py::arg("num_digits"))
        .def("set_digit_size",           &Params::SetDigitSize,           py::arg("size"))
        .def("set_key_switch_technique", &Params::SetKeySwitchTechnique,  py::arg("tech"))
        .def("set_secret_key_dist",      &Params::SetSecretKeyDist,       py::arg("dist"))
        .def("set_security_level",       &Params::SetSecurityLevel,       py::arg("level"))
        .def("set_devices", [](Params& p, std::vector<int> devices) {
            p.SetDevices(std::move(devices));
        }, py::arg("devices") = std::vector<int>{0})
        .def("set_plaintext_autoload",  &Params::SetPlaintextAutoload,  py::arg("autoload"))
        .def("set_ciphertext_autoload", &Params::SetCiphertextAutoload, py::arg("autoload"))
        .def("get_multiplicative_depth", &Params::GetMultiplicativeDepth)
        .def("get_batch_size",           &Params::GetBatchSize)
        .def("get_secret_key_dist",      &Params::GetSecretKeyDist);

    py::class_<PK, PKPtr>(m, "PublicKey");
    py::class_<SK, SKPtr>(m, "PrivateKey");

    py::class_<KP>(m, "KeyPair")
        .def_readwrite("public_key", &KP::publicKey)
        .def_readwrite("secret_key", &KP::secretKey);

    py::class_<PT, PTPtr>(m, "Plaintext")
        .def("set_length",           &PT::SetLength,          py::arg("length"))
        .def("set_slots",            &PT::SetSlots,           py::arg("slots"))
        .def("get_log_precision",    &PT::GetLogPrecision)
        .def("get_level",            &PT::GetLevel)
        .def("get_ckks_packed_value",&PT::GetCKKSPackedValue)
        .def("get_real_packed_value",&PT::GetRealPackedValue)
        .def("__repr__", [](const PT& pt) {
            auto vals = pt.GetRealPackedValue();
            std::string s = "Plaintext([";
            for (size_t i = 0; i < std::min(vals.size(), size_t(8)); ++i) {
                if (i) s += ", ";
                s += std::to_string(vals[i]);
            }
            if (vals.size() > 8) s += ", ...";
            s += "])";
            return s;
        });

    py::class_<CT, CTPtr>(m, "Ciphertext")
        .def("get_level",          &CT::GetLevel)
        .def("get_noise_scale_deg",&CT::GetNoiseScaleDeg)
        .def("clone",              &CT::Clone)
        .def("set_level",          &CT::SetLevel, py::arg("level"))
        .def("set_slots",          &CT::SetSlots, py::arg("slots"))
        .def_readonly("loaded",    &CT::loaded);

    py::class_<CC, CCPtr>(m, "CryptoContext")
        .def("enable", py::overload_cast<PKESchemeFeature>(&CC::Enable), py::arg("feature"))
        .def("enable_mask", py::overload_cast<uint32_t>(&CC::Enable),   py::arg("mask"))

        .def("get_ring_dimension",  &CC::GetRingDimension)
        .def("get_cyclotomic_order",&CC::GetCyclotomicOrder)
        .def("get_pre_scale_factor",&CC::GetPreScaleFactor, py::arg("slots"))

        .def("set_auto_load_plaintexts",  &CC::SetAutoLoadPlaintexts,  py::arg("autoload"))
        .def("set_auto_load_ciphertexts", &CC::SetAutoLoadCiphertexts, py::arg("autoload"))
        .def("set_devices",               &CC::SetDevices,             py::arg("devices"))

        .def("load_context",    &CC::LoadContext,    py::arg("public_key"))
        .def("load_plaintext",  [](CC& cc, PTPtr pt) { cc.LoadPlaintext(pt); }, py::arg("pt"))
        .def("load_ciphertext", [](CC& cc, CTPtr ct) { cc.LoadCiphertext(ct); }, py::arg("ct"))
        .def("synchronize",     &CC::Synchronize)

        .def("key_gen",             &CC::KeyGen)
        .def("eval_mult_key_gen",   [](CC& cc, SKPtr sk) { cc.EvalMultKeyGen(sk); },
             py::arg("sk"))
        .def("eval_rotate_key_gen", [](CC& cc, SKPtr sk, std::vector<int32_t> steps) {
                 cc.EvalRotateKeyGen(sk, steps);
             }, py::arg("sk"), py::arg("steps"))

        .def("eval_bootstrap_setup", [](CC& cc,
                 std::vector<uint32_t> level_budget,
                 std::vector<uint32_t> dim1,
                 uint32_t slots,
                 uint32_t correction_factor) {
             cc.EvalBootstrapSetup(level_budget, dim1, slots, correction_factor);
         }, py::arg("level_budget"), py::arg("dim1"), py::arg("slots"),
            py::arg("correction_factor") = 0)
        .def("eval_bootstrap_key_gen", [](CC& cc, SKPtr sk, uint32_t slots) {
             cc.EvalBootstrapKeyGen(sk, slots);
         }, py::arg("sk"), py::arg("slots"))

        .def("make_ckks_packed_plaintext",
             [](CC& cc, const std::vector<double>& values,
                size_t noise_scale_deg, uint32_t level, uint32_t slots) {
                 return cc.MakeCKKSPackedPlaintext(values, noise_scale_deg, level, nullptr, slots);
             },
             py::arg("value"),
             py::arg("noise_scale_deg") = 1,
             py::arg("level") = 0,
             py::arg("slots") = 0)
        .def("make_ckks_packed_plaintext_complex",
             [](CC& cc, const std::vector<std::complex<double>>& values,
                size_t noise_scale_deg, uint32_t level, uint32_t slots) {
                 return cc.MakeCKKSPackedPlaintext(values, noise_scale_deg, level, nullptr, slots);
             },
             py::arg("value"),
             py::arg("noise_scale_deg") = 1,
             py::arg("level") = 0,
             py::arg("slots") = 0)

        .def("encrypt",
             [](CC& cc, PKPtr pk, PTPtr pt) { return cc.Encrypt(pk, pt); },
             py::arg("public_key"), py::arg("pt"))
        .def("encrypt_sk",
             [](CC& cc, SKPtr sk, PTPtr pt) { return cc.Encrypt(sk, pt); },
             py::arg("secret_key"), py::arg("pt"))

        .def("decrypt",
             [](CC& cc, SKPtr sk, CTPtr ct) {
                 PTPtr result;
                 cc.Decrypt(sk, ct, &result);
                 return result;
             },
             py::arg("secret_key"), py::arg("ct"))

        .def("eval_add",
             [](CC& cc, CTPtr ct1, CTPtr ct2) { return cc.EvalAdd(ct1, ct2); },
             py::arg("ct1"), py::arg("ct2"))
        .def("eval_add_pt",
             [](CC& cc, CTPtr ct, PTPtr pt) { return cc.EvalAdd(ct, pt); },
             py::arg("ct"), py::arg("pt"))
        .def("eval_add_scalar",
             [](CC& cc, CTPtr ct, double s) { return cc.EvalAdd(ct, s); },
             py::arg("ct"), py::arg("scalar"))
        .def("eval_sub",
             [](CC& cc, CTPtr ct1, CTPtr ct2) { return cc.EvalSub(ct1, ct2); },
             py::arg("ct1"), py::arg("ct2"))
        .def("eval_sub_pt",
             [](CC& cc, CTPtr ct, PTPtr pt) { return cc.EvalSub(ct, pt); },
             py::arg("ct"), py::arg("pt"))
        .def("eval_sub_scalar",
             [](CC& cc, CTPtr ct, double s) { return cc.EvalSub(ct, s); },
             py::arg("ct"), py::arg("scalar"))
        .def("eval_mult",
             [](CC& cc, CTPtr ct1, CTPtr ct2) { return cc.EvalMult(ct1, ct2); },
             py::arg("ct1"), py::arg("ct2"))
        .def("eval_mult_pt",
             [](CC& cc, CTPtr ct, PTPtr pt) { return cc.EvalMult(ct, pt); },
             py::arg("ct"), py::arg("pt"))
        .def("eval_mult_scalar",
             [](CC& cc, CTPtr ct, double s) { return cc.EvalMult(ct, s); },
             py::arg("ct"), py::arg("scalar"))
        .def("eval_square",
             [](CC& cc, CTPtr ct) { return cc.EvalSquare(ct); },
             py::arg("ct"))
        .def("eval_negate",
             [](CC& cc, CTPtr ct) { return cc.EvalNegate(ct); },
             py::arg("ct"))
        .def("eval_rotate",
             [](CC& cc, CTPtr ct, int32_t index) { return cc.EvalRotate(ct, index); },
             py::arg("ct"), py::arg("index"))
        .def("rescale",
             [](CC& cc, CTPtr ct) { return cc.Rescale(ct); },
             py::arg("ct"))
        .def("rescale_in_place",
             [](CC& cc, CTPtr ct) { cc.RescaleInPlace(ct); },
             py::arg("ct"))
        .def("eval_bootstrap",
             [](CC& cc, CTPtr ct, uint32_t num_iter, uint32_t precision) {
                 return cc.EvalBootstrap(ct, num_iter, precision);
             },
             py::arg("ct"),
             py::arg("num_iterations") = 1,
             py::arg("precision") = 0)
        .def("accumulate_sum",
             [](CC& cc, CTPtr ct, int slots, int stride) {
                 return cc.AccumulateSum(ct, slots, stride);
             },
             py::arg("ct"), py::arg("slots"), py::arg("stride") = 1)
        .def_static("serialize_eval_mult_key",
             [](const std::string& path) {
                 std::ofstream ofs(path, std::ios::binary);
                 return CC::SerializeEvalMultKey(ofs, BINARY);
             }, py::arg("path"))
        .def_static("serialize_eval_automorphism_key",
             [](const std::string& path) {
                 std::ofstream ofs(path, std::ios::binary);
                 return CC::SerializeEvalAutomorphismKey(ofs, BINARY);
             }, py::arg("path"))
        .def("deserialize_eval_mult_key",
             [](CC& cc, const std::string& path) {
                 std::ifstream ifs(path, std::ios::binary);
                 return cc.DeserializeEvalMultKey(ifs, BINARY);
             }, py::arg("path"))
        .def("deserialize_eval_automorphism_key",
             [](CC& cc, const std::string& path) {
                 std::ifstream ifs(path, std::ios::binary);
                 return cc.DeserializeEvalAutomorphismKey(ifs, BINARY);
             }, py::arg("path"))
        .def_static("set_level", [](CTPtr ct, size_t level) {
            CC::SetLevel(ct, level);
        }, py::arg("ct"), py::arg("level"));

    m.def("gen_crypto_context", &GenCryptoContext, py::arg("params"),
          "Create a CryptoContext from CCParams.");

    init_matmul_bindings(m);

    m.def("serialize_crypto_context",
          [](const std::string& path, CCPtr cc) {
              return Serial::SerializeToFile(path, cc, BINARY);
          }, py::arg("path"), py::arg("cc"));
    m.def("serialize_public_key",
          [](const std::string& path, PKPtr pk) {
              return Serial::SerializeToFile(path, pk, BINARY);
          }, py::arg("path"), py::arg("pk"));
    m.def("serialize_private_key",
          [](const std::string& path, SKPtr sk) {
              return Serial::SerializeToFile(path, sk, BINARY);
          }, py::arg("path"), py::arg("sk"));
    m.def("deserialize_crypto_context",
          [](const std::string& path, CCPtr& cc) {
              return Serial::DeserializeFromFile(path, cc, BINARY);
          }, py::arg("path"), py::arg("cc"));
    m.def("deserialize_public_key",
          [](const std::string& path, PKPtr& pk) {
              return Serial::DeserializeFromFile(path, pk, BINARY);
          }, py::arg("path"), py::arg("pk"));
    m.def("deserialize_private_key",
          [](const std::string& path, SKPtr& sk) {
              return Serial::DeserializeFromFile(path, sk, BINARY);
          }, py::arg("path"), py::arg("sk"));
}
