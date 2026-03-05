# CMakeLists.txt for _pyfideslib pybind11 module
#
# USAGE (standalone build — after pyFIDESlib has been installed):
#
#   cmake -S . -B build \
#     -DFIDESLIB_INSTALL_PREFIX=/path/to/fideslib/install \
#     -DOPENFHE_INSTALL_PREFIX=/path/to/openfhe/install \
#     -DPYTHON_EXECUTABLE=$(which python3)
#   cmake --build build -j$(nproc)
#   # The .so will appear in build/ — copy or link it to your Python env.
#
# USAGE (add_subdirectory from pyFIDESlib's CMakeLists.txt):
#   # In src/pyFIDESlib/CMakeLists.txt, append:
#   #   add_subdirectory(bindings)
#   # Then the 'fideslib' target and its include dirs are already available.
#
# KEY: set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF) MUST come BEFORE
#      pybind11_add_module() to prevent pybind11 from injecting -flto via
#      pybind11::lto, which causes `fatbinData` undefined symbol at import time.

cmake_minimum_required(VERSION 3.25)
cmake_policy(VERSION 3.21.3...3.30)

# ---- Determine build mode: standalone or subdirectory ----

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(PYFIDESLIB_BINDINGS_STANDALONE ON)
    message(STATUS "_pyfideslib: standalone build")
else()
    set(PYFIDESLIB_BINDINGS_STANDALONE OFF)
    message(STATUS "_pyfideslib: subdirectory build (reuses parent 'fideslib' target)")
endif()

# ---- Standalone project boilerplate ----

if(PYFIDESLIB_BINDINGS_STANDALONE)
    # Install prefix for already-built fideslib
    if(NOT DEFINED FIDESLIB_INSTALL_PREFIX)
        set(FIDESLIB_INSTALL_PREFIX "/usr/local" CACHE STRING "fideslib install prefix")
    endif()
    if(NOT DEFINED OPENFHE_INSTALL_PREFIX)
        set(OPENFHE_INSTALL_PREFIX "/usr/local" CACHE STRING "OpenFHE install prefix")
    endif()

    # Compiler: inherit from environment (must match what was used to build fideslib).
    # Override via -DCMAKE_CXX_COMPILER=... on the cmake command line if needed.

    # CUDA arch: A100 only on Leonardo.  Must use FIDESLIB_ARCH, not CMAKE_CUDA_ARCHITECTURES
    # (pyFIDESlib's CMakeLists overrides the latter from FIDESLIB_ARCH).
    if(NOT DEFINED FIDESLIB_ARCH)
        set(FIDESLIB_ARCH "80-real" CACHE STRING "CUDA arch for NVCC")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${FIDESLIB_ARCH}")

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    project(_pyfideslib LANGUAGES CXX CUDA)

    # Find OpenFHE (needed to get its include dirs for fideslib headers)
    find_package(OpenFHE 1.4.2 CONFIG EXACT REQUIRED PATHS "${OPENFHE_INSTALL_PREFIX}")
    include_directories(${OpenFHE_INCLUDE})
    include_directories(${OpenFHE_INCLUDE}/third-party/include)
    include_directories(${OpenFHE_INCLUDE}/core)
    include_directories(${OpenFHE_INCLUDE}/pke)
    include_directories(${OpenFHE_INCLUDE}/binfhe)

    # Find installed fideslib
    find_package(fideslib REQUIRED PATHS "${FIDESLIB_INSTALL_PREFIX}")
    set(FIDESLIB_TARGET fideslib::fideslib)
    set(FIDESLIB_API_DIR "${FIDESLIB_INSTALL_PREFIX}/include/fideslib")
endif()

# ---- pybind11 ----

# Prevent pybind11 from adding -flto (would break NVCC device link -> fatbinData undef)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Use pybind11 from the Python venv (no internet required on compute nodes).
# Pass -Dpybind11_DIR=<venv>/share/cmake/pybind11 on the cmake command line, or
# set PYBIND11_DIR env var, or let find_package search CMAKE_PREFIX_PATH.
find_package(pybind11 REQUIRED)

# ---- Module definition ----
# NOTE: CMAKE_INTERPROCEDURAL_OPTIMIZATION must already be OFF before this call.

set(BERT_TINY_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../examples/bert-tiny/src")

pybind11_add_module(_pyfideslib
    pyfideslib_bindings.cpp
    matmul_bindings.cu
    ${BERT_TINY_SRC}/MatMul.cu
    ${BERT_TINY_SRC}/MatMul.cpp
)

# bert-tiny headers: MatMul.cuh, MatMul.h, PolyApprox.cuh, Transpose.cuh, Inputs.cuh
target_include_directories(_pyfideslib PRIVATE "${BERT_TINY_SRC}")

# Link against fideslib (provides both the library and the API include dirs)
if(PYFIDESLIB_BINDINGS_STANDALONE)
    target_link_libraries(_pyfideslib PRIVATE ${FIDESLIB_TARGET})
    target_include_directories(_pyfideslib PRIVATE "${FIDESLIB_API_DIR}")
else()
    # Parent project already has 'fideslib' target and API_DIR variable
    target_link_libraries(_pyfideslib PRIVATE fideslib)
    target_include_directories(_pyfideslib PRIVATE "${API_DIR}")
    # MatMul.cu/MatMul.cuh include <CKKS/Context.cuh> etc. from the source tree
    target_include_directories(_pyfideslib PRIVATE "${SOURCE_DIR}")
endif()

# Disable LTO at target level too (belt-and-suspenders)
set_target_properties(_pyfideslib PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)

# Required for CUDA separable compilation when fideslib is static+CUDA
set_target_properties(_pyfideslib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

message(STATUS "_pyfideslib module configured")
