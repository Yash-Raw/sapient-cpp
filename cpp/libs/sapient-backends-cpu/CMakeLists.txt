# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Port of crates/sapient-backends/cpu (sub-project 1a plans C/D/E). `parallel` stands in for rayon,
# `sgemm` for matrixmultiply, `cpu_features` for is_*_feature_detected!; thermal/spinpool are
# plan-E stubs until plan E lands.
find_package(Threads REQUIRED)
add_library(sapient_backends_cpu STATIC
  src/cpu_features.cpp src/parallel.cpp src/thermal.cpp src/spinpool.cpp src/sgemm.cpp src/kernels/elementwise.cpp src/kernels/softmax.cpp src/kernels/reduce.cpp src/kernels/layernorm.cpp src/kernels/rope.cpp src/kernels/attention.cpp src/kernels/conv2d.cpp src/kernels/matmul.cpp src/kernels/quant.cpp)
add_library(sapient::backends_cpu ALIAS sapient_backends_cpu)
target_include_directories(sapient_backends_cpu PUBLIC include)
target_link_libraries(sapient_backends_cpu PUBLIC sapient::core Threads::Threads)
sapient_apply_warnings(sapient_backends_cpu)

if(SAPIENT_BUILD_TESTS)
  add_executable(sapient_backends_cpu_tests
    tests/cpu_features_test.cpp tests/parallel_test.cpp tests/sgemm_test.cpp tests/elementwise_test.cpp tests/softmax_test.cpp tests/reduce_test.cpp tests/layernorm_test.cpp tests/rope_test.cpp tests/attention_test.cpp tests/conv2d_test.cpp tests/matmul_test.cpp tests/quant_test.cpp tests/golden_kernels_test.cpp)
  target_link_libraries(sapient_backends_cpu_tests PRIVATE sapient::backends_cpu sapient::testing GTest::gtest_main)
  sapient_apply_warnings(sapient_backends_cpu_tests)
  gtest_discover_tests(sapient_backends_cpu_tests)
endif()
