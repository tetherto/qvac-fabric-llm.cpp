# test-coreml-template-budget: a library that embeds one toy template under
# QVAC_COREML_TEMPLATE_BUDGET_MB=0 must fail to configure (the test passes
# on that error message).
cmake_minimum_required(VERSION 3.20)
project(qvac-coreml-template-budget LANGUAGES CXX)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp" "int qvac_coreml_budget_probe() { return 0; }\n")
add_library(budget-probe STATIC "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp")
add_library(ggml INTERFACE)  # generation needs a ggml target; nothing is built
include(${CMAKE_CURRENT_SOURCE_DIR}/../../qvac_coreml.cmake)
qvac_coreml_add_to(budget-probe NAMESPACE budget_probe PREFIX BUDGET_PROBE ENABLE ON
                   TEMPLATES ${CMAKE_CURRENT_SOURCE_DIR}/../toy-templates/toy-stateless)
