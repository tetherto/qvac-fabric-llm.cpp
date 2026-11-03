// Stands in for a CUDA runtime DLL in test-backend-loader. Built outside the test
// executable's directory so the loader can only find it through CUDA_PATH.

extern "C" __declspec(dllexport) int loader_test_dep_score(void) {
    return 5;
}
