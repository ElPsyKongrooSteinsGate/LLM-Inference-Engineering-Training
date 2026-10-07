#include <iostream>
#include <vector>
#include <iomanip>
#include "../common/benchmark.h"
#include "../common/utils.h"

// Declarations
void gemm_ikj(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_blocked(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_simd(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_openmp(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_blas(const float* A, const float* B, float* C, int M, int N, int K);

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    std::cout << "========================================================\n";
    std::cout << " Level 01: GEMM Benchmark (" << M << "x" << K << " * " << K << "x" << N << ")\n";
    std::cout << "========================================================\n\n";

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_ref(M * N);
    std::vector<float> C_test(M * N);

    fill_random(A.data(), A.size());
    fill_random(B.data(), B.size());

    Timer timer;
    double time_ms = 0.0;

    // 1. Reference: OpenBLAS
    timer.start();
    gemm_blas(A.data(), B.data(), C_ref.data(), M, N, K);
    time_ms = timer.stop();
    std::cout << std::left << std::setw(22) << "1. OpenBLAS (cblas):" 
              << std::setw(10) << time_ms << " ms  |  " 
              << compute_gflops(M, N, K, time_ms) << " GFLOPS\n";

    // 2. Loop IKJ
    timer.start();
    gemm_ikj(A.data(), B.data(), C_test.data(), M, N, K);
    time_ms = timer.stop();
    bool passed = verify_matrix(C_ref.data(), C_test.data(), M * N);
    std::cout << std::left << std::setw(22) << "2. IKJ Loop:" 
              << std::setw(10) << time_ms << " ms  |  " 
              << compute_gflops(M, N, K, time_ms) << " GFLOPS  [" 
              << (passed ? "PASS" : "FAIL") << "]\n";

    // 3. Blocked (Tiled)
    timer.start();
    gemm_blocked(A.data(), B.data(), C_test.data(), M, N, K);
    time_ms = timer.stop();
    passed = verify_matrix(C_ref.data(), C_test.data(), M * N);
    std::cout << std::left << std::setw(22) << "3. Cache Blocked:" 
              << std::setw(10) << time_ms << " ms  |  " 
              << compute_gflops(M, N, K, time_ms) << " GFLOPS  [" 
              << (passed ? "PASS" : "FAIL") << "]\n";

    // 4. AVX2 SIMD
    timer.start();
    gemm_simd(A.data(), B.data(), C_test.data(), M, N, K);
    time_ms = timer.stop();
    passed = verify_matrix(C_ref.data(), C_test.data(), M * N);
    std::cout << std::left << std::setw(22) << "4. AVX2 SIMD:" 
              << std::setw(10) << time_ms << " ms  |  " 
              << compute_gflops(M, N, K, time_ms) << " GFLOPS  [" 
              << (passed ? "PASS" : "FAIL") << "]\n";

    // 5. OpenMP + AVX2
    timer.start();
    gemm_openmp(A.data(), B.data(), C_test.data(), M, N, K);
    time_ms = timer.stop();
    passed = verify_matrix(C_ref.data(), C_test.data(), M * N);
    std::cout << std::left << std::setw(22) << "5. OpenMP + AVX2:" 
              << std::setw(10) << time_ms << " ms  |  " 
              << compute_gflops(M, N, K, time_ms) << " GFLOPS  [" 
              << (passed ? "PASS" : "FAIL") << "]\n";

    return 0;
}