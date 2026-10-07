#include <cstddef>

// Naive GEMM with IKJ loop ordering: C = A * B
// Matrix A: M x K, Matrix B: K x N, Matrix C: M x N
void gemm_ikj(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0f;
        }
        for (int k = 0; k < K; ++k) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}