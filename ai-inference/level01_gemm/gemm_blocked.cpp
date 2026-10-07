#include <algorithm>

#define BLOCK_SIZE 64

// Blocked (Tiled) GEMM for improved L1/L2 cache locality
void gemm_blocked(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

    for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        int imax = std::min(i0 + BLOCK_SIZE, M);
        for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            int kmax = std::min(k0 + BLOCK_SIZE, K);
            for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                int jmax = std::min(j0 + BLOCK_SIZE, N);

                for (int i = i0; i < imax; ++i) {
                    for (int k = k0; k < kmax; ++k) {
                        float a_ik = A[i * K + k];
                        for (int j = j0; j < jmax; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}