#include <immintrin.h>

// AVX2 SIMD Vectorized GEMM (Processes 8 floats at a time)
void gemm_simd(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            __m256 vec_a = _mm256_set1_ps(A[i * K + k]);
            
            int j = 0;
            // Vector loop (step size 8 floats = 256 bits)
            for (; j <= N - 8; j += 8) {
                __m256 vec_b = _mm256_loadu_ps(&B[k * N + j]);
                __m256 vec_c = _mm256_loadu_ps(&C[i * N + j]);
                
                // C[i][j] += A[i][k] * B[k][j] using FMA
                vec_c = _mm256_fmadd_ps(vec_a, vec_b, vec_c);
                _mm256_storeu_ps(&C[i * N + j], vec_c);
            }
            
            // Scalar cleanup loop for remaining elements
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}