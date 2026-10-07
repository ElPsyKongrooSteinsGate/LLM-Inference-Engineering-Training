#include <immintrin.h>
#include <omp.h>

// Multi-threaded AVX2 GEMM using OpenMP
void gemm_openmp(const float* A, const float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for collapse(1) schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0f;
        }
        for (int k = 0; k < K; ++k) {
            __m256 vec_a = _mm256_set1_ps(A[i * K + k]);
            
            int j = 0;
            for (; j <= N - 8; j += 8) {
                __m256 vec_b = _mm256_loadu_ps(&B[k * N + j]);
                __m256 vec_c = _mm256_loadu_ps(&C[i * N + j]);
                
                vec_c = _mm256_fmadd_ps(vec_a, vec_b, vec_c);
                _mm256_storeu_ps(&C[i * N + j], vec_c);
            }
            
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}