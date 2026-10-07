#include <cblas.h>

// Hardware-accelerated GEMM using OpenBLAS
void gemm_blas(const float* A, const float* B, float* C, int M, int N, int K) {
    cblas_sgemm(
        CblasRowMajor, 
        CblasNoTrans, 
        CblasNoTrans, 
        M, N, K, 
        1.0f, 
        A, K, 
        B, N, 
        0.0f, 
        C, N
    );
}