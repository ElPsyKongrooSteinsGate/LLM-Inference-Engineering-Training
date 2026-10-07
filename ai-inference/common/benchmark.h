#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <chrono>
#include <string>
#include <iostream>

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        return duration.count(); // returns milliseconds
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

// Computes GFLOPS for standard GEMM (2 * M * N * K operations)
inline double compute_gflops(int M, int N, int K, double time_ms) {
    double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return (ops / (time_ms / 1000.0)) / 1e9;
}

#endif // BENCHMARK_H