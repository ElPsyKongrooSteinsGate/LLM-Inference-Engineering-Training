#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

inline void fill_random(float* data, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

inline bool verify_matrix(const float* ref, const float* test, size_t size, float eps = 1e-3f) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(ref[i] - test[i]) > eps) {
            std::cerr << "Mismatch at index " << i << ": Expected " << ref[i] << ", Got " << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

#endif // UTILS_H