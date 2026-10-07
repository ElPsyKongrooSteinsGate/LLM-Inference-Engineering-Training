#include "tensor.h"
#include <random>
#include <iomanip>

void Tensor::compute_strides() {
    strides.resize(shape.size());
    total_elements = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = static_cast<int>(total_elements);
        total_elements *= shape[i];
    }
}

Tensor::Tensor() : total_elements(0) {}

Tensor::Tensor(std::vector<int> s) : shape(std::move(s)) {
    compute_strides();
    data.resize(total_elements, 0.0f);
}

Tensor::Tensor(std::initializer_list<int> s) : shape(s) {
    compute_strides();
    data.resize(total_elements, 0.0f);
}

void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

void Tensor::fill_random(float min_val, float max_val) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (float& val : data) {
        val = dist(gen);
    }
}

int Tensor::get_flat_index(const std::vector<int>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("Index dimension mismatch");
    }
    int flat_idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Tensor index out of bounds");
        }
        flat_idx += indices[i] * strides[i];
    }
    return flat_idx;
}

float& Tensor::at(const std::vector<int>& indices) {
    return data[get_flat_index(indices)];
}

const float& Tensor::at(const std::vector<int>& indices) const {
    return data[get_flat_index(indices)];
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    size_t new_total = 1;
    for (int dim : new_shape) {
        new_total *= dim;
    }
    if (new_total != total_elements) {
        throw std::invalid_argument("Cannot reshape: total element count must remain identical");
    }
    shape = new_shape;
    compute_strides();
}

void Tensor::print_info(const std::string& name) const {
    std::cout << name << " [Shape: (";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    std::cout << "), Strides: (";
    for (size_t i = 0; i < strides.size(); ++i) {
        std::cout << strides[i] << (i + 1 < strides.size() ? ", " : "");
    }
    std::cout << "), Total Elements: " << total_elements << "]\n";
}