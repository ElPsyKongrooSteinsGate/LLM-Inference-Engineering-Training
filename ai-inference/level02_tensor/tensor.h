#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <numeric>
#include <initializer_list>
#include <iostream>
#include <stdexcept>

class Tensor {
public:
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<float> data;
    size_t total_elements;

    // Constructors
    Tensor();
    Tensor(std::vector<int> shape);
    Tensor(std::initializer_list<int> shape);

    // Memory initialization helpers
    void fill(float value);
    void fill_random(float min_val = -1.0f, float max_val = 1.0f);

    // Dimensions & Indexing
    int ndim() const { return static_cast<int>(shape.size()); }
    size_t size() const { return total_elements; }
    
    // Compute flat index from multi-dimensional indices
    int get_flat_index(const std::vector<int>& indices) const;

    // Accessors
    float& operator[](size_t flat_idx) { return data[flat_idx]; }
    const float& operator[](size_t flat_idx) const { return data[flat_idx]; }

    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;

    // Reshape & Utilities
    void reshape(const std::vector<int>& new_shape);
    void print_info(const std::string& name = "Tensor") const;

private:
    void compute_strides();
};

#endif // TENSOR_H