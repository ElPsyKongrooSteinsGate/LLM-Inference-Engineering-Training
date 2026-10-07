#include "tensor.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "========================================================\n";
    std::cout << " Level 02: Tensor Abstraction Validation\n";
    std::cout << "========================================================\n\n";

    // 1. Create a 4D Tensor simulating Transformer activations: [Batch, Heads, SeqLen, HeadDim]
    Tensor t1({2, 8, 128, 64});
    t1.print_info("Attention Activation");

    // Verify stride calculations for 4D layout
    assert(t1.strides[0] == 8 * 128 * 64);
    assert(t1.strides[1] == 128 * 64);
    assert(t1.strides[2] == 64);
    assert(t1.strides[3] == 1);
    std::cout << "-> 4D Stride Computation: PASSED\n";

    // 2. Test multi-dimensional indexing lookup
    t1.fill_random();
    t1.at({1, 3, 10, 5}) = 42.0f; // Flat offset = 1*65536 + 3*8192 + 10*64 + 5 = 90757
    int expected_flat_idx = 1 * t1.strides[0] + 3 * t1.strides[1] + 10 * t1.strides[2] + 5 * t1.strides[3];
    assert(t1[expected_flat_idx] == 42.0f);
    std::cout << "-> Multi-dimensional Strided Indexing: PASSED\n";

    // 3. Test reshaping (flattening batch and heads: [2, 8, 128, 64] -> [16, 128, 64])
    t1.reshape({16, 128, 64});
    t1.print_info("Reshaped Activation");
    
    // Fix: Batch 1, Head 3 maps to 1*8 + 3 = 11 in the new dimension 0
    assert(t1.at({11, 10, 5}) == 42.0f); 
    std::cout << "-> Zero-Copy Reshape & Offset Consistency: PASSED\n\n";

    std::cout << "All Level 02 Tensor tests completed successfully!\n";
    return 0;
}