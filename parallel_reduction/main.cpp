#include <iostream>
#include <vector>
#include <cstdlib>
#include "reduction.hpp"

int main() {
    //size_t N = 1 << 20;  // Create 2²⁰ (1,048,576) elements.
    size_t N = 1000000;  // 1M elements.
    std::vector<double> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = 1.0;  // Sum should be N

    double result = reduction_gpu(data.data(), N);
    std::cout << "GPU reduction result: " << result << std::endl;
}
