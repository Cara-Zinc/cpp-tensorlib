#include "tensor.h"  // Include the provided Tensor class
#include <algorithm>
#include <random>

namespace ts {

    // Helper function to calculate total size from shape
    size_t totalSize(const std::vector<size_t>& shape) {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    }

    // Random Tensor Creation
    template <typename T>
    Tensor<T> rand(const std::vector<size_t>& shape) {
        size_t total = totalSize(shape);
        std::vector<T> data(total);
        std::default_random_engine generator;
        std::uniform_real_distribution<T> distribution(0.0, 1.0);

        for (size_t i = 0; i < total; ++i) {
            data[i] = distribution(generator);
        }

        return Tensor(shape, typeid(T).name(), data);
    }

    // Zero Tensor Creation
    template <typename T>
    Tensor<T> zeros(const std::vector<size_t>& shape) {
        size_t total = totalSize(shape);
        std::vector<T> data(total, 0.0);

        return Tensor<T>(shape, typeid(T).name(), data);
    }

    // One Tensor Creation
    template <typename T>
    Tensor<T> ones(const std::vector<size_t>& shape) {
        size_t total = totalSize(shape);
        std::vector<T> data(total, 1.0);

        return Tensor<T>(shape, typeid(T).name(), data);
    }

    // Full Tensor Creation
    template <typename T>
    Tensor<T> full(const std::vector<size_t>& shape, T value) {
        size_t total = totalSize(shape);
        std::vector<T> data(total, static_cast<T>(value));

        return Tensor<T>(shape, "double", data);
    }

    // Identity Matrix Creation
    template <typename T>
    Tensor<T> eye(const std::vector<size_t>& shape) {
        if (shape.size() != 2 || shape[0] != shape[1]) {
            throw std::invalid_argument("Identity matrix must be square.");
        }
        
        size_t total = totalSize(shape);
        std::vector<T> data(total, 0.0);

        for (size_t i = 0; i < shape[0]; ++i) {
            data[i * shape[0] + i] = 1.0;
        }

        return Tensor<T>(shape, "double", data);
    }

    

}
