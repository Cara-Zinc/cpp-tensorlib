#include "tensor.h"

namespace ts {

    Tensor operator+(const Tensor& a, const Tensor& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Shapes of the tensors must match for addition.");
        }
        Tensor result(a.size(), a.type());
        size_t total_size = 1;
        for (size_t i = 0; i < a.dimension; ++i) {
            total_size *= a.shape[i];
        }
        for (size_t i = 0; i < total_size; ++i) {
            result.data_[i] = a.data_[i] + b.data_[i];
        }
        return result;
    }

    // 成员函数，实现Tensor加Tensor
    Tensor Tensor::add(const Tensor& other) const {
        return *this + other; // Reuse the operator+ for Tensor objects
    }

    // 成员函数，实现Tensor加标量
    Tensor Tensor::add(double value) const {
        Tensor result(shape, dtype_);
        size_t total_size = 1;
        for (size_t i = 0; i < dimension; ++i) {
            total_size *= shape[i];
        }
        for (size_t i = 0; i < total_size; ++i) {
            result.data_[i] = this->data_[i] + value;
        }
        return result;
    }

    Tensor add(const Tensor& a, const Tensor& b) {
        return a + b; // Reuse the operator+ for Tensor objects
    }

    Tensor add(const Tensor& a, double value) {
        return a.add(value); // Reuse the Tensor's member function for scalar addition
    }

    Tensor operator-(const Tensor& a, const Tensor& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Shapes of the tensors must match for addition.");
        }
        Tensor result(a.size(), a.type());
        size_t total_size = 1;
        for (size_t i = 0; i < a.dimension; ++i) {
            total_size *= a.shape[i];
        }
        for (size_t i = 0; i < total_size; ++i) {
            result.data_[i] = a.data_[i] - b.data_[i];
        }
        return result;
    }

    // 成员函数，实现Tensor加Tensor
    Tensor Tensor::sub(const Tensor& other) const {
        return *this - other; // Reuse the operator+ for Tensor objects
    }

    // 成员函数，实现Tensor加标量
    Tensor Tensor::sub(double value) const {
        Tensor result(shape, dtype_);
        size_t total_size = 1;
        for (size_t i = 0; i < dimension; ++i) {
            total_size *= shape[i];
        }
        for (size_t i = 0; i < total_size; ++i) {
            result.data_[i] = this->data_[i] - value;
        }
        return result;
    }

    Tensor sub(const Tensor& a, const Tensor& b) {
        return a - b; // Reuse the operator+ for Tensor objects
    }

    Tensor sub(const Tensor& a, double value) {
        return a.sub(value); // Reuse the Tensor's member function for scalar addition
    }

}