#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <variant>

namespace ts
{
    // Forward declaration
    template <typename T>
    class Tensor;

    // Function to print tensor information
    template <typename T>
    std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor);

    template <typename T>
    class Tensor
    {
    public:
        Tensor();
        Tensor(const std::vector<std::vector<T>> &data);
        Tensor(const std::vector<size_t> &shape, const std::string &dtype, T init_value = T());
        Tensor(const std::vector<size_t> &shape, const std::string &dtype, const std::vector<T> &data);
        ~Tensor();
        std::vector<size_t> size() const;
        std::vector<size_t> get_shape() const;
        std::vector<size_t> get_stride() const;
        std::string type() const;
        T *data_ptr() const;
        T get_element(size_t index) const;
        void set_element(size_t index, T value);
        size_t total_size() const;
        int dimens() const;

        // slicing
        template <typename... Args>
        Tensor<T> operator()(Args... args);
        // mutating
        void operator=(T val);

        void operator=(std::vector<T> val);

        // transpose
        Tensor<T> transpose(int dim1, int dim2);
        // permute
        Tensor<T> permute(std::vector<int> dims); // view
        Tensor<T> view(std::vector<size_t> shape);
        Tensor<T> add(const Tensor<T> &other) const;
        Tensor<T> add(T value) const;
        Tensor<T> sub(const Tensor<T> &other) const;
        Tensor<T> sub(T value) const;
        Tensor<T> mul(const Tensor<T> &other) const;
        Tensor<T> mul(T value) const;
        Tensor<T> div(const Tensor<T> &other) const;
        Tensor<T> div(T value) const;
        Tensor<bool> eq(const Tensor<T> &other) const;
        Tensor<bool> eq(T value) const;
        Tensor<bool> ne(const Tensor<T> &other) const;
        Tensor<bool> ne(T value) const;
        Tensor<bool> gt(const Tensor<T> &other) const;
        Tensor<bool> gt(T value) const;
        Tensor<bool> ge(const Tensor<T> &other) const;
        Tensor<bool> ge(T value) const;
        Tensor<bool> lt(const Tensor<T> &other) const;
        Tensor<bool> lt(T value) const;
        Tensor<bool> le(const Tensor<T> &other) const;
        Tensor<bool> le(T value) const;

        // Tensor<bool> Tensor<T>::eq(const Tensor<T> &a, const Tensor<T> &b);

        template <typename U>
        friend Tensor<U> operator+(const Tensor<U> &a, const Tensor<U> &b);

        template <typename U>
        friend Tensor<U> operator-(const Tensor<U> &a, const Tensor<U> &b);

        template <typename U>
        friend Tensor<U> operator*(const Tensor<U> &a, const Tensor<U> &b);

        template <typename U>
        friend Tensor<U> operator/(const Tensor<U> &a, const Tensor<U> &b);

        template <typename U>
        friend Tensor<bool> operator==(const Tensor<T> &a, const Tensor<T> &b);

        template <typename U>
        friend Tensor<bool> operator!=(const Tensor<T> &a, const Tensor<T> &b);

        template <typename U>
        friend Tensor<bool> operator>(const Tensor<T> &a, const Tensor<T> &b);

        template <typename U>
        friend Tensor<bool> operator>=(const Tensor<T> &a, const Tensor<T> &b);

        template <typename U>
        friend Tensor<bool> operator<(const Tensor<T> &a, const Tensor<T> &b);

        template <typename U>
        friend Tensor<bool> operator<=(const Tensor<T> &a, const Tensor<T> &b);

        // Other member functions for tensor operations, indexing, slicing, etc.
        Tensor<T> sum(int dim) const;
        Tensor<T> mean(int dim) const;
        Tensor<T> max(int dim) const;
        Tensor<T> min(int dim) const;

    private:
        T *data_;
        int dimension;             // the number of dimensions this tensor has
        std::vector<size_t> shape; // shape of the tensor, storing the length of every dimension of the tensor
        std::string dtype_;
        std::vector<size_t> offset; // the shift between the start of the tensor to tensor->data
        std::vector<size_t> stride; // store the stride of every dimension of the tensor
        std::vector<int> data_pos;  // for get_element
    };

    // // Other utility functions or global operator overloads

    template <typename T>
    Tensor<T> dot(const Tensor<T> &a, const Tensor<T> &b);

    template <>
    class Tensor<bool>
    {

    public:
        Tensor();
        Tensor(const std::vector<std::vector<bool>> &data);
        Tensor(const std::vector<size_t> &shape, const std::string &dtype, bool init_value = false);
        Tensor(const std::vector<size_t> &shape, const std::string &dtype, std::vector<bool> data_vector);
        ~Tensor();

        std::vector<size_t> size() const;
        std::vector<size_t> get_shape() const;
        std::vector<size_t> get_stride() const;
        std::string type() const;
        bool *data_ptr() const;
        bool get_element(size_t index) const;
        void set_element(size_t index, bool value);
        size_t total_size() const;
        int dimens() const;

    private:
        bool *data_;
        int dimension;             // the number of dimensions this tensor has
        std::vector<size_t> shape; // shape of the tensor, storing the length of every dimension of the tensor
        std::string dtype_;
        std::vector<size_t> offset; // the shift between the start of the tensor to tensor->data
        std::vector<size_t> stride;
    };

}

// Include the implementation file here (e.g., in a .cpp file)
#include "tensor_operation.hpp"
#include "tensor_creation.hpp"
#include "math_operation.hpp"
#include "reduction_operation.hpp"
#include "comparison_operation.hpp"
#include "einsum.hpp"