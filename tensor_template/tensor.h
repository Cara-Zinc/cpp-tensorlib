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
        Tensor<T> operator()(Args... args)
        {
            std::vector<std::variant<int, std::vector<int>>> slice_shape;
            (slice_shape.push_back(args), ...);

            // Now use slice_shape...
            std::vector<int> pos(total_size(), 1);
            // ... Initialize indices and strides based on the shape of your Tensor

            std::vector<size_t> new_shape;
            std::vector<T> new_data;

            // Iterate over the dimensions of the Tensor
            for (int i = 0; i < slice_shape.size(); ++i)
            {
                if (std::holds_alternative<int>(slice_shape[i]))
                {
                    // If the i-th element of slice_shape is an int, take the corresponding element in the i-th dimension
                    int a = std::get<int>(slice_shape[i]);
                    if (i == 0)
                    {
                        for (int j = 0; j < total_size(); ++j)
                        {
                            if (j < stride[i] * a || j >= stride[i] * (a + 1))
                            {
                                pos[j] = 0;
                            }
                        }
                    }
                    else
                    {
                        for (int j = 0; j < total_size(); ++j)
                        {
                            if (j % stride[i - 1] < stride[i] * a || j % stride[i - 1] >= stride[i] * (a + 1))
                            {
                                pos[j] = 0;
                            }
                        }
                    }
                }
                else
                {
                    // If the i-th element of slice_shape is an array, create a new dimension in the resulting Tensor
                    auto &arr = std::get<std::vector<int>>(slice_shape[i]);

                    new_shape.push_back(arr.size());

                    std::vector<int> cnt(total_size(), 0);
                    for (int k = 0; k < arr.size(); k++)
                    {
                        if (i == 0)
                        {
                            for (int j = 0; j < total_size(); ++j)
                            {
                                if (j >= stride[i] * arr[k] && j < stride[i] * (arr[k] + 1))
                                {
                                    cnt[j] = 1;
                                }
                            }
                        }
                        else
                        {
                            for (int j = 0; j < total_size(); ++j)
                            {
                                if (j % stride[i - 1] >= stride[i] * arr[k] && j % stride[i - 1] < stride[i] * (arr[k] + 1))
                                {
                                    cnt[j] = 1;
                                }
                            }
                        }
                    }
                    for (int j = 0; j < total_size(); ++j)
                    {
                        if (cnt[j] == 0)
                        {
                            pos[j] = 0;
                        }
                    }
                }
            }
            for (int i = slice_shape.size(); i < shape.size(); ++i)
            {
                new_shape.push_back(shape[i]);
            }
            if (new_shape.empty())
            {
                new_shape.push_back(0);
            }
            std::vector<size_t> new_stride;
            for (int i = new_shape.size() - 1; i >= 0; --i)
            {
                int a = 1;
                for (int j = 0; j < i; ++j)
                {
                    a = a * new_shape[new_shape.size() - j - 1];
                }
                new_stride.push_back(a);
            }
            Tensor t;
            t.shape = new_shape;
            t.dtype_ = dtype_;
            t.dimension = new_shape.size();
            t.stride = new_stride;
            t.data_ = data_;
            for (int i = 0; i < total_size(); ++i)
            {
                if (pos[i] != 0)
                {
                    t.data_pos.push_back(i);
                }
            }
            return t;
        }

        // mutating
        void operator=(T val)
        {
            for (int i = 0; i < data_pos.size(); ++i)
            {
                data_[i] = val;
            }
        }

        void operator=(std::vector<T> val)
        {
            if (val.size() != data_pos.size())
            {
                throw std::invalid_argument("the size of val is wrong");
            }
            for (int i = 0; i < data_pos.size(); ++i)
            {
                data_[i] = val[i];
            }
        }

        // transpose
        Tensor<T> transpose(int dim1, int dim2)
        {
            std::vector<size_t> new_shape;
            std::vector<size_t> new_stride;
            Tensor t;
            t.shape = new_shape;
            t.dtype_ = dtype_;
            t.dimension = new_shape.size();
            t.stride = new_stride;
            t.data_ = data_;
            for (int i = 0; i < shape.size(); ++i)
            {
                if (i == dim1)
                {
                    new_shape.push_back(shape[dim2]);
                }
                else if (i == dim2)
                {
                    new_shape.push_back(shape[dim1]);
                }
                else
                {
                    new_shape.push_back(shape[i]);
                }
            }
            for (int i = new_shape.size() - 1; i >= 0; --i)
            {
                int a = 1;
                for (int j = 0; j < i; ++j)
                {
                    a = a * new_shape[new_shape.size() - j - 1];
                }
                new_stride.push_back(a);
            }
            for (int i = 0; i < total_size(); ++i)
            {
                std::vector<int> pos;
                int a = i;
                for (int j = 0; j < shape.size(); ++j)
                {
                    int b = a / stride[j];
                    pos.push_back(b);
                    a = a - b * stride[j];
                }
                int fin = 0;
                for (int j = 0; j < shape.size(); ++j)
                {
                    if (j == dim1)
                    {
                        fin += pos[dim2] * new_stride[dim1];
                    }
                    else if (j == dim2)
                    {
                        fin += pos[dim1] * new_stride[dim2];
                    }
                    else
                    {
                        fin += pos[j] * new_stride[j];
                    }
                }
                t.data_pos.push_back(fin);
            }
            return t;
        }

        // permute
        Tensor<T> permute(std::vector<int> dims)
        {
            if (dims.size() != shape.size())
            {
                throw std::invalid_argument("the dimension of dims is wrong");
            }
            std::vector<size_t> new_shape;
            std::vector<size_t> new_stride;
            Tensor t;
            t.shape = new_shape;
            t.dtype_ = dtype_;
            t.dimension = new_shape.size();
            t.stride = new_stride;
            t.data_ = data_;
            for (int i = 0; i < shape.size(); ++i)
            {
                new_shape.push_back(shape[dims[i]]);
            }
            for (int i = new_shape.size() - 1; i >= 0; --i)
            {
                int a = 1;
                for (int j = 0; j < i; ++j)
                {
                    a = a * new_shape[new_shape.size() - j - 1];
                }
                new_stride.push_back(a);
            }
            for (int i = 0; i < total_size(); ++i)
            {
                std::vector<int> pos;
                int a = i;
                for (int j = 0; j < this->shape.size(); ++j)
                {
                    int b = a / this->stride[j];
                    pos.push_back(b);
                    a = a - b * this->stride[j];
                }
                int fin = 0;
                for (int j = 0; j < this->shape.size(); ++j)
                {
                    fin += pos[dims[j]] * new_stride[j];
                }
                t.data_pos.push_back(fin);
            }
            return t;
        }

        // view
        Tensor<T> view(std::vector<size_t> shape)
        {
            Tensor t;
            t.shape = shape;
            t.dtype_ = dtype_;
            t.dimension = shape.size();
            std::vector<size_t> new_stride;
            for (int i = shape.size() - 1; i >= 0; --i)
            {
                int a = 1;
                for (int j = 0; j < i; ++j)
                {
                    a = a * shape[shape.size() - j - 1];
                }
                new_stride.push_back(a);
            }
            t.stride = new_stride;
            t.data_ = data_;
            return t;
        }

        Tensor<T> add(const Tensor<T> &other) const;
        Tensor<T> add(T value) const;
        Tensor<T> sub(const Tensor<T> &other) const;
        Tensor<T> sub(T value) const;
        Tensor<T> mul(const Tensor<T> &other) const;
        Tensor<T> mul(T value) const;
        Tensor<T> div(const Tensor<T> &other) const;
        Tensor<T> div(T value) const;

        template <typename U>
        friend Tensor<U> operator+(const Tensor<U> &a, const Tensor<U> &b);

        template <typename U>
        friend Tensor<U> operator-(const Tensor<U> &a, const Tensor<U> &b);

        template <typename U>
        friend Tensor<U> operator*(const Tensor<U> &a, const Tensor<U> &b);

        template <typename U>
        friend Tensor<U> operator/(const Tensor<U> &a, const Tensor<U> &b);

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
    Tensor<T> cat(std::vector<Tensor<T>> Tensors, int dim);

    template <typename T>
    Tensor<T> tile(Tensor<T> tensor, std::vector<int> dims);

    template <typename T>
    Tensor<T> transpose(Tensor<T> tensor, int dim1, int dim2);

    template <typename T>
    Tensor<T> permute(Tensor<T> tensor, std::vector<int> dims);

    template <typename T>
    Tensor<T> view(Tensor<T> tensor, std::vector<size_t> shape);

    template <typename T>
    Tensor<T> add(const Tensor<T> &a, const Tensor<T> &b);

    template <typename T>
    Tensor<T> add(const Tensor<T> &a, T value);

    template <typename T>
    Tensor<T> sub(const Tensor<T> &a, const Tensor<T> &b);

    template <typename T>
    Tensor<T> sub(const Tensor<T> &a, T value);

     template <typename T>
    Tensor<T> mul(const Tensor<T> &a, const Tensor<T> &b);

    template <typename T>
    Tensor<T> mul(const Tensor<T> &a, T value);

    template <typename T>
    Tensor<T> div(const Tensor<T> &a, const Tensor<T> &b);

    template <typename T>
    Tensor<T> div(const Tensor<T> &a, T value);

    // template <typename T>
    // Tensor<T> sum(const Tensor<T> &t, int dim);

    // template <typename T>
    // Tensor<T> mean(const Tensor<T> &t, int dim);

    // template <typename T>
    // Tensor<T> max(const Tensor<T> &t, int dim);

    // template <typename T>
    // Tensor<T> min(const Tensor<T> &t, int dim);

    // template <typename T>
    // Tensor<T> dot(const Tensor<T> &a, const Tensor<T> &b);

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
#include "tensor_operation.h"