#include "tensor.h"
using namespace std;

namespace ts
{
    template <typename T>
    Tensor<T>::Tensor() : data_(nullptr), dimension(0), dtype_("undefined") {}

    template <typename T>
    Tensor<T>::Tensor(const std::vector<std::vector<T>> &data)
    {
        if (!data.empty() && !data[0].empty())
        {
            dimension = 2;
            shape = {data.size(), data[0].size()};
            stride = {shape[1], 1};
            dtype_ = typeid(T).name();
            data_ = new T[shape[0] * shape[1]];
            for (size_t i = 0; i < shape[0]; ++i)
            {
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    data_[i * stride[0] + j] = data[i][j];
                }
            }
        }
        else
        {
            throw std::invalid_argument("Tensor data cannot be empty.");
        }
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t> &shape, const std::string &dtype, T init_value)
        : shape(shape), dtype_(dtype), data_(nullptr)
    {
        if (!shape.empty())
        {
            dimension = shape.size();
            size_t total_size = 1;
            stride.resize(dimension);
            for (int i = dimension - 1; i >= 0; --i)
            {
                stride[i] = (i == dimension - 1) ? 1 : stride[i + 1] * shape[i + 1];
                total_size *= shape[i];
            }
            data_ = new T[total_size];
            std::fill_n(data_, total_size, init_value);
        }
        else
        {
            throw std::invalid_argument("Tensor shape cannot be empty.");
        }
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t> &shape, const std::string &dtype, const std::vector<T> &data_vector)
        : shape(shape), dtype_(dtype), data_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Tensor shape cannot be empty.");
        }
        dimension = shape.size();
        size_t total_size = 1;
        stride.resize(dimension);
        std::cout << "dimension: " << dimension << std::endl;
        for (int i = dimension - 1; i >= 0; --i)
        {
            stride[i] = (i == dimension - 1) ? 1 : stride[i + 1] * shape[i + 1];
            total_size *= shape[i];
            std::cout << shape[i] << std::endl;
        }

        if (data_vector.size() != total_size)
        {

            throw std::invalid_argument("1: Data vector size does not match tensor's total size.");
        }
        data_ = new T[total_size];
        std::copy(data_vector.begin(), data_vector.end(), data_);
    }

    template <typename T>
    Tensor<T>::~Tensor() {}

    // Implement other member functions...
    template <typename T>
    vector<size_t> Tensor<T>::size() const { return shape; }

    template <typename T>
    vector<size_t> Tensor<T>::get_shape() const { return shape; }

    template <typename T>
    string Tensor<T>::type() const { return dtype_; }

    template <typename T>
    T *Tensor<T>::data_ptr() const { return data_; }

    template <typename T>
    vector<size_t> Tensor<T>::get_stride() const { return stride; }

    template <typename T>
    T Tensor<T>::get_element(size_t index) const
    {
        if (index >= total_size())
        {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    template <typename T>
    void Tensor<T>::set_element(size_t index, T value)
    {
        if (index >= total_size())
        {
            throw std::out_of_range("Index out of range");
        }
        data_[index] = value;
    }
    template <typename T>
    size_t Tensor<T>::total_size() const
    {
        size_t total = 1;
        for (size_t dim_size : shape)
        {
            total *= dim_size;
        }
        return total;
    }
    template <typename T>
    int Tensor<T>::dimens() const
    {
        return dimension;
    }
    // Explicit template instantiation
    template class Tensor<int>;
    template class Tensor<float>;
    template class Tensor<double>;

    Tensor<bool>::Tensor() : data_(nullptr), dimension(0) {}
    Tensor<bool>::Tensor(const std::vector<size_t> &shape, const std::string &dtype, bool init_value) : shape(shape), dtype_(dtype)
    {
        if (!shape.empty())
        {
            dimension = shape.size();
            size_t total_size = 1;
            stride.resize(dimension);
            for (int i = dimension - 1; i >= 0; --i)
            {
                stride[i] = (i == dimension - 1) ? 1 : stride[i + 1] * shape[i + 1];
                total_size *= shape[i];
            }

            data_ = new bool[total_size];
            std::fill_n(data_, total_size, init_value);
        }
        else
        {
            throw std::invalid_argument("Tensor shape cannot be empty.");
        }
    }

    Tensor<bool>::Tensor(const std::vector<size_t> &shape, const std::string &dtype, std::vector<bool> data_vector)
        : shape(shape), dtype_(dtype)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Tensor shape cannot be empty.");
        }

        dimension = shape.size();
        size_t total_size = 1;
        stride.resize(dimension);
        for (int i = dimension - 1; i >= 0; --i)
        {
            stride[i] = (i == dimension - 1) ? 1 : stride[i + 1] * shape[i + 1];
            total_size *= shape[i];
        }

        if (data_vector.size() != total_size)
        {
            throw std::invalid_argument("2: Data vector size does not match tensor's total size.");
        }

        data_ = new bool[total_size];
        std::copy(data_vector.begin(), data_vector.end(), data_);
    }
    Tensor<bool>::~Tensor() {}

    vector<size_t> Tensor<bool>::size() const { return shape; }

    vector<size_t> Tensor<bool>::get_shape() const { return shape; }

    string Tensor<bool>::type() const { return dtype_; }

    bool *Tensor<bool>::data_ptr() const { return data_; }

    vector<size_t> Tensor<bool>::get_stride() const { return stride; }

    bool Tensor<bool>::get_element(size_t index) const
    {
        if (index >= total_size())
        {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    void Tensor<bool>::set_element(size_t index, bool value)
    {
        if (index >= total_size())
        {
            throw std::out_of_range("Index out of range");
        }
        data_[index] = value;
    }

    size_t Tensor<bool>::total_size() const
    {
        size_t total = 1;
        for (size_t dim_size : shape)
        {
            total *= dim_size;
        }
        return total;
    }

    int Tensor<bool>::dimens() const
    {
        return dimension;
    }

}