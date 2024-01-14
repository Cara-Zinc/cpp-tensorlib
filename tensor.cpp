#include "tensor.h"
using namespace std;

namespace ts
{

    Tensor::Tensor() : shape{}, dtype_{}, data_{nullptr} {}

    Tensor::Tensor(const vector<vector<double>> &data)
    {
        if (!data.empty() && !data[0].empty())
        {
            dimension = 2; // assuming 2D data
            shape = {data.size(), data[0].size()};
            stride = {shape[1], 1};
            dtype_ = "double"; // set the data type
            data_ = new double[shape[0] * shape[1]];
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

    Tensor::Tensor(const vector<size_t> &shape, const string &dtype, double init_value) : shape(shape), dtype_(dtype)
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

            data_ = new double[total_size];
            std::fill_n(data_, total_size, init_value);
        }
        else
        {
            throw std::invalid_argument("Tensor shape cannot be empty.");
        }
    }
    
    Tensor::Tensor(const vector<size_t> &shape, const string &dtype, vector<double> data_vector)
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
            throw std::invalid_argument("Data vector size does not match tensor's total size.");
        }

        data_ = new double[total_size];
        std::copy(data_vector.begin(), data_vector.end(), data_);
    }

    vector<size_t> Tensor::size() const
    {
        return shape;
    }

    string Tensor::type() const
    {
        return dtype_;
    }

    double *Tensor::data_ptr() const
    {
        return data_;
    }

    vector<size_t> Tensor::get_stride() const
    {
        return stride;
    }

    double Tensor::get_element(size_t index) const {
        if (index >= total_size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    void Tensor::set_element(size_t index, double value) {
        if (index >= total_size()) {
             throw std::out_of_range("Index out of range");
         }
         data_[index] = value;
    }

    size_t Tensor::total_size() const {
        size_t total = 1;
         for (size_t dim_size : shape) {
            total *= dim_size;
        }
        return total;
     }

}