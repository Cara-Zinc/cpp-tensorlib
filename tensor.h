#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

namespace ts
{

    class Tensor;

    // Function to print tensor information
    std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

    class Tensor
    {
    public:
        Tensor();
        Tensor(const std::vector<std::vector<double>> &data);
        Tensor(const std::vector<size_t> &shape, const std::string &dtype, double init_value = 0.0);
        Tensor(const std::vector<size_t> &shape, const std::string &dtype, std::vector<double>);

        std::vector<size_t> size() const;
        std::string type() const;
        double *data_ptr() const;
        Tensor operator()(size_t index) const;                                         // Indexing
        Tensor operator()(size_t start, const std::vector<size_t> &slice_shape) const; // Slicing

        // Other member functions for tensor operations, indexing, slicing, etc.
    private:
        double *data_;
        int dimenison; // the number of dimensions this tensor has
        std::vector<size_t> shape; // shape of the tensor, storing the length of every dimension of the tensor
        std::string dtype_;
        std::vector<size_t> offset; // the shift between the start of the tensor to tensor->data
        std::vector<size_t> stride; // store the stride of every dimension of the tensor
        // following is an example:
        // if the shape of a tensor is { 2 , 3 , 4 }, then the stride of a tensor is { 12 , 4 , 1 }, that is { 3*4*1 , 4*1 , 1 }
        // the ith value in the stride vector actually stores the multiplicant of all the dimension lengths that stay behind it
        // In this way, we can store a multi-dimenison tensor in an array and use stride to help us managing the tensor
    };

    // Other utility functions or global operator overloads
}
