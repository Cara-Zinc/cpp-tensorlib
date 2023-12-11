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

        std::vector<size_t> size() const;
        std::string type() const;
        double *data_ptr() const;
        Tensor operator()(size_t index) const;                                         // Indexing
        Tensor operator()(size_t start, const std::vector<size_t> &slice_shape) const; // Slicing

        // Other member functions for tensor operations, indexing, slicing, etc.
    private:
        std::vector<size_t> shape_;
        std::string dtype_;
        double *data_;
    };

    // Other utility functions or global operator overloads
}
