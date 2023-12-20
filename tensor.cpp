#include "tensor.h"
using namespace std;

namespace ts
{

    Tensor::Tensor() : shape{}, dtype_{}, data_{nullptr} {}

    Tensor::Tensor(const vector<vector<double>> &data)
    {
        // Implement constructor
    }

    Tensor::Tensor(const vector<size_t> &shape, const string &dtype, double init_value)
    {
        // Implement constructor
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

    

}
