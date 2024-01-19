#include "tensor.h"
#include <vector>
#include <iostream>

namespace ts {


    // Function to compare equality of two tensors element-wise
    template <typename T>
    Tensor<bool> eq(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Shapes must match for comparison.");
        }
        
        size_t total_size = a.total_size();
        std::vector<bool> result_data(total_size);

        for (size_t i = 0; i < total_size; ++i) {
            result_data[i] = a.get_element(i) == b.get_element(i);
        return Tensor<bool>(a.size(), result_data);
        }
    }

    // Overloading the '==' operator to use the eq function
    template <typename T>
    Tensor<bool> operator==(const Tensor<T>& a, const Tensor<T>& b) {
        return eq(a, b);
    }

    template Tensor<bool> eq(const Tensor<double>& a, const Tensor<double>& b);
    template Tensor<bool> operator==(const Tensor<double>& a, const Tensor<double>& b);
    


}