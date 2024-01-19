#ifndef EINSUM_HPP
#define EINSUM_HPP
#include "tensor.h"

namespace ts
{
    inline std::string standardize_einsum_notation(const std::string &equation)
    {
        std::unordered_map<char, char> index_map;
        char current_replacement = 'i';

        // Replace each unique index with 'i', 'j', 'k', ...
        for (char c : equation)
        {
            if (isalpha(c) && index_map.find(c) == index_map.end())
            {
                index_map[c] = current_replacement++;
            }
        }

        std::string standardized_equation;
        for (char c : equation)
        {
            if (isalpha(c))
            {
                standardized_equation += index_map[c];
            }
            else
            {
                standardized_equation += c;
            }
        }

        return standardized_equation;
    }

    template <typename T>
    Tensor<T> outer_product(const Tensor<T> &a, const Tensor<T> &b)
    {
        if (a.get_shape().size() > 1 || b.get_shape().size() > 1)
        {
            throw std::invalid_argument("Tensors must have 1 dimension for outer product.");
        }
        // Calculate the shape of the result tensor by appending the shapes of a and b
        std::vector<size_t> result_shape;
        for (auto &dim : a.get_shape())
        {
            result_shape.push_back(dim);
        }
        for (auto &dim : b.get_shape())
        {
            result_shape.push_back(dim);
        }

        // Create the result tensor with the calculated shape and default initialization
        Tensor<T> result(result_shape, a.type());

        // Compute the outer product by iterating over all elements in a and b
        size_t a_total_size = a.total_size();
        size_t b_total_size = b.total_size();
        std::vector<T> result_data(a_total_size * b_total_size); // Create a flat array for the result data

        for (size_t i = 0; i < a_total_size; ++i)
        {
            for (size_t j = 0; j < b_total_size; ++j)
            {
                // The index in the result tensor is a combination of indices in a and b
                size_t index = i * b_total_size + j;
                result_data[index] = a.get_element(i) * b.get_element(j);
            }
        }

        // Create the result tensor with the result data
        return Tensor<T>(result_shape, a.type(), result_data);
    }

    template <typename T>
    Tensor<T> batch_mul(const Tensor<T> &a, const Tensor<T> &b)
    {
        auto a_shape = a.get_shape();
        auto b_shape = b.get_shape();
        if (a_shape != b_shape)
        {
            throw std::invalid_argument("The dimension of a and b must match.");
        }
        if (a.dimens() < 3 || b.dimens() < 3)
        {
            throw std::invalid_argument("Input tensors must have at least 3 dimensions for batch matrix multiplication.");
        }
        if (a_shape[a_shape.size() - 1] != b_shape[b_shape.size() - 2])
        {
            throw std::invalid_argument("The inner dimensions of a and b must match.");
        }
        int batch_num = 1;
        for (int i = 0; i < a_shape.size() - 2; i++)
        {
            if (a_shape[i] != b_shape[i])
            {
                throw std::invalid_argument("The number of batch must match");
            }
            batch_num *= a_shape[i];
        }

        std::vector<size_t> result_shape(a_shape.begin(), a_shape.end() - 2);
        result_shape.push_back(a_shape[a_shape.size() - 2]); // Rows of the first matrix
        result_shape.push_back(b_shape[b_shape.size() - 1]); // Columns of the second matrix
        Tensor<T> result(result_shape, a.type());

        int M = a_shape[a.dimens() - 2];
        int N = b_shape[b.dimens() - 1];
        int step = b_shape[b.dimens() - 2];
        int stride = b.get_stride()[b.dimens() - 2];
        int a_batch_size = a_shape[a.dimens() - 2] * a_shape[a.dimens() - 1];
        int b_batch_size = b_shape[a.dimens() - 2] * b_shape[a.dimens() - 1];
        int result_batch_size = M * N;
        for (int t = 0; t < batch_num; t++)
        {
            for (int i = 0; i < M; ++i)
            {
                for (int j = 0; j < N; j++)
                {
                    int ans = 0;
                    for (int k = 0; k < step; k++)
                    {
                        ans += a.get_element(t * a_batch_size + i * step + k) * b.get_element(t * b_batch_size + k * stride + j);
                    }
                    result.set_element(t * result_batch_size + i * N + j, ans);
                }
            }
        }
        return result;
    }

    template <typename T>
    Tensor<T> einsum(const std::string &equation, const Tensor<T> &a, const Tensor<T> &b = Tensor<T>())
    {
        // First, standardize the einsum notation
        std::string standardized_equation = standardize_einsum_notation(equation);
        if (standardized_equation == "i,i->")
        {
            return dot(a, b);
        }
        else if (standardized_equation == "i,i->i")
        {
            return mul(a, b);
        }
        else if (standardized_equation == "i,j->ij")
        {

            return outer_product(a, b);
        }
        else if (standardized_equation == "bij,bjk->bik")
        {
            return batch_mul(a,b);
        }
        else if (standardized_equation == "ii->i")
        {
            return mul(a, eye(a.get_shape()));
        } 
        else
        {
            throw std::invalid_argument("There is no such einsum operation");
        }
    }
}
#endif