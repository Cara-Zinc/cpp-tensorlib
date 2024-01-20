#ifndef EINSUM_HPP
#define EINSUM_HPP
#include "tensor.h"

//  1) Extracting elements along diagonal, ‘ii->i’ , //
//  2) Transpose, ‘ij->ji’,
//  3) Permuate, ‘…ij->…ji’ ,
// 4) Reduce sum, ‘ij->’,
// 5) Sum along dimension, ‘ij->j’ ,
// 6) Matrix and vector mul, ‘ik, k->i’,
// 7) Matrix mul, ‘ik, kj->ij’ ,
// 8) Dot product, ‘i,i->’ , //
// 9) Pointwise mul and reduce sum, ‘ij,ij->’ ,
// 10) Outer product, ‘i,j->ij’ , //
// 11) Batch matrix mul, ‘ijk,ikl->ijl’ , //
// 12) Tensor contraction, ‘pqrs,tuqvr->pstuv’ ,
// 13) Bilinear transformation, ‘ik,jkl->ij’
namespace ts
{ // 该方法用于判断->符号并确保equation输入为‘…ij->…ji’形式
    inline bool is_permute_last_two_dims(const std::string &equation)
    {
        // 找到 '->' 分隔符
        auto arrow_pos = equation.find("->");
        if (arrow_pos == std::string::npos)
        {
            return false; // 没有找到分隔符
        }

        // 提取输入和输出部分
        std::string input_part = equation.substr(0, arrow_pos);
        std::string output_part = equation.substr(arrow_pos + 2);

        // 检查输入和输出部分的长度
        if (input_part.size() < 2 || output_part.size() < 2)
        {
            return false; // 输入或输出部分长度小于2，不符合排列操作的模式
        }

        // 检查除最后两个字符外，其他字符是否一致
        if (input_part.substr(0, input_part.size() - 2) != output_part.substr(0, output_part.size() - 2))
        {
            return false; // 除最后两个维度外，其他维度不匹配
        }

        // 检查最后两个字符是否交换
        return input_part[input_part.size() - 2] == output_part[output_part.size() - 1] &&
               input_part[input_part.size() - 1] == output_part[output_part.size() - 2];
    }

    template <typename T>
    inline T elementwise_mul_sum(const Tensor<T> &a, const Tensor<T> &b)
    {
        if (a.get_shape() != b.get_shape())
        {
            throw std::invalid_argument("Shapes of a and b must be the same for elementwise multiplication.");
        }

        T sum = 0;
        size_t total_elements = a.total_size();
        for (size_t i = 0; i < total_elements; ++i)
        {
            sum += a.get_element(i) * b.get_element(i);
        }

        return sum;
    }

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
        if (a.dimens() != b.dimens())
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
    T reduce_sum(const Tensor<T> &tensor)
    {
        T sum = 0;
        size_t total_size = tensor.total_size();

        for (size_t i = 0; i < total_size; ++i)
        {
            sum += tensor.get_element(i);
        }

        return sum;
    }

    template <typename T>
    Tensor<T> einsum(const std::string &equation, const Tensor<T> &a, const Tensor<T> &b = Tensor<T>())
    {
        // First, standardize the einsum notation
        std::string standardized_equation = standardize_einsum_notation(equation);
        // 8，点积
        if (standardized_equation == "i,i->")
        {
            return dot(a, b);
        }
        // 9 逐元素乘法再求和
        else if (standardized_equation == "ij,ij->")
        {
            T sum = elementwise_mul_sum(a, b);
            return Tensor<T>({1}, a.type(), {sum});
        }
        // 10  外积
        else if (standardized_equation == "i,j->ij")
        {

            return outer_product(a, b);
        }
        // 11 批量矩阵乘法
        else if (standardized_equation == "ijk,ikl->ijl")
        {
            return batch_mul(a, b);
        }
        // 1 对角线提取
        else if (standardized_equation == "ii->i")
        {
            return mul(a, eye<T>(a.get_shape()));
        }
        // 2 转置
        else if (standardized_equation == "ij->ji")
        {
            if (a.dimens() != 2)
            {
                throw std::invalid_argument("The dimension for transposing must be 2");
            }
            return transpose(a, 0, 1);
        }
        // 3 最后两个维度转置
        else if (is_permute_last_two_dims(standardized_equation) == true)
        {
            if (a.dimens() < 2)
            {
                throw std::invalid_argument("The dimension for transposing must be bigger than 2");
            }
            return transpose(a, a.dimens() - 2, a.dimens() - 1);
        }
        // 4 对矩阵所有元素求和
        else if (standardized_equation == "ij->")
        {
            T sum = reduce_sum(a);
            return Tensor<T>({1}, a.type(), {sum});
        }
        // 5 沿维度求和
        else if (standardized_equation == "ij->j")
        {
            if (a.dimens() != 2)
            {
                throw std::invalid_argument("The dimension for transposing must be 2");
            }
            return sum(a, 0);
        }
        // 6 矩阵和向量乘法
        else if (standardized_equation == "ij, j->i")
        {
            if (a.dimens() != 2)
            {
                throw std::invalid_argument("a is not a 2_dimension tensor");
            }
            if (b.dimens() != 1)
            {
                throw std::invalid_argument("b is not a 1_dimension scaler");
            }
            return dot(a, b);
        }
        // 7 矩阵和矩阵乘法
        else if (standardized_equation == "ij, jk->ik")
        {
            if (a.dimens() != 2 || b.dimens() != 2)
            {
                throw std::invalid_argument("a or b is not a 2_dimension matrix");
            }
            return dot(a, b);
        }
        else
        {
            throw std::invalid_argument("There is no such einsum operation");
        }
    }
}
#endif