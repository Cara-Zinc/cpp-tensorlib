#include "tensor.h"
#include <algorithm>
namespace ts
{

    std::vector<size_t> calculateBroadcastShape(const std::vector<size_t> &shapeA, const std::vector<size_t> &shapeB)
    {
        std::vector<size_t> resultShape;

        // Start from the last dimension and move backwards
        auto itA = shapeA.rbegin();
        auto itB = shapeB.rbegin();
        while (itA != shapeA.rend() || itB != shapeB.rend())
        {
            // If one tensor is shorter, prepend 1 to its shape
            size_t dimA = itA != shapeA.rend() ? *itA : 1;
            size_t dimB = itB != shapeB.rend() ? *itB : 1;

            // Check for broadcast compatibility
            if (dimA != dimB && dimA != 1 && dimB != 1)
            {
                throw std::invalid_argument("Shapes are not compatible for broadcasting");
            }

            // Append the maximum dimension to the result shape
            resultShape.push_back(std::max(dimA, dimB));

            // Move to the next dimension
            if (itA != shapeA.rend())
                ++itA;
            if (itB != shapeB.rend())
                ++itB;
        }

        // Reverse to get the correct order
        std::reverse(resultShape.begin(), resultShape.end());
        return resultShape;
    }

    size_t calculateBroadcastIndex(size_t index, const std::vector<size_t> &shape, const std::vector<size_t> &stride)
    {
        size_t originalIndex = 0;
        for (size_t i = 0; i < shape.size(); ++i)
        {
            // Calculate the index in the current dimension
            size_t dimIndex = (index / stride[i]) % shape[i];

            // Accumulate the index considering the stride
            originalIndex += dimIndex * stride[i];
        }
        return originalIndex;
    }

    Tensor operator+(const Tensor &a, const Tensor &b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Shapes of the tensors must match for addition.");
        }
        // Tensor result(a.size(), a.type());
        // size_t total_size = 1;
        // for (size_t i = 0; i < a.dimension; ++i)
        // {
        //     total_size *= a.shape[i];
        // }
        // for (size_t i = 0; i < total_size; ++i)
        // {
        //     result.data_[i] = a.data_[i] + b.data_[i];
        // }
        std::vector<size_t> result_shape = calculateBroadcastShape(a.shape, b.shape);

        // Create result tensor with broadcasted shape
        Tensor result(result_shape, a.type());

        // Iterate over elements in the broadcasted shape
        for (size_t i = 0; i < result.total_size(); ++i)
        {
            size_t idx_a = calculateBroadcastIndex(i, a.shape, a.stride);
            size_t idx_b = calculateBroadcastIndex(i, b.shape, b.stride);

            // Perform the addition
            result.data_[i] = a.data_[idx_a] + b.data_[idx_b];
        }
        return result;
    }

    Tensor operator-(const Tensor &a, const Tensor &b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Shapes of the tensors must match for addition.");
        }
        // Tensor result(a.size(), a.type());
        // size_t total_size = 1;
        // for (size_t i = 0; i < a.dimension; ++i)
        // {
        //     total_size *= a.shape[i];
        // }
        // for (size_t i = 0; i < total_size; ++i)
        // {
        //     result.data_[i] = a.data_[i] + b.data_[i];
        // }
        std::vector<size_t> result_shape = calculateBroadcastShape(a.shape, b.shape);

        // Create result tensor with broadcasted shape
        Tensor result(result_shape, a.type());

        // Iterate over elements in the broadcasted shape
        for (size_t i = 0; i < result.total_size(); ++i)
        {
            size_t idx_a = calculateBroadcastIndex(i, a.shape, a.stride);
            size_t idx_b = calculateBroadcastIndex(i, b.shape, b.stride);

            // Perform the addition
            result.data_[i] = a.data_[idx_a] - b.data_[idx_b];
        }
        return result;
    }

    Tensor operator*(const Tensor &a, const Tensor &b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Shapes of the tensors must match for addition.");
        }
        // Tensor result(a.size(), a.type());
        // size_t total_size = 1;
        // for (size_t i = 0; i < a.dimension; ++i)
        // {
        //     total_size *= a.shape[i];
        // }
        // for (size_t i = 0; i < total_size; ++i)
        // {
        //     result.data_[i] = a.data_[i] + b.data_[i];
        // }
        std::vector<size_t> result_shape = calculateBroadcastShape(a.shape, b.shape);

        // Create result tensor with broadcasted shape
        Tensor result(result_shape, a.type());

        // Iterate over elements in the broadcasted shape
        for (size_t i = 0; i < result.total_size(); ++i)
        {
            size_t idx_a = calculateBroadcastIndex(i, a.shape, a.stride);
            size_t idx_b = calculateBroadcastIndex(i, b.shape, b.stride);

            // Perform the addition
            result.data_[i] = a.data_[idx_a] * b.data_[idx_b];
        }
        return result;
    }

    Tensor operator/(const Tensor &a, const Tensor &b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Shapes of the tensors must match for addition.");
        }
        // Tensor result(a.size(), a.type());
        // size_t total_size = 1;
        // for (size_t i = 0; i < a.dimension; ++i)
        // {
        //     total_size *= a.shape[i];
        // }
        // for (size_t i = 0; i < total_size; ++i)
        // {
        //     result.data_[i] = a.data_[i] + b.data_[i];
        // }
        std::vector<size_t> result_shape = calculateBroadcastShape(a.shape, b.shape);

        // Create result tensor with broadcasted shape
        Tensor result(result_shape, a.type());

        // Iterate over elements in the broadcasted shape
        for (size_t i = 0; i < result.total_size(); ++i)
        {
            size_t idx_a = calculateBroadcastIndex(i, a.shape, a.stride);
            size_t idx_b = calculateBroadcastIndex(i, b.shape, b.stride);

            // Perform the addition
            result.data_[i] = a.data_[idx_a] / b.data_[idx_b];
        }
        return result;
    }

    // 成员函数，实现Tensor加Tensor
    Tensor Tensor::add(const Tensor &other) const
    {
        return *this + other; // Reuse the operator+ for Tensor objects
    }

    // 成员函数，实现Tensor加标量
    Tensor Tensor::add(double value) const
    {
        Tensor result(shape, dtype_);
        size_t total_size = 1;
        for (size_t i = 0; i < dimension; ++i)
        {
            total_size *= shape[i];
        }
        for (size_t i = 0; i < total_size; ++i)
        {
            result.data_[i] = this->data_[i] + value;
        }
        return result;
    }

    Tensor add(const Tensor &a, const Tensor &b)
    {
        return a + b; // Reuse the operator+ for Tensor objects
    }

    Tensor add(const Tensor &a, double value)
    {
        return a.add(value); // Reuse the Tensor's member function for scalar addition
    }

    // Tensor operator-(const Tensor &a, const Tensor &b)
    // {
    //     if (a.size() != b.size())
    //     {
    //         throw std::invalid_argument("Shapes of the tensors must match for addition.");
    //     }
    //     Tensor result(a.size(), a.type());
    //     size_t total_size = 1;
    //     for (size_t i = 0; i < a.dimension; ++i)
    //     {
    //         total_size *= a.shape[i];
    //     }
    //     for (size_t i = 0; i < total_size; ++i)
    //     {
    //         result.data_[i] = a.data_[i] - b.data_[i];
    //     }
    //     return result;
    // }

    // 成员函数，实现Tensor加Tensor
    Tensor Tensor::sub(const Tensor &other) const
    {
        return *this - other; // Reuse the operator+ for Tensor objects
    }

    // 成员函数，实现Tensor加标量
    Tensor Tensor::sub(double value) const
    {
        Tensor result(shape, dtype_);
        size_t total_size = 1;
        for (size_t i = 0; i < dimension; ++i)
        {
            total_size *= shape[i];
        }
        for (size_t i = 0; i < total_size; ++i)
        {
            result.data_[i] = this->data_[i] - value;
        }
        return result;
    }

    Tensor sub(const Tensor &a, const Tensor &b)
    {
        return a - b; // Reuse the operator+ for Tensor objects
    }

    Tensor sub(const Tensor &a, double value)
    {
        return a.sub(value); // Reuse the Tensor's member function for scalar addition
    }

    Tensor Tensor::mul(const Tensor &other) const
    {
        return *this * other; // Reuse the operator+ for Tensor objects
    }

    // 成员函数，实现Tensor加标量
    Tensor Tensor::mul(double value) const
    {
        Tensor result(shape, dtype_);
        size_t total_size = 1;
        for (size_t i = 0; i < dimension; ++i)
        {
            total_size *= shape[i];
        }
        for (size_t i = 0; i < total_size; ++i)
        {
            result.data_[i] = this->data_[i] * value;
        }
        return result;
    }

    Tensor mul(const Tensor &a, const Tensor &b)
    {
        return a * b; // Reuse the operator+ for Tensor objects
    }

    Tensor mul(const Tensor &a, double value)
    {
        return a.mul(value); // Reuse the Tensor's member function for scalar addition
    }

    Tensor Tensor::div(const Tensor &other) const
    {
        return *this / other; // Reuse the operator+ for Tensor objects
    }

    // 成员函数，实现Tensor加标量
    Tensor Tensor::div(double value) const
    {
        Tensor result(shape, dtype_);
        size_t total_size = 1;
        for (size_t i = 0; i < dimension; ++i)
        {
            total_size *= shape[i];
        }
        for (size_t i = 0; i < total_size; ++i)
        {
            result.data_[i] = this->data_[i] / value;
        }
        return result;
    }

    Tensor div(const Tensor &a, const Tensor &b)
    {
        return a / b; // Reuse the operator+ for Tensor objects
    }

    Tensor div(const Tensor &a, double value)
    {
        return a.div(value); // Reuse the Tensor's member function for scalar addition
    }

    Tensor dot(const Tensor &a, const Tensor &b)
    {
        if (a.dimens() < 1 || b.dimens() < 1)
        {
            throw std::invalid_argument("Tensors must have at least 1 dimension for dot product.");
        }

        if (a.size()[a.dimens() - 1] != b.size()[0])
        {
            throw std::invalid_argument("Incompatible dimensions for dot product.");
        }

        // 确定结果张量的形状
        std::vector<size_t> result_shape;
        // 添加张量A的形状，除去最后一个维度
        for (size_t i = 0; i < a.dimens() - 1; ++i)
        {
            result_shape.push_back(a.size()[i]);
        }
        // 添加张量B的形状，除去第一个维度
        for (size_t i = 1; i < b.dimens(); ++i)
        {
            result_shape.push_back(b.size()[i]);
        }
        // 创建结果张量
        Tensor result(result_shape, a.type());

        // 计算点积
        // M = ? N = ?
        int M = a.total_size() / a.get_shape()[a.dimens() - 1];
        int N = b.total_size() / b.get_shape()[0];
        int step = b.get_shape()[0];
        int stride = b.get_stride()[0];
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; j++)
            {
                int ans = 0;
                for (int k = 0; k < step; k++)
                {
                    ans += a.get_element(i * step + k) * b.get_element(k * stride + j);
                }
                result.set_element(i * N + j, ans);
            }
        }

        return result;
    }
}
