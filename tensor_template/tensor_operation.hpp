#ifndef TENSOR_OPERATION_HPP
#define TENSOR_OPERATION_HPP
#include <numeric>
#include "tensor.h"
namespace ts
{

    // slicing
    template <typename T>
    template <typename... Args>
    Tensor<T> Tensor<T>::operator()(Args... args)
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
    template <typename T>
    void Tensor<T>::operator=(T val)
    {
        for (int i = 0; i < data_pos.size(); ++i)
        {
            data_[i] = val;
        }
    }

    template <typename T>
    void Tensor<T>::operator=(std::vector<T> val)
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
    template <typename T>
    Tensor<T> Tensor<T>::transpose(int dim1, int dim2)
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
    template <typename T>
    Tensor<T> Tensor<T>::permute(std::vector<int> dims)
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
    template <typename T>
    Tensor<T> Tensor<T>::view(std::vector<size_t> shape)
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

    template <typename T>
    Tensor<T> cat(std::vector<ts::Tensor<T>> Tensors, int dim)
    {
        // 检查输入列表是否为空
        if (Tensors.empty())
        {
            throw std::invalid_argument("Input tensor list is empty");
        }

        // 获取输入张量的形状
        std::vector<size_t> shape = Tensors[0].size();

        // 增加指定维度的大小
        shape[dim] = 0;
        int total_size = 0;
        for (const auto &tensor : Tensors)
        {
            shape[dim] += tensor.size()[dim];
            total_size += tensor.size()[0] * tensor.get_stride()[0];
        }

        // 创建输出张量的数据容器
        std::vector<T> new_data;
        if (dim == 0)
        {
            for (const auto &tensor : Tensors)
            {
                for (int i = 0; i < tensor.size()[0] * tensor.get_stride()[0]; ++i)
                {
                    new_data.push_back(tensor.data_ptr()[i]);
                }
            }
        }
        else
        {
            int a = Tensors[0].size()[0] * Tensors[0].get_stride()[0] / Tensors[0].get_stride()[dim - 1];
            for (int j = 0; j < a; ++j)
            {
                for (const auto &tensor : Tensors)
                {
                    for (int i = 0; i < tensor.get_stride()[dim - 1]; ++i)
                    {
                        new_data.push_back(tensor.data_ptr()[i + j * tensor.get_stride()[dim - 1]]);
                    }
                }
            }
        }

        return Tensor(shape, typeid(T).name(), new_data);
    }

    template <typename T>
    Tensor<T> tile(Tensor<T> tensor, std::vector<int> dims)
    {
        if (dims.size() != tensor.size().size())
        {
            throw std::invalid_argument("the dimension of dims is wrong");
        }

        Tensor t = tensor;
        for (int i = 0; i < dims.size(); ++i)
        {
            int n = dims[i];
            if (n != 1)
            {
                std::vector<Tensor<T>> a(n, t);
                t = cat(a, i);
            }
        }
        return t;
    }

    template <typename T>
    Tensor<T> transpose(Tensor<T> tensor, int dim1, int dim2)
    {
        return tensor.transpose(dim1, dim2);
    }

    template <typename T>
    Tensor<T> permute(Tensor<T> tensor, std::vector<int> dims)
    {
        return tensor.permute(dims);
    }

    template <typename T>
    Tensor<T> view(Tensor<T> tensor, std::vector<size_t> shape)
    {
        return tensor.view(shape);
    }

}
#endif