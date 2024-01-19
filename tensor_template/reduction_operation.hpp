#ifndef REDUCTION_OPERATION_HPP
#define REDUCTION_OPERATION_HPP
#include "tensor.h"

namespace ts
{
    template <typename T>
    Tensor<T> sum(const Tensor<T> &t, int dim)
    {
        int group_num = 1;
        std::vector<size_t> new_shape;
        if (dim < 0 || dim >= t.dimens())
        {
            throw std::invalid_argument("Invalid dimension number!");
        }
        for (int i = 0; i < dim; i++)
        {
            group_num *= t.get_shape()[i];
            new_shape.push_back(t.get_shape()[i]);
        }
        for (int i = dim + 1; i < t.dimens(); i++)
        {
            new_shape.push_back(t.get_shape()[i]);
        }

        std::vector<T> new_data;
        int stride = t.get_stride()[dim];
        int group_length = t.get_stride()[0] * t.get_shape()[0] / group_num;
        int dim_len = t.get_shape()[dim];
        for (int i = 0; i < group_num; i++)
        {
            for (int j = 0; j < stride; j++)
            {
                T tmp = T();
                for (int k = 0; k < dim_len; k++)
                {
                    tmp += t.get_element(i * group_length + k * stride + j);
                }
                new_data.push_back(tmp);
            }
        }
        std::cout << "reduction: " << std::endl;
        return Tensor<T>(new_shape, typeid(T).name(), new_data);
    }

    template <typename T>
    Tensor<T> Tensor<T>::sum(int dim) const { return ts::sum(*this, dim); }

    template <typename T>
    Tensor<T> mean(const Tensor<T> &t, int dim)
    {
        int group_num = 1;
        std::vector<size_t> new_shape;
        if (dim < 0 || dim >= t.dimens())
        {
            throw std::invalid_argument("Invalid dimension number!");
        }
        for (int i = 0; i < dim; i++)
        {
            group_num *= t.get_shape()[i];
            new_shape.push_back(t.get_shape()[i]);
        }
        for (int i = dim + 1; i < t.dimens(); i++)
        {
            new_shape.push_back(t.get_shape()[i]);
        }

        std::vector<T> new_data;
        int stride = t.get_stride()[dim];
        int group_length = t.get_stride()[0] * t.get_shape()[0] / group_num;
        int dim_len = t.get_shape()[dim];
        for (int i = 0; i < group_num; i++)
        {
            for (int j = 0; j < stride; j++)
            {
                T tmp = T();
                for (int k = 0; k < dim_len; k++)
                {
                    tmp += t.get_element(i * group_length + k * stride + j);
                }
                tmp = tmp / dim_len;
                new_data.push_back(tmp);
            }
        }
        return Tensor<T>(new_shape, typeid(T).name(), new_data);
    }

    template <typename T>
    Tensor<T> Tensor<T>::mean(int dim) const { return ts::mean(*this, dim); }

    template <typename T>
    Tensor<T> max(const Tensor<T> &t, int dim)
    {
        int group_num = 1;
        std::vector<size_t> new_shape;
        if (dim < 0 || dim >= t.dimens())
        {
            throw std::invalid_argument("Invalid dimension number!");
        }
        for (int i = 0; i < dim; i++)
        {
            group_num *= t.get_shape()[i];
            new_shape.push_back(t.get_shape()[i]);
        }
        for (int i = dim + 1; i < t.dimens(); i++)
        {
            new_shape.push_back(t.get_shape()[i]);
        }

        std::vector<T> new_data;
        int stride = t.get_stride()[dim];
        int group_length = t.get_stride()[0] * t.get_shape()[0] / group_num;
        int dim_len = t.get_shape()[dim];
        for (int i = 0; i < group_num; i++)
        {
            for (int j = 0; j < stride; j++)
            {
                T tmp = t.get_element(i * group_length + j);
                for (int k = 0; k < dim_len; k++)
                {
                    if (t.get_element(i * group_length + k * stride + j) > tmp)
                    {
                        tmp = t.get_element(i * group_length + k * stride + j);
                    }
                }
                new_data.push_back(tmp);
            }
        }
        return Tensor<T>(new_shape, typeid(T).name(), new_data);
    }

    template <typename T>
    Tensor<T> Tensor<T>::max(int dim) const
    {
        return ts::max(*this, dim);
    }

    template <typename T>
    Tensor<T> min(const Tensor<T> &t, int dim)
    {
        int group_num = 1;
        std::vector<size_t> new_shape;
        if (dim < 0 || dim >= t.dimens())
        {
            throw std::invalid_argument("Invalid dimension number!");
        }
        for (int i = 0; i < dim; i++)
        {
            group_num *= t.get_shape()[i];
            new_shape.push_back(t.get_shape()[i]);
        }
        for (int i = dim + 1; i < t.dimens(); i++)
        {
            new_shape.push_back(t.get_shape()[i]);
        }

        std::vector<T> new_data;
        int stride = t.get_stride()[dim];
        int group_length = t.get_stride()[0] * t.get_shape()[0] / group_num;
        int dim_len = t.get_shape()[dim];
        for (int i = 0; i < group_num; i++)
        {
            for (int j = 0; j < stride; j++)
            {
                T tmp = t.get_element(i * group_length + j);
                for (int k = 0; k < dim_len; k++)
                {
                    if (t.get_element(i * group_length + k * stride + j) < tmp)
                    {
                        tmp = t.get_element(i * group_length + k * stride + j);
                    }
                }
                new_data.push_back(tmp);
            }
        }
        return Tensor<T>(new_shape, typeid(T).name(), new_data);
    }

    template <typename T>
    Tensor<T> Tensor<T>::min(int dim) const
    {
        return ts::min(*this, dim);
    }

}
#endif