#include "tensor.h"

namespace ts
{

    Tensor sum(const Tensor &t, int dim)
    {
        int group_num = 1;
        std::vector<double> new_shape;
        if (dim < 0 || dim >= t.dimens())
        {
            throw std::invalid_argument("Invalid dimension number!");
        }
        for (int i = 0; i < dim; i++)
        {
            group_num *= t.get_shape()[i];
            new_shape.push_back(t.get_shape()[i]);
        }
        //for(int i = dim; i < )
        std::vector<double> new_data;
        int stride = t.get_stride()[dim];
        int group_length = t.get_stride()[0] * t.get_shape()[0] / group_num;
        int dim_len = t.get_shape()[dim];
        for (int i = 0; i < group_num; i++)
        {
            for (int j = 0; j < stride; j++)
            {
                int tmp = 0;
                for(int k =0;k<dim_len;k++)
                {
                    tmp += t.get_element(i*group_length+k*stride+j);
                }
                new_data.push_back(tmp);
            }
        }
        //return Tensor()
    }

    Tensor Tensor::sum(int dim) const
    {
    }

    Tensor mean(const Tensor &t, int dim)
    {
    }

    Tensor Tensor::mean(int dim) const
    {
    }

    Tensor max(const Tensor &t, int dim)
    {
    }

    Tensor Tensor::max(int dim) const
    {
    }

    Tensor min(const Tensor &t, int dim)
    {
    }

    Tensor Tensor::min(int dim) const
    {
    }
}