#include "tensor.h"

namespace ts
{

    Tensor sum(const Tensor &t, int dim) 
    {
        int group_length = 1;
        if( dim<0||dim>=t.dimens())
        {
            throw std::invalid_argument("Invalid dimension number!");
        }
        for(int i=0;i<dim;i++){
            group_length*=t.shape[i];
        }
        for(int i=0;i<group_length;i++)
        {
            
        }
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