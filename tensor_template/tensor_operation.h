
#include <numeric>
namespace ts
{
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
