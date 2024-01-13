#include <numeric>
#include "tensor.h"

namespace ts {
    Tensor cat(std::vector<ts::Tensor> Tensors, int dim) {
        // 检查输入列表是否为空
        if (Tensors.empty()) {
            throw std::invalid_argument("Input tensor list is empty");
        }

        // 获取输入张量的形状
        std::vector<size_t> shape = Tensors[0].size();

        // 增加指定维度的大小
        shape[dim] = 0;
        int total_size = 0;
        for (const auto& tensor : Tensors) {
            shape[dim] += tensor.size()[dim];
            total_size += tensor.size()[0]*tensor.get_stride()[0];
        }

        // 创建输出张量的数据容器
        std::vector<double> new_data;
        if (dim == 0) {
            for (const auto& tensor : Tensors) {
                for (int i = 0; i < tensor.size()[0]*tensor.get_stride()[0]; ++i) {
                    new_data.push_back(tensor.data_ptr()[i]);
                }
            }
        } else {
            int a = Tensors[0].size()[0]*Tensors[0].get_stride()[0]/Tensors[0].get_stride()[dim-1];
            for (int j = 0; j < a; ++j) {
                for (const auto& tensor : Tensors) {
                    for (int i = 0; i < tensor.get_stride()[dim-1]; ++i) {
                        new_data.push_back(tensor.data_ptr()[i + j * tensor.get_stride()[dim-1]]);
                    }
                }
            }
        }

        return Tensor(shape, "double", new_data);
    }

    Tensor tile(Tensor tensor, std::vector<int> dims) {
        if (dims.size() != tensor.size().size()) {
            throw std::invalid_argument("the dimension of dims is wrong");
        }

        Tensor t = tensor;
        for (int i = 0; i < dims.size(); ++i) {
            int n = dims[i];
            if (n != 1) {
                std::vector<Tensor> a(n, t);
                t = cat(a, i);
            }

        }
        return t;
    }

    Tensor transpose(Tensor tensor, int dim1, int dim2) {
        std::vector<size_t> new_shape;
        std::vector<size_t> new_stride;
        for (int i = 0; i < tensor.size().size(); ++i) {
            if (i == dim1) {
                new_shape.push_back(dim2);
            } else if (i == dim2) {
                new_shape.push_back(dim1);
            } else {
                new_shape.push_back(i);
            }
        }
        for (int i = new_shape.size() - 1; i >= 0; --i) {
            int a = 1;
            for (int j = 0; j < i; ++j) {
                a = a * new_shape[new_shape.size() - j - 1];
            }
            new_stride.push_back(a);
        }
        std::vector<double> new_data;
        for (int i = 0; i < tensor.size()[0]*tensor.get_stride()[0]; ++i) {
            std::vector<int> pos;
            int a = i;
            for (int j = 0; j < tensor.size().size(); ++j) {
                int b = a/tensor.get_stride()[j];
                pos.push_back(b);
                a = a - b*tensor.get_stride()[j];
            }
            int fin = 0;
            for (int j = 0; j < tensor.size().size(); ++j) {
                if (j == dim1) {
                    fin += pos[dim2]*new_stride[dim1];
                } else if (j == dim2) {
                    fin += pos[dim1]*new_stride[dim2];
                } else {
                    fin += pos[j]*new_stride[j];
                }
            }
            new_data.push_back(fin);
        }
        return Tensor(new_shape, "double", new_data);
    }

    Tensor permute(Tensor tensor, std::vector<int> dims) {
        if (dims.size() != tensor.size().size()) {
            throw std::invalid_argument("the dimension of dims is wrong");
        }
        std::vector<size_t> new_shape;
        std::vector<size_t> new_stride;
        for (int i = 0; i < tensor.size().size(); ++i) {
            new_shape.push_back(dims[i]);
        }
        for (int i = new_shape.size() - 1; i >= 0; --i) {
            int a = 1;
            for (int j = 0; j < i; ++j) {
                a = a * new_shape[new_shape.size() - j - 1];
            }
            new_stride.push_back(a);
        }
        std::vector<double> new_data;
        for (int i = 0; i < tensor.size()[0]*tensor.get_stride()[0]; ++i) {
            std::vector<int> pos;
            int a = i;
            for (int j = 0; j < tensor.size().size(); ++j) {
                int b = a/tensor.get_stride()[j];
                pos.push_back(b);
                a = a - b*tensor.get_stride()[j];
            }
            int fin = 0;
            for (int j = 0; j < tensor.size().size(); ++j) {
                fin += pos[dims[j]]*new_stride[j];
            }
            new_data.push_back(fin);
        }
        return Tensor(new_shape, "double", new_data);
    }

    Tensor view(Tensor tensor, std::vector<size_t> shape) {
        std::vector<double> a(tensor.data_ptr(), tensor.data_ptr() + tensor.size()[0]*tensor.get_stride()[0]);
        return Tensor(shape, tensor.type(), a);
    }
}
