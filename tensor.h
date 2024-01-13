#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <variant>

namespace ts
{

    class Tensor;

    // Function to print tensor information
    std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

    class Tensor
    {
    public:
        Tensor();
        Tensor(const std::vector<std::vector<double>> &data);
        Tensor(const std::vector<size_t> &shape, const std::string &dtype, double init_value = 0.0);
        Tensor(const std::vector<size_t> &shape, const std::string &dtype, std::vector<double>);

        std::vector<size_t> size() const;
        std::vector<size_t> get_stride() const;
        std::string type() const;
        double *data_ptr() const;
        std::vector<double*>  data_pos;//for mutating



        //slicing
        template <typename... Args>
        Tensor operator()(Args... args) {
            std::vector<std::variant<int, std::vector<int>>> slice_shape;
            (slice_shape.push_back(args), ...);

            // Now use slice_shape...
            std::vector<int> indices(shape[0]*stride[0],1);
            // ... Initialize indices and strides based on the shape of your Tensor

            std::vector<size_t> new_shape;
            std::vector<double> new_data;

            // Iterate over the dimensions of the Tensor
            for (int i = 0; i < slice_shape.size(); ++i) {
                if (std::holds_alternative<int>(slice_shape[i])) {
                    // If the i-th element of slice_shape is an int, take the corresponding element in the i-th dimension
                    int a = std::get<int>(slice_shape[i]);
                    if (i == 0) {
                        for (int j = 0; j < shape[0]*stride[0]; ++j) {
                            if (j < stride[i]*a || j >= stride[i]*(a+1)) {
                                indices[j] = 0;
                            }
                        }
                    } else {
                        for (int j = 0; j < shape[0]*stride[0]; ++j) {
                            if (j%stride[i-1] < stride[i]*a || j%stride[i-1] >= stride[i]*(a+1)) {
                                indices[j] = 0;
                            }
                        }
                    }
                } else {
                    // If the i-th element of slice_shape is an array, create a new dimension in the resulting Tensor
                    auto& arr = std::get<std::vector<int>>(slice_shape[i]);

                    new_shape.push_back(arr.size());

                    std::vector<int> cnt(shape[0]*stride[0],0);
                    for (int k = 0; k < arr.size(); k++) {
                        if (i == 0) {
                            for (int j = 0; j < shape[0]*stride[0]; ++j) {
                                if (j >= stride[i]*arr[k] && j < stride[i]*(arr[k]+1)) {
                                    cnt[j] = 1;
                                }
                            }
                        } else {
                            for (int j = 0; j < shape[0]*stride[0]; ++j) {
                                if (j%stride[i-1] >= stride[i]*arr[k] && j%stride[i-1] < stride[i]*(arr[k]+1)) {
                                    cnt[j] = 1;
                                }
                            }
                        }
                    }
                    for (int j = 0; j < shape[0]*stride[0]; ++j) {
                        if (cnt[j] == 0) {
                            indices[j] = 0;
                        }
                    }
                }
            }
            for (int i = slice_shape.size(); i < shape.size(); ++i) {
                new_shape.push_back(shape[i]);
            }
            if (new_shape.empty()) {
                new_shape.push_back(0);
            }
            if (!data_pos.empty()) {
                data_pos.clear();
            }
            for (int i = 0; i < shape[0]*stride[0]; ++i) {
                if (indices[i] != 0) {
                    new_data.push_back(data_[i]);
                    data_pos.push_back(&data_[i]);
                }
            }
            // Create a new Tensor with the new shape, stride, and data
            Tensor t = Tensor(new_shape, dtype_, new_data);
            t.data_pos = data_pos;
            return t;
        }

        //mutating
        void operator = (double val) {
            for (int i = 0; i < data_pos.size(); ++i) {
                *data_pos[i] = val;
            }
        }
        void operator = (std::vector<double> val) {
            if (val.size() != data_pos.size()) {
                throw std::invalid_argument("the size of val is wrong");
            }
            for (int i = 0; i < data_pos.size(); ++i) {
                *data_pos[i] = val[i];
            }
        }

        //transpose
        Tensor transpose(int dim1, int dim2) {
            std::vector<size_t> new_shape;
            std::vector<size_t> new_stride;
            for (int i = 0; i < this->shape.size(); ++i) {
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
            for (int i = 0; i < this->shape[0]*this->stride[0]; ++i) {
                std::vector<int> pos;
                int a = i;
                for (int j = 0; j < this->shape.size(); ++j) {
                    int b = a/this->stride[j];
                    pos.push_back(b);
                    a = a - b*this->stride[j];
                }
                int fin = 0;
                for (int j = 0; j < this->shape.size(); ++j) {
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
            return Tensor(new_shape, dtype_, new_data);
        }

        //permute
        Tensor permute(std::vector<int> dims) {
            if (dims.size() != this->shape.size()) {
                throw std::invalid_argument("the dimension of dims is wrong");
            }
            std::vector<size_t> new_shape;
            std::vector<size_t> new_stride;
            for (int i = 0; i < this->shape.size(); ++i) {
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
            for (int i = 0; i < this->shape[0]*this->stride[0]; ++i) {
                std::vector<int> pos;
                int a = i;
                for (int j = 0; j < this->shape.size(); ++j) {
                    int b = a/this->stride[j];
                    pos.push_back(b);
                    a = a - b*this->stride[j];
                }
                int fin = 0;
                for (int j = 0; j < this->shape.size(); ++j) {
                    fin += pos[dims[j]]*new_stride[j];
                }
                new_data.push_back(fin);
            }
            return Tensor(new_shape, dtype_, new_data);
        }

        //view
        Tensor view(std::vector<size_t> shape) {
            std::vector<double> a(data_, data_ + shape[0]*stride[0]);
            return Tensor(shape, dtype_, a);
        }
        // Other member functions for tensor operations, indexing, slicing, etc.
    private:
        double *data_;
        int dimenison; // the number of dimensions this tensor has
        std::vector<size_t> shape; // shape of the tensor, storing the length of every dimension of the tensor
        std::string dtype_;
        std::vector<size_t> offset; // the shift between the start of the tensor to tensor->data
        std::vector<size_t> stride; // store the stride of every dimension of the tensor
        // following is an example:
        // if the shape of a tensor is { 2 , 3 , 4 }, then the stride of a tensor is { 12 , 4 , 1 }, that is { 3*4*1 , 4*1 , 1 }
        // the ith value in the stride vector actually stores the multiplicant of all the dimension lengths that stay behind it
        // In this way, we can store a multi-dimenison tensor in an array and use stride to help us managing the tensor

    };

    // Other utility functions or global operator overloads
    Tensor cat(std::vector<ts::Tensor> Tensors, int dim);
    Tensor tile(Tensor tensor, std::vector<int> dims);
    Tensor transpose(Tensor tensor, int dim1, int dim2);
    Tensor permute(Tensor tensor, std::vector<int> dims);
    Tensor view(Tensor tensor, std::vector<size_t> shape);
}

