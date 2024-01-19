// #pragma once

// #include <iostream>
// #include <vector>
// #include <cstdlib>
// #include <ctime>
// #include <variant>

// namespace tis
// {
//     // Forward declaration
//     template <typename T>
//     class TensorTemplate;

//     // Function to print tensor information
//     template <typename T>
//     std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor);

//     template <typename T>
//     class Tensor
//     {
//     public:
//         Tensor();
//         Tensor(const std::vector<std::vector<T>> &data);
//         //Tensor(const std::vector<size_t> &shape, const std::string &dtype, T init_value = T());
//         Tensor(const std::vector<size_t> &shape, const std::string &dtype, std::vector<T> data);

//         std::vector<size_t> size() const;
//         std::vector<size_t> get_shape() const;
//         std::vector<size_t> get_stride() const;
//         std::string type() const;
//         T *data_ptr() const;
//         T get_element(size_t index) const;
//         //void set_element(size_t index, T value);
//         size_t total_size() const;
//         int dimens() const;

//         // slicing
//         // template <typename... Args>
//         // Tensor<T> operator()(Args... args);

//         // // mutating
//         // void operator=(T val);
//         // void operator=(std::vector<T> val);

//         // transpose
//         // Tensor<T> transpose(int dim1, int dim2);

//         // // permute
//         // Tensor<T> permute(std::vector<int> dims);

//         // // view
//         // Tensor<T> view(std::vector<size_t> shape);

//         // Tensor<T> add(const Tensor<T> &other) const;
//         // Tensor<T> add(T value) const;
//         // Tensor<T> sub(const Tensor<T> &other) const;
//         // Tensor<T> sub(T value) const;
//         // Tensor<T> mul(const Tensor<T> &other) const;
//         // Tensor<T> mul(T value) const;
//         // Tensor<T> div(const Tensor<T> &other) const;
//         // Tensor<T> div(T value) const;

//         // friend Tensor<T> operator+(const Tensor<T> &a, const Tensor<T> &b);
//         // friend Tensor<T> operator-(const Tensor<T> &a, const Tensor<T> &b);
//         // friend Tensor<T> operator*(const Tensor<T> &a, const Tensor<T> &b);
//         // friend Tensor<T> operator/(const Tensor<T> &a, const Tensor<T> &b);

//         // Other member functions for tensor operations, indexing, slicing, etc.
//         Tensor<T> sum(int dim) const;
//         Tensor<T> mean(int dim) const;
//         Tensor<T> max(int dim) const;
//         Tensor<T> min(int dim) const;

//     private:
//         T *data_;
//         int dimension;                 // the number of dimensions this tensor has
//         std::vector<size_t> shape;     // shape of the tensor, storing the length of every dimension of the tensor
//         std::string dtype_;
//         std::vector<size_t> offset;    // the shift between the start of the tensor to tensor->data
//         std::vector<size_t> stride;    // store the stride of every dimension of the tensor
//         std::vector<int> data_pos;     // for get_element
//     };

//     // // Other utility functions or global operator overloads
//     // template <typename T>
//     // Tensor<T> cat(std::vector<Tensor<T>> Tensors, int dim);

//     // template <typename T>
//     // Tensor<T> tile(Tensor<T> tensor, std::vector<int> dims);

//     // template <typename T>
//     // Tensor<T> transpose(Tensor<T> tensor, int dim1, int dim2);

//     // template <typename T>
//     // Tensor<T> permute(Tensor<T> tensor, std::vector<int> dims);

//     // template <typename T>
//     // Tensor<T> view(Tensor<T> tensor, std::vector<size_t> shape);

//     // template <typename T>
//     // Tensor<T> add(const Tensor<T> &a, const Tensor<T> &b);

//     // template <typename T>
//     // Tensor<T> add(const Tensor<T> &a, T value);

//     // template <typename T>
//     // Tensor<T> sub(const Tensor<T> &a, const Tensor<T> &b);

//     // template <typename T>
//     // Tensor<T> sub(const Tensor<T> &a, T value);

//     template <typename T>
//     Tensor<T> sum(const Tensor<T> &t, int dim);

//     template <typename T>
//     Tensor<T> mean(const Tensor<T> &t, int dim);

//     template <typename T>
//     Tensor<T> max(const Tensor<T> &t, int dim);

//     template <typename T>
//     Tensor<T> min(const Tensor<T> &t, int dim);

//     template <typename T>
//     Tensor<T> dot(const Tensor<T> &a, const Tensor<T> &b);
// }

// // Include the implementation file here (e.g., in a .cpp file)

