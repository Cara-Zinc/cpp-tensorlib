# cpp-tensorlib

## Project Structure
```shell
cpp-tensorlib
├── CMakeLists.txt
├── tensor.h
├── tensor.cpp
├── tensor_creation.cpp
├── tensor_operations.cpp
├── math_operations.cpp
└── test.cpp
```
## goals
- 1
    - ts::Tensor t = ts::tensor(T data[]);//生成

    - ts::Tensor t = ts::rand<T>(int size[]);//随机生成

    - ts::Tensor t = ts::zeros<T>(int size[]);
    - ts::Tensor t = ts::ones<T>(int size[]);
    - ts::Tensor t = ts::full(int size[], T value);

    - ts::Tensor t = ts::eye<T>(int size[]); // This creates an identity matrix.

- 2
    - ts::Tensor t = ts::tensor(T data[]);
    - ts::Tensor t1 = t(1); // This indexes the second element of t.
    - ts::Tensor t2 = t(2,{2,4}); // This slices the third to fifth (excluded) elements of the third dimension of t.

    - ts::Tensor t1 = ts::tensor(T data1[]);
    - ts::Tensor t2 = ts::tensor(T data2[]);
    - ts::Tensor t3 = ts::cat({t1, t2}, int dim); // This joins t1 and t2 along the given dimension.
    - ts::Tensor t4 = ts::tile(t1, int dims[]); // This construct t4 by repeating the elements of t1

    - ts::Tensor t = ts::tensor(T data[]); // t(1) = 1; // This sets the second element of t to 1. // t(2,{2,4}) = [1,2]; // This sets the third to fifth (excluded) elements of the third dimension of t to [1,2].

    - ts::Tensor t = ts::tensor(T data[]);
    - ts::Tensor t1 = ts::transpose(t, int dim1, int dim2); // This transposes the tensor t along the given dimensions.
    - ts::Tensor t2 = t.transpose(int dim1, int dim2); // Another way to transpose the tensor t.
    - ts::Tensor t3 = ts::permute(t, int dims[]); // This permutes the tensor t according to the given dimensions.
    - ts::Tensor t4 = t.permute(int dims[]); // Another way to permute the tensor t.

    - ts::Tensor t = ts::tensor(T data[]);
    - ts::Tensor t3 = ts::view(t, int shape[]); // This views the tensor t according to the given shape.
    - ts::Tensor t4 = t.view(int shape[]); // Another way to view the tensor t.

- 3
    - ts::Tensor t1 = ts::tensor(T data1[]);
    - ts::Tensor t2 = ts::tensor(T data2[]);
    - ts::Tensor t3 = ts::add(t1, t2); // This adds t1 and t2 element-wise.
    - ts::Tensor t4 = t1.add(t2); // Another way to add t1 and t2 element-wise.
    - ts::Tensor t5 = t1 + t2; // Another way to add t1 and t2 element-wise.
    - ts::Tensor t6 = ts::add(t1, T value); // This adds t1 and a scalar value element-wise.
    - ts::Tensor t7 = t1.add(T value); // Another way to add t1 and a scalar value element-wise.// ... Similar for sub, mul, div, log.

    - ts::Tensor t = ts::tensor(T data[]);
    - ts::Tensor t1 = ts::sum(t, int dim);// This sums the tensor t along the given dimension.
    - ts::Tensor t2 = t.sum(int dim);// Another way to sum the tensor t along the given dimension.// ... Similar for mean, max, min.

    - ts::Tensor t1 = ts::tensor(T data1[]);
    - ts::Tensor t2 = ts::tensor(T data2[]);
    - ts::Tensor<bool> t3 = ts::eq(t1, t2);// This compares t1 and t2 element-wise.
    - ts::Tensor t4<bool> = t1.eq(t2);// Another way to compare t1 and t2 element-wise.
    - ts::Tensor t5<bool> = t1 == t2;// Another way to compare t1 and t2 element-wise.// ... Similar for ne, gt, ge, lt, le.

    - ts::Tensor t1 = ts::tensor(T data1[]);
    - ts::Tensor t2 = ts::tensor(T data2[]);
    - ts::Tensor t3 = ts::einsum("i,i->", t1, t2); // This computes the dot product of t1 and t2.
    - ts::Tensor t4 = ts::einsum("i,i->i", t1, t2); // This computes the element-wise product of t1 and t2.
    - ts::Tensor t5 = ts::einsum("ii->i", t1); // This computes the diagonal of t1.
    - ts::Tensor t6 = ts::einsum("i,j->ij", t1, t2); // This computes the outer product of t1 and t2.
    - ts::Tensor t7 = ts::einsum("bij,bjk->bik", t1, t2); // This computes the batch matrix multiplication of t1 and t2

- advanced
    - safe & load
    - ts::Tensor t = ts::tensor(T data[]);
    - ts::save(t, string filename); // This saves the tensor t to the given file.
    - ts::Tensor t1 = ts::load(string filename); // This loads the tensor t from the given file.
    

- 加速
- 梯度
- more