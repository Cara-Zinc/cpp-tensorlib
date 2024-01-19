#include "tensor.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
using namespace ts;
using namespace std;
void testConstructor();
void testTemplate();
void testOperation();
void testMath();
void testComparison();
template <typename T>
void testReduction();

int main()
{
    // testConstructor();
    // testOperation();
    testReduction<double>();
    testComparison();
    // testMath();
    return 0;
}

void testOperation()
{

    cout << "Test 3: Slicing" << endl;

    // Create Tensor objects
    vector<size_t> shape = {2, 3, 4};
    string dtype = "double";
    vector<double> a(24);
    iota(a.begin(), a.end(), 1); // Fill with values from 1 to 24

    vector<size_t> shape1 = {2, 4, 4};
    vector<double> b(32);
    iota(b.begin(), b.end(), 1); // Fill with values from 1 to 32

    Tensor<double> t(shape, dtype, a);
    Tensor<double> t1(shape1, dtype, b);

    // Create a vector of Tensor objects
    vector<Tensor<double>> tensors;
    tensors.push_back(t);
    tensors.push_back(t1);

    // Slicing operations
    Tensor<double> t_index = t(1, 2);
    Tensor<double> t_slicing = t(vector{0, 1}, 0, vector{1, 3});

    // Using the previously created vector
    Tensor<double> t_cat = cat(tensors, 1);
    Tensor<double> t_tile = tile(t, vector{1, 2, 3});

    // Mutations
    t(0) = 1;
    t(0, vector{0, 2}, 0) = vector{3.0, 2.0};

    // Transpose operations
    Tensor<double> t_tra1 = transpose(t, 0, 2);
    Tensor<double> t_tra2 = t.transpose(0, 2);

    // Permutation and view operations
    Tensor<double> t_per1 = permute(t, {2, 0, 1});
    Tensor<double> t_per2 = t.permute({2, 0, 1});
    Tensor<double> t_view1 = view(t, {4, 3, 2});
    Tensor<double> t_view2 = t.view({3, 4, 2});

    // Print Tensor information
    cout << "Tensor t size: ";
    for (auto s : t.size())
    {
        cout << s << " ";
    }
    cout << "\nTensor data: \n";
    for (size_t i = 0; i < t.size()[0]; ++i)
    {
        for (size_t j = 0; j < t.size()[1]; ++j)
        {
            for (size_t k = 0; k < t.size()[2]; ++k)
            {
                cout << t.data_ptr()[i * t.size()[1] * t.size()[2] + j * t.size()[2] + k] << " ";
            }
            cout << endl;
        }
    }
    cout << endl;
}

void testConstructor()
{
    // Test 1: Using the data constructor
    cout << "Test 1: Data Constructor" << endl;
    vector<vector<double>> data = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    ts::Tensor tensor1(data);

    cout << "Tensor 1 size: ";
    for (auto s : tensor1.size())
        cout << s << " ";
    cout << "\nTensor 1 data type: " << tensor1.type() << endl;
    cout << "Tensor 1 data: ";
    for (size_t i = 0; i < tensor1.size()[0]; ++i)
    {
        for (size_t j = 0; j < tensor1.size()[1]; ++j)
        {
            cout << tensor1.data_ptr()[i * tensor1.size()[1] + j] << " ";
        }
    }
    cout << "\n\n";

    // Test 2: Using the shape-type constructor
    cout << "Test 2: Shape-Type Constructor" << endl;
    vector<size_t> shape = {2, 3};
    string dtype = "double";
    double init_value = 0.5;
    ts::Tensor tensor2(shape, dtype, init_value);

    cout << "Tensor 2 size: ";
    for (auto s : tensor2.size())
        cout << s << " ";
    cout << "\nTensor 2 data type: " << tensor2.type() << endl;
    cout << "Tensor 2 data: ";
    for (size_t i = 0; i < tensor2.size()[0]; ++i)
    {
        for (size_t j = 0; j < tensor2.size()[1]; ++j)
        {
            cout << tensor2.data_ptr()[i * tensor2.size()[1] + j] << " ";
        }
    }
    cout << "\n\n";

    // Test 3: Using the third constructor with data_vector
    cout << "Test: Third Constructor" << endl;
    vector<double> data_vector = {1.2, 3.4, 5.6, 7.8, 9.0, 10.1};

    try
    {
        ts::Tensor tensor(shape, dtype, data_vector);

        cout << "Tensor size: ";
        for (auto s : tensor.size())
            cout << s << " ";
        cout << "\nTensor data type: " << tensor.type() << endl;
        cout << "Tensor data: ";
        for (size_t i = 0; i < tensor.size()[0]; ++i)
        {
            for (size_t j = 0; j < tensor.size()[1]; ++j)
            {
                cout << tensor.data_ptr()[i * tensor.size()[1] + j] << " ";
            }
        }
        cout << endl;
    }
    catch (const std::exception &e)
    {
        cout << "Exception occurred: " << e.what() << endl;
    }
}

template <typename T>
void printTensorData(const Tensor<T> &tensor)
{
    for (size_t i = 0; i < tensor.total_size(); ++i)
    {
        cout << tensor.data_ptr()[i] << " ";
        if ((i + 1) % tensor.get_shape().back() == 0)
            cout << endl;
    }
    cout << endl;
}

void testMath()
{
    // Create Tensors of different types
    Tensor<int> tensorInt({2, 2}, "int", vector<int>{1, 2, 3, 4});
    Tensor<float> tensorFloat({2, 2}, "float", vector<float>{1.1f, 2.2f, 3.3f, 4.4f});
    Tensor<double> tensorDouble({2, 2}, "double", vector<double>{1.01, 2.02, 3.03, 4.04});

    // Perform operations and print results
    cout << "Int Tensor:" << endl;
    printTensorData(tensorInt);

    cout << "Float Tensor:" << endl;
    printTensorData(tensorFloat);

    cout << "Double Tensor:" << endl;
    printTensorData(tensorDouble);

    // Example operation: Add tensor with itself
    auto resultInt = tensorInt + tensorInt;
    auto resultFloat = tensorFloat + tensorFloat;
    auto resultDouble = tensorDouble + tensorDouble;

    cout << "Int Tensor after addition:" << endl;
    printTensorData(resultInt);

    cout << "Float Tensor after addition:" << endl;
    printTensorData(resultFloat);

    cout << "Double Tensor after addition:" << endl;
    printTensorData(resultDouble);

    // Continue with subtraction, multiplication and division tests
    // ...

    return;
}

// Include your Tensor class definition here

// Function to test the sum() function
template <typename T>
void testReduction()
{
    // Create a tensor with some data
    std::vector<T> testData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<size_t> shape = {3, 3};

    ts::Tensor<T> testTensor = Tensor<T>(shape, "double", testData);

    // Test the sum() function along dimension 0
    ts::Tensor<T> result = testTensor.sum(0);

    // Verify the result
    std::vector<size_t> expectedShape = {3};
    assert(result.get_shape() == expectedShape);

    T *resultData = result.data_ptr();
    assert(resultData[0] == 12); // 1 + 4 + 7
    assert(resultData[1] == 15); // 2 + 5 + 8
    assert(resultData[2] == 18); // 3 + 6 + 9

    std::cout << "Test passed: sum() along dimension 0\n";

    std::vector<T> testData1;
    for (int i = 0; i < 24; i++)
    {
        testData1.push_back(T(i + 1));
    }
    std::vector<size_t> shape1 = {3, 2, 4};
    ts::Tensor<T> testTensor1 = Tensor<T>(shape1, "double", testData1);
    std::cout << testData1.size() << " " << shape1.size() << std::endl;

    ts::Tensor<T> result1 = testTensor1.mean(1);
    printTensorData(result1);

    ts::Tensor<T> result2 = testTensor1.max(2);
    printTensorData(result2);

    ts::Tensor<T> result3 = testTensor1.min(0);
    printTensorData(result3);
}

void testComparison()
{
    std::vector<int> data1 = {1, 2, 3, 4};
    std::vector<int> data2 = {1, 2, 3, 5}; // Note the last element is different
    ts::Tensor<int> t1({2, 2}, "int", data1);
    ts::Tensor<int> t2({2, 2}, "int", data2);

    // Define a lambda function for printing Tensor<bool>
    auto printTensorBool = [](const ts::Tensor<bool> &tensor)
    {
        for (size_t i = 0; i < tensor.total_size(); ++i)
        {
            std::cout << (tensor.get_element(i) ? "true " : "false ");
            if ((i + 1) % tensor.get_shape().back() == 0)
                std::cout << std::endl;
        }
        std::cout << std::endl;
    };

    // Test each comparison operation
    std::cout << "Testing eq:" << std::endl;
    printTensorBool(t1.eq(t2));

    std::cout << "Testing ne:" << std::endl;
    printTensorBool(t1.ne(t2));

    std::cout << "Testing gt:" << std::endl;
    printTensorBool(t1.gt(t2));

    std::cout << "Testing ge:" << std::endl;
    printTensorBool(t1.ge(t2));

    std::cout << "Testing lt:" << std::endl;
    printTensorBool(t1.lt(t2));

    std::cout << "Testing le:" << std::endl;
    printTensorBool(t1.le(t2));

    // Testing with a scalar value
    int scalar = 3;
    std::cout << "Comparing t1 with scalar " << scalar << ":" << std::endl;
    std::cout << "eq:" << std::endl;
    printTensorBool(t1.eq(scalar));

    std::cout << "ne:" << std::endl;
    printTensorBool(t1.ne(scalar));

    std::cout << "gt:" << std::endl;
    printTensorBool(t1.gt(scalar));

    std::cout << "ge:" << std::endl;
    printTensorBool(t1.ge(scalar));

    std::cout << "lt:" << std::endl;
    printTensorBool(t1.lt(scalar));

    std::cout << "le:" << std::endl;
    printTensorBool(t1.le(scalar));
}

void testTemplate()
{
    // 测试创建一个 double 类型的 Tensor
    vector<vector<double>> doubleData = {{1.0, 2.0}, {3.0, 4.0}};
    Tensor<double> tensorDouble(doubleData);
    cout << "Tensor of doubles:" << endl;
    for (size_t i = 0; i < tensorDouble.total_size(); ++i)
    {
        cout << tensorDouble.get_element(i) << " ";
    }
    cout << endl;

    // 测试创建一个 int 类型的 Tensor
    vector<size_t> shape = {2, 2};
    Tensor<int> tensorInt(shape, "int", 5); // 初始化所有元素为 5
    cout << "Tensor of ints:" << endl;
    for (size_t i = 0; i < tensorInt.total_size(); ++i)
    {
        cout << tensorInt.get_element(i) << " ";
    }
    cout << endl;

    // 测试设置和获取元素
    tensorInt.set_element(1, 10); // 将第二个元素设置为 10
    cout << "Modified tensor of ints:" << endl;
    for (size_t i = 0; i < tensorInt.total_size(); ++i)
    {
        cout << tensorInt.get_element(i) << " ";
    }
    cout << endl;

    // 测试 bool 类型的 Tensor
    vector<bool> boolData = {true, false, true, false};
    Tensor<bool> tensorBool(shape, "bool", boolData);
    cout << "Tensor of bools:" << endl;
    for (size_t i = 0; i < tensorBool.total_size(); ++i)
    {
        cout << (tensorBool.get_element(i) ? "true" : "false") << " ";
    }
    cout << endl;
}