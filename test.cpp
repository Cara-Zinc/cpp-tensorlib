#include "tensor.h"
#include <iostream>

using namespace std;

void testConstructor();
void testMath();
void printTensor(const ts::Tensor& t);
int main()
{
    // ts::Tensor t = ts::Tensor({{0.1, 1.2}, {2.2, 3.1}, {4.9, 5.2}});
    // cout << t << endl;

    // ts::Tensor t1 = t(1);
    // cout << "Indexed Tensor (2nd element):\n" << t1 << endl;

    // ts::Tensor t2 = t(2, {2, 1});
    // cout << "Sliced Tensor (3rd to 4th elements in the 3rd dimension):\n" << t2 << endl;
//    testConstructor();
    testMath();
    return 0;
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

    try {
        ts::Tensor tensor(shape, dtype, data_vector);

        cout << "Tensor size: ";
        for (auto s : tensor.size()) cout << s << " ";
        cout << "\nTensor data type: " << tensor.type() << endl;
        cout << "Tensor data: ";
        for (size_t i = 0; i < tensor.size()[0]; ++i) {
            for (size_t j = 0; j < tensor.size()[1]; ++j) {
                cout << tensor.data_ptr()[i * tensor.size()[1] + j] << " ";
            }
        }
        cout << endl;
    } catch (const std::exception& e) {
        cout << "Exception occurred: " << e.what() << endl;
    }
}

void testMath(){
    // 创建测试用的 Tensor 对象
    ts::Tensor t1({2, 3}, "double", 1.0); // 创建一个 2x3 的 Tensor，初始化为 1.0
    ts::Tensor t2({2, 3}, "double", 2.0); // 创建另一个 2x3 的 Tensor，初始化为 2.0

    // 使用非成员函数进行加法
    ts::Tensor t3 = ts::add(t1, t2);
    std::cout << "Tensor t1 + t2 is: ";
    printTensor(t3);

    // 使用成员函数和标量进行加法
    ts::Tensor t4 = t1.add(5.0);
    std::cout << "Tensor t1 + 5.0 is: ";
    printTensor(t4);

    // 使用运算符重载进行加法
    ts::Tensor t5 = t1 + t2;
    std::cout << "Tensor t1 + t2 (operator+) is: ";
    printTensor(t5);

    //t6 t7 t8为减法测试
    ts::Tensor t6 = ts::sub(t1, t2);
    std::cout << "Tensor t1 - t2 is: ";
    printTensor(t6);

    ts::Tensor t7 = t1.sub(5.0);
    std::cout << "Tensor t1 - 5.0 is: ";
    printTensor(t7);

    ts::Tensor t8 = t1 - t2;
    std::cout << "Tensor t1 - t2 (operator+) is: ";
    printTensor(t8);

    

}

void printTensor(const ts::Tensor& t) {
    size_t total_size = 1;
    for (auto dim : t.size()) {
        total_size *= dim;
    }
    double* data = t.data_ptr();
    for (size_t i = 0; i < total_size; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}