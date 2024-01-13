#include "tensor.h"
#include <iostream>

using namespace std;

void testConstructor();
void testOperation();
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
    testConstructor();
    testOperation();
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

void testOperation()
{

    //test 3:using slicing
    cout << "Test 3: Slicing" << endl;
    vector<size_t> shape = {2, 3, 4};
    string dtype = "double";
    std::vector<double> a(24);
    
    for (int i = 0; i < 24; ++i) {
        a[i] = i+1;
    }
    vector<size_t> shape1 = {2, 4, 4};
    std::vector<double> b(32);
    for (int i = 0; i < 32; ++i) {
        b[i] = i+1;
    }
    ts::Tensor t(shape, dtype, a);
    ts::Tensor t1(shape1, dtype, b);
    ts::Tensor t_index = t(1, 2);
    ts::Tensor t_slicing = t(vector{0, 1}, 0, vector{1, 3});
    
    ts::Tensor t_cat = cat(vector{t, t1}, 1);

    ts::Tensor t_tile = tile(t, vector{1, 2, 3});
    
    t(0) = 1;
    t(0, vector{0, 2}, 0) = vector{3.0, 2.0};
    //here comes the problem line 120 throws an error
    ts::Tensor t_tra1 = transpose(t, 0, 2);
    
    ts::Tensor t_tra2 = t.transpose(0, 2);
    
    ts::Tensor t_per1 = permute(t, {2, 0, 1});
    ts::Tensor t_per2 = t.permute({2, 0, 1});
    ts::Tensor t_view1 = view(t, {2, 0, 1});
    
    ts::Tensor t_view2 = t.view({2, 0, 1});

    cout << "Tensor t size: ";
    for (auto s: t.size()) {
        std::cout << s << " ";
    }
    cout << "Tensor data: \n";
    for (size_t i = 0; i < t.size()[0]; ++i) {
        for (size_t j = 0; j < t.size()[1]; ++j) {
            for (int k = 0; k < t.size()[2]; ++k) {
                cout << t.data_ptr()[i * t.size()[1]*t.size()[2] + j*t.size()[2] + k] << " ";
            }
            cout << endl;
        }
    }
    cout << endl;
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