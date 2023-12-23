#include "tensor.h"
#include <iostream>

using namespace std;

void testConstructor();
int main()
{
    // ts::Tensor t = ts::Tensor({{0.1, 1.2}, {2.2, 3.1}, {4.9, 5.2}});
    // cout << t << endl;

    // ts::Tensor t1 = t(1);
    // cout << "Indexed Tensor (2nd element):\n" << t1 << endl;

    // ts::Tensor t2 = t(2, {2, 1});
    // cout << "Sliced Tensor (3rd to 4th elements in the 3rd dimension):\n" << t2 << endl;
    testConstructor();
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
