#include "tensor.h"
#include <iostream>

using namespace std;

void testConstructor();
int main()
{
    testConstructor();
    return 0;
}

void testConstructor()
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