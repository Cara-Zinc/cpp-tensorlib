#include "tensor.h"
#include <iostream>

using namespace std;

int main() {
    // Test constructor
    ts::Tensor t = ts::Tensor({{0.1, 1.2}, {2.2, 3.1}, {4.9, 5.2}});
    cout << t << endl;

    ts::Tensor t1 = t(1);
    cout << "Indexed Tensor (2nd element):\n" << t1 << endl;

    ts::Tensor t2 = t(2, {2, 1});
    cout << "Sliced Tensor (3rd to 4th elements in the 3rd dimension):\n" << t2 << endl;


    return 0;
}
