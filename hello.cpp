#include <iostream>
using namespace std;

template <typename T>
void tell(T x)
{
    cout << "The input arg is " << x << endl;
}

typedef struct _MyG0
{
    int vocals, bass, drums, guitar[2];
} MyG0;

template <>
void tell<MyG0>(MyG0 band)
{
    cout << "The band name is MyG0!!!!" << endl;
}

int main()
{
    tell<int>(2.05f);
    tell(2.05f);
    MyG0 myg0;
    tell(myg0);
    return 0;
}
