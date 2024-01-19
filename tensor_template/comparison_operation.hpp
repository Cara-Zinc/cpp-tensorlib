#ifndef COMPARISON_OPERATION_HPP
#define COMPARISON_OPERATION_HPP
#include "tensor.h"
#include <vector>
#include <iostream>

namespace ts {

    template <typename T>
    Tensor<bool> operator==(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Shapes must match for comparison.");
        }

        size_t total_size = a.total_size();
        std::vector<bool> result_data(total_size);

        for (size_t i = 0; i < total_size; ++i) {
            result_data[i] = a.get_element(i) == b.get_element(i);
        }

        return Tensor<bool>({total_size}, "bool", result_data);
    }

    // 静态函数定义
    template <typename T>
    Tensor<bool> Tensor<T>::eq(const Tensor<T> &other) const {
            return *this == other;
        }

    // 成员函数，实现 Tensor 与标量的逐元素比较
    template <typename T>
    Tensor<bool> Tensor<T>::eq(T value) const {
        Tensor<bool> result(this->get_shape(), "bool");
        size_t total_size = this->total_size();
        auto data_ptr = this->data_ptr(); // Use the public method to get the data pointer
        for (size_t i = 0; i < total_size; ++i) {
            result.data_ptr()[i] = data_ptr[i] == value; // Use the data pointer to set values
        }
        return result;
    }
    

    // 全局函数，用于比较两个 Tensor 对象
    template <typename T>
    Tensor<bool> eq(const Tensor<T> &a, const Tensor<T> &b) {
        return a == b;
    }

    // 全局函数，用于比较 Tensor 对象和标量
    template <typename T>
    Tensor<bool> eq(const Tensor<T> &a, T value) {
        return a.eq(value);
    }
    


    //ne
    template <typename T>
    Tensor<bool> operator!=(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Shapes must match for comparison.");
        }

        size_t total_size = a.total_size();
        std::vector<bool> result_data(total_size);

        for (size_t i = 0; i < total_size; ++i) {
            result_data[i] = a.get_element(i) != b.get_element(i);
        }

        return Tensor<bool>({total_size}, "bool", result_data);
    }

    // 静态函数定义
    template <typename T>
    Tensor<bool> Tensor<T>::ne(const Tensor<T> &other) const {
            return *this != other;
        }

    // 成员函数，实现 Tensor 与标量的逐元素比较
    template <typename T>
    Tensor<bool> Tensor<T>::ne(T value) const {
        Tensor<bool> result(this->get_shape(), "bool");
        size_t total_size = this->total_size();
        auto data_ptr = this->data_ptr(); // Use the public method to get the data pointer
        for (size_t i = 0; i < total_size; ++i) {
            result.data_ptr()[i] = data_ptr[i] != value; // Use the data pointer to set values
        }
        return result;
    }
    

    // 全局函数，用于比较两个 Tensor 对象
    template <typename T>
    Tensor<bool> ne(const Tensor<T> &a, const Tensor<T> &b) {
        return a != b;
    }

    // 全局函数，用于比较 Tensor 对象和标量
    template <typename T>
    Tensor<bool> ne(const Tensor<T> &a, T value) {
        return a.ne(value);
    }
    

    //gt
    template <typename T>
    Tensor<bool> operator>(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Shapes must match for comparison.");
        }

        size_t total_size = a.total_size();
        std::vector<bool> result_data(total_size);

        for (size_t i = 0; i < total_size; ++i) {
            result_data[i] = a.get_element(i) > b.get_element(i);
        }

        return Tensor<bool>({total_size}, "bool", result_data);
    }

    // 静态函数定义
    template <typename T>
    Tensor<bool> Tensor<T>::gt(const Tensor<T> &other) const {
            return *this > other;
        }

    // 成员函数，实现 Tensor 与标量的逐元素比较
    template <typename T>
    Tensor<bool> Tensor<T>::gt(T value) const {
        Tensor<bool> result(this->get_shape(), "bool");
        size_t total_size = this->total_size();
        auto data_ptr = this->data_ptr(); // Use the public method to get the data pointer
        for (size_t i = 0; i < total_size; ++i) {
            result.data_ptr()[i] = data_ptr[i] > value; // Use the data pointer to set values
        }
        return result;
    }
    

    // 全局函数，用于比较两个 Tensor 对象
    template <typename T>
    Tensor<bool> gt(const Tensor<T> &a, const Tensor<T> &b) {
        return a > b;
    }

    // 全局函数，用于比较 Tensor 对象和标量
    template <typename T>
    Tensor<bool> gt(const Tensor<T> &a, T value) {
        return a.gt(value);
    }
    


    //ge
    template <typename T>
    Tensor<bool> operator>=(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Shapes must match for comparison.");
        }

        size_t total_size = a.total_size();
        std::vector<bool> result_data(total_size);

        for (size_t i = 0; i < total_size; ++i) {
            result_data[i] = a.get_element(i) >= b.get_element(i);
        }

        return Tensor<bool>({total_size}, "bool", result_data);
    }

    // 静态函数定义
    template <typename T>
    Tensor<bool> Tensor<T>::ge(const Tensor<T> &other) const {
            return *this >= other;
        }

    // 成员函数，实现 Tensor 与标量的逐元素比较
    template <typename T>
    Tensor<bool> Tensor<T>::ge(T value) const {
        Tensor<bool> result(this->get_shape(), "bool");
        size_t total_size = this->total_size();
        auto data_ptr = this->data_ptr(); // Use the public method to get the data pointer
        for (size_t i = 0; i < total_size; ++i) {
            result.data_ptr()[i] = data_ptr[i] >= value; // Use the data pointer to set values
        }
        return result;
    }
    

    // 全局函数，用于比较两个 Tensor 对象
    template <typename T>
    Tensor<bool> ge(const Tensor<T> &a, const Tensor<T> &b) {
        return a >= b;
    }

    // 全局函数，用于比较 Tensor 对象和标量
    template <typename T>
    Tensor<bool> ge(const Tensor<T> &a, T value) {
        return a.ge(value);
    }
    

    //lt
    template <typename T>
    Tensor<bool> operator<(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Shapes must match for comparison.");
        }

        size_t total_size = a.total_size();
        std::vector<bool> result_data(total_size);

        for (size_t i = 0; i < total_size; ++i) {
            result_data[i] = a.get_element(i) < b.get_element(i);
        }

        return Tensor<bool>({total_size}, "bool", result_data);
    }

    // 静态函数定义
    template <typename T>
    Tensor<bool> Tensor<T>::lt(const Tensor<T> &other) const {
            return *this < other;
        }

    // 成员函数，实现 Tensor 与标量的逐元素比较
    template <typename T>
    Tensor<bool> Tensor<T>::lt(T value) const {
        Tensor<bool> result(this->get_shape(), "bool");
        size_t total_size = this->total_size();
        auto data_ptr = this->data_ptr(); // Use the public method to get the data pointer
        for (size_t i = 0; i < total_size; ++i) {
            result.data_ptr()[i] = data_ptr[i] < value; // Use the data pointer to set values
        }
        return result;
    }
    

    // 全局函数，用于比较两个 Tensor 对象
    template <typename T>
    Tensor<bool> lt(const Tensor<T> &a, const Tensor<T> &b) {
        return a < b;
    }

    // 全局函数，用于比较 Tensor 对象和标量
    template <typename T>
    Tensor<bool> lt(const Tensor<T> &a, T value) {
        return a.lt(value);
    }
    

    //ge
    template <typename T>
    Tensor<bool> operator<=(const Tensor<T>& a, const Tensor<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Shapes must match for comparison.");
        }

        size_t total_size = a.total_size();
        std::vector<bool> result_data(total_size);

        for (size_t i = 0; i < total_size; ++i) {
            result_data[i] = a.get_element(i) <= b.get_element(i);
        }

        return Tensor<bool>({total_size}, "bool", result_data);
    }

    // 静态函数定义
    template <typename T>
    Tensor<bool> Tensor<T>::le(const Tensor<T> &other) const {
            return *this <= other;
        }

    // 成员函数，实现 Tensor 与标量的逐元素比较
    template <typename T>
    Tensor<bool> Tensor<T>::le(T value) const {
        Tensor<bool> result(this->get_shape(), "bool");
        size_t total_size = this->total_size();
        auto data_ptr = this->data_ptr(); // Use the public method to get the data pointer
        for (size_t i = 0; i < total_size; ++i) {
            result.data_ptr()[i] = data_ptr[i] <= value; // Use the data pointer to set values
        }
        return result;
    }
    

    // 全局函数，用于比较两个 Tensor 对象
    template <typename T>
    Tensor<bool> le(const Tensor<T> &a, const Tensor<T> &b) {
        return a <= b;
    }

    // 全局函数，用于比较 Tensor 对象和标量
    template <typename T>
    Tensor<bool> le(const Tensor<T> &a, T value) {
        return a.le(value);
    }
    


}
#endif