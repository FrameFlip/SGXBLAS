#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdbool>
#include <cmath>
#include <cstdarg>
#include <vector>
#include <stdexcept>

#include "utils.h"

template <typename T>
class Tensor;

template <typename T>
ostream &operator<<(ostream &os, const Tensor<T>& obj);

template <typename T>
int print_tensor_rec(
    ostream &os, 
    const Tensor<T>& obj, 
    size_t layer, 
    size_t base, 
    string delimiter=", "
);


template <typename T>
class Tensor
{
public:
    // constructors
    Tensor():
        arr(NULL), local_arr(false), total_elems(0) {}
    Tensor(const Tensor<T>& t);                 // copy constructor
    Tensor(Tensor<T> && t);                     // move constructor
    Tensor(T *input_arr, const vector<int>& shape_): 
        arr(input_arr), shape(shape_), local_arr(false) {}
    Tensor(int dim, ...);
    // destructors
    ~Tensor();

    Tensor<T>& operator=(const Tensor<T>& t);
    Tensor<T>& operator=(Tensor<T> && obj);
    // arithmetics
    Tensor<T> operator+(const Tensor<T>& t) const;        // element-wise
    Tensor<T> operator-(const Tensor<T> &t) const;        // element-wise
    Tensor<T> operator*(const Tensor<T> &t) const;        // element-wise

    friend ostream &operator<< <>(ostream &os, const Tensor<T>& obj);
    friend int print_tensor_rec<>(
        ostream &os, const Tensor<T>& obj, size_t layer, size_t base, string delimiter);

    T& at(int n1, ...);
    const T& at(int n1, ...) const;
    // T& fast_at(int n1, ...);        // fast but dangerous! no boundary check
    const vector<int>& get_shape() const { return shape; }
    int get_total_elems(void) const;
    bool same_size_with(const Tensor<T> &t) const;
    void local_reshape(int dim, ...);
    void load_from(const char *filename);
    void load_from(const string& filename);
    void load_from(ifstream& in_file);
    
    // For developers only, may cause memory leakage
    inline T& at_3d(int n1, int n2, int n3);
    inline const T& at_3d(int n1, int n2, int n3) const;
    inline T& at_4d(int n1, int n2, int n3, int n4);
    inline const T& at_4d(int n1, int n2, int n3, int n4) const;
    T *get_pointer(void) { return arr; }
    T *get_pointer(void) const { return arr; }
    void set_pointer(T *new_ptr, bool is_local);
    void set_shape(const vector<int>& new_shape) { shape = new_shape; update_total_elems(); }
    void inplace_addition(const Tensor<T>& t);
    int update_total_elems(void);

private:
    T *arr;             // pointer to data storage
    vector<int> shape;  // tensor shape
    bool local_arr;     // true if `arr` is allocated locally
    int total_elems;

    int load_from_rec(ifstream& in_file, int layer, int start_idx);
};



template <typename T>
Tensor<T>::~Tensor()
{
    if (local_arr)
    {
        delete [] arr;
    }
}


template <typename T>
int Tensor<T>::get_total_elems(void) const
{
    // if (shape.size() < 1)
    // {
    //     return 0;
    // }
    // int total_elems = 1;
    // for (const int & i: this->shape)
    // {
    //     total_elems *= i;
    // }
    // return total_elems;
    return total_elems;
}


template <typename T>
int Tensor<T>::update_total_elems(void)
{
    if (shape.size() < 1)
    {
        return 0;
    }
    total_elems = 1;
    for (const int & i: this->shape)
    {
        total_elems *= i;
    }
    return total_elems;
}


template <typename T>
Tensor<T>::Tensor(const Tensor<T>& t):
    shape(t.shape), total_elems(t.total_elems)
{
    // cout << "copy constructor called" << endl;
    if (this != &t)
    {
        if (t.arr && shape.size() > 0)
        {
            // cout << "total elems " << get_total_elems() << endl;
            this->local_arr = true;
            this->arr = new T[this->get_total_elems()];
            for (int i = 0; i < this->get_total_elems(); ++ i)
            {
                this->arr[i] = t.arr[i];
            }
            // cout << "copied" << endl;
        }
        else
        {
            this->local_arr = false;
            this->arr = NULL;
        }
    }
}


template <typename T>
Tensor<T>::Tensor(Tensor<T> && obj):
    arr(obj.arr),
    shape(move(obj.shape)),
    local_arr(obj.local_arr),
    total_elems(move(obj.total_elems))
{
    // cout << "move constructor called" << endl;
    
    // adopt members from `obj`
    // arr = obj.arr;
    // shape = move(obj.shape);
    // local_arr = obj.local_arr;

    // release storage ownership in `obj`
    obj.arr = NULL;
    if (obj.local_arr)
    {
        obj.local_arr = false;
    }
}



template <typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor<T>& t)
{
    // cout << "copy assignment operator called" << endl;   
    if (this != &t)
    {
        this->shape = t.shape;
        this->total_elems = t.total_elems;
        if (this->local_arr)
            delete [] this->arr;
        this->local_arr = true;
        this->arr = new T[this->get_total_elems()];
        for (int i = 0; i < this->get_total_elems(); ++ i)
        {
            this->arr[i] = t.arr[i];
        }
    }
    return *this;
}


template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor<T> && obj)
{
    // cout << "move assignment operator called" << endl;

    // first release the original storage space
    if (local_arr)
    {
        delete [] arr;
    }

    // then adopt members from `obj`
    arr = obj.arr;
    shape = move(obj.shape);
    total_elems = move(obj.total_elems);
    local_arr = obj.local_arr;

    // at last release storage ownership in `obj`
    obj.arr = NULL;
    if (obj.local_arr)
    {
        obj.local_arr = false;
    }

    return *this;
}



template <typename T>
Tensor<T>::Tensor(int dim, ...)
{
    // read in dimensions
    va_list ap;
    va_start(ap, dim);
    this->shape.resize(dim);
    for (int i = 0; i < dim; ++ i)
    {
        this->shape[i] = va_arg(ap, int);
    }
    va_end(ap);

    // cout << "New tensor shape: ";
    // print_vector(this->shape, ",");
    update_total_elems();

    this->arr = new T[this->get_total_elems()];
    this->local_arr = true;
}


template <typename T>
const T& Tensor<T>::at(int n1, ...) const
{
    if (this->shape.size() < 1)
    {
        throw invalid_argument("Cannot call at() for tensor with no dimension\n");
    }
    else if (n1 >= this->shape[0])
    {
        if (DEBUG_FLAG)
        {
            cout << "n1 = " << n1 << "\tshape: ";
            print_vector(this->shape);
            fflush(stdout);
        }
        throw invalid_argument("First index out of bound");
    }

    int pos = n1; // alg: pos = ( ... (((n1 * d2) + n2) * d3) + n3 ... )

    va_list ap;
    int n_i;
    va_start(ap, n1);
    for (size_t i = 1; i < this->shape.size(); ++ i)
    {
        n_i = va_arg(ap, int);
        if (n_i >= this->shape[i])
        {
            if (DEBUG_FLAG)
            {
                printf("n[%ld] = %d\tshape: ", i, n_i);
                print_vector(this->shape);
            }
            throw invalid_argument("Index out of bound");
        }
        pos *= this->shape[i];
        pos += n_i;
    }
    va_end(ap);

    return this->arr[pos];
}


// template <typename T>
// T& Tensor<T>::fast_at(int n1, ...)
// {
//     int pos = n1; // alg: pos = ( ... (((n1 * d2) + n2) * d3) + n3 ... )

//     va_list ap;
//     int n_i;
//     va_start(ap, n1);
//     for (size_t i = 1; i < this->shape.size(); ++ i)
//     {
//         n_i = va_arg(ap, int);
//         pos *= this->shape[i];
//         pos += n_i;
//     }
//     va_end(ap);

//     return this->arr[pos];
// }


template <typename T>
inline T& Tensor<T>::at_3d(int n1, int n2, int n3)
{
    return arr[((n1 * shape[1]) + n2) * shape[2] + n3];
}


template <typename T>
inline const T& Tensor<T>::at_3d(int n1, int n2, int n3) const
{
    return arr[((n1 * shape[1]) + n2) * shape[2] + n3];
}



template <typename T>
inline T& Tensor<T>::at_4d(int n1, int n2, int n3, int n4)
{
    return arr[(((n1 * shape[1]) + n2) * shape[2] + n3) * shape[3] + n4];
}


template <typename T>
inline const T& Tensor<T>::at_4d(int n1, int n2, int n3, int n4) const
{
    return arr[(((n1 * shape[1]) + n2) * shape[2] + n3) * shape[3] + n4];
}


template <typename T>
T& Tensor<T>::at(int n1, ...)
{
    if (this->shape.size() < 1)
    {
        throw invalid_argument("Cannot call at() for tensor with no dimension\n");
    }
    else if (n1 >= this->shape[0])
    {
        if (DEBUG_FLAG)
        {
            cout << "n1 = " << n1 << "\tshape: ";
            print_vector(this->shape);
            fflush(stdout);
        }
        throw invalid_argument("First index out of bound");
    }

    int pos = n1; // alg: pos = ( ... (((n1 * d2) + n2) * d3) + n3 ... )

    va_list ap;
    int n_i;
    va_start(ap, n1);
    for (size_t i = 1; i < this->shape.size(); ++ i)
    {
        n_i = va_arg(ap, int);
        if (n_i >= this->shape[i])
        {
            if (DEBUG_FLAG)
            {
                printf("n[%ld] = %d\tshape: ", i, n_i);
                print_vector(this->shape);
            }
            throw invalid_argument("Index out of bound");
        }
        pos *= this->shape[i];
        pos += n_i;
    }
    va_end(ap);

    return this->arr[pos];
}


template <typename T>
bool Tensor<T>::same_size_with(const Tensor<T> &t) const
{
    if (this->shape.size() != t.shape.size())
        return false;

    for (size_t i = 0; i < this->shape.size(); ++ i)
    {
        if (this->shape[i] != t.shape[i])
            return false;
    }

    return true;
}


template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& t) const
{
    if (not same_size_with(t))
    {
        printf("Report tensor shape:\nthis->shape: ");
        print_vector(shape);
        printf("t.shape: ");
        print_vector(t.shape);
        throw invalid_argument("Different size in addition");
    }

    Tensor<T> ret(*this);
    for (int i = 0; i < get_total_elems(); ++ i)
    {
        ret.arr[i] += t.arr[i];
    }

    return ret;
}


template <typename T>
void Tensor<T>::inplace_addition(const Tensor<T>& t)
{
    if (not same_size_with(t))
    {
        printf("Report tensor shape:\nthis->shape: ");
        print_vector(shape);
        printf("t.shape: ");
        print_vector(t.shape);
        throw invalid_argument("Different size in addition");
    }

    for (int i = 0; i < get_total_elems(); ++ i)
    {
        arr[i] += t.arr[i];
    }
}


template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& t) const
{
    if (not same_size_with(t))
    {
        throw invalid_argument("Different size in addition");
    }

    Tensor<T> ret(*this);
    for (int i = 0; i < get_total_elems(); ++ i)
    {
        ret.arr[i] -= t.arr[i];
    }

    return ret;
}


template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& t) const
{
    if (not same_size_with(t))
    {
        throw invalid_argument("Different size in addition");
    }

    Tensor<T> ret(*this);
    for (int i = 0; i < get_total_elems(); ++ i)
    {
        ret.arr[i] *= t.arr[i];
    }

    return ret;
}


template <typename T>
int print_tensor_rec(
    ostream &os, 
    const Tensor<T>& obj, 
    size_t layer, 
    size_t base, 
    string delimiter
)
{   // returns the new base (index of the next element to print)
    if (layer >= obj.shape.size())
    {   // impossible layer
        return base;
    }
    else if (layer == obj.shape.size() - 1)
    {   // print vector
        os << '[';
        for (int i = 0; i < obj.shape[layer]; ++ i)
        {
            if (i) os << delimiter;
            os << obj.arr[base ++];
        }
        os << ']';
    }
    else
    {   // make recursive calls 
        os << '[';
        for (int i = 0; i < obj.shape[layer]; ++ i)
        {
            if (i) for (size_t j = 0; j <= layer; ++ j) os << ' ';
            base = print_tensor_rec(os, obj, layer + 1, base, delimiter);
            if (i != obj.shape[layer] - 1) 
                for (size_t j = 1; j < (obj.shape.size() - layer); ++ j)
                    os << endl;
        }
        os << ']';
    }

    return base;
}


template <typename T>
ostream &operator<<(ostream &os, const Tensor<T>& obj)
{
    print_tensor_rec(os, obj, 0, 0, ", ");

    return os;
}


template <typename T>
void Tensor<T>::local_reshape(int dim, ...)
{
    vector<int> new_shape;

    va_list ap;
    va_start(ap, dim);
    for (int i = 0; i < dim; ++ i)
    {
        new_shape.push_back(va_arg(ap, int));
    }
    va_end(ap);

    int mul = 1;
    for (const int& i: new_shape)
    {
        mul *= i;
    }
    if (mul == this->get_total_elems())
    {
        this->shape = new_shape;
    }
    else
    {
        throw runtime_error("New shape does not match the original shape");
    }
}


template <typename T>
void Tensor<T>::load_from(const char *filename)
{
    ifstream in_file;
    in_file.open(filename, ios::in);
    load_from(in_file);
    in_file.close();
}


template <typename T>
void Tensor<T>::load_from(const string& filename)
{
    load_from(filename.c_str());
}



static double round_my(double var, int multipler){
    double value = (int)(var * multipler + .5);
    return (double)value / multipler;
}

template <typename T>
int Tensor<T>::load_from_rec(ifstream& in_file, int layer, int start_idx)
{
    char ch;
    double tmp;
    int multipler = 10000;

    do { in_file >> ch; } while (ch != '[');
    if (layer + 1 == (int)shape.size())
    {
        for (int i = 0; i < shape[layer]; ++ i)
        {
            // in_file >> tmp;
            // double round_tmp = round_my(tmp, multipler);
            // arr[start_idx ++] = round_tmp;
            in_file >> arr[start_idx ++];
        }
    }
    else
    {
        for (int i = 0 ; i < shape[layer]; ++ i)
            start_idx = load_from_rec(in_file, layer + 1, start_idx);
    }
    do { in_file >> ch; } while (ch != ']');
    
    return start_idx;
}


template <typename T>
void Tensor<T>::load_from(ifstream& in_file)
{
    load_from_rec(in_file, 0, 0);
}


template <typename T>
void Tensor<T>::set_pointer(T *new_ptr, bool is_local)
{
    if (arr != NULL && local_arr)
    {
        delete [] arr;
    }

    arr = new_ptr;
    local_arr = is_local;
}


#endif