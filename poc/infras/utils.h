#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <chrono>

using namespace std;

template <typename T>
void print_vector(vector<T> vec, string delimiter=",")
{
    cout << "(";
    for (size_t i = 0; i < vec.size(); ++ i)
    {
        if (i) cout << delimiter;
        cout << vec[i];
    }
    cout << ")" << endl;
}

class MyTimer
{
public:
    MyTimer(void) { reset(); }

    void reset(void) { start = chrono::steady_clock::now(); }
    double elapsed_time(bool reset_start = false) 
    { 
        auto end = chrono::steady_clock::now();
        chrono::duration<double> elapased_seconds = end - start;
        
        if (reset_start)
            start = chrono::steady_clock::now();
        
        return elapased_seconds.count();
    }

private:
    chrono::time_point<chrono::steady_clock> start;
};


#endif