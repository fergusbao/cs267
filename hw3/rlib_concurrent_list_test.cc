
#include "rlib_concurrent_list.hpp"
#include "rlib.stdio.min.hpp"

int main() {
    rlib::concurrency::single_list<int> l;
    l.push_back(123);
    l.push_back(666);
    l.push_back(666);
    l.push_back(666);
    l.push_back(666);
    l.push_back(6166);
    l.push_back(666);
    l.push_back(666);
    for(auto &&ele : l) {
        rlib::println(ele);
    }
}
