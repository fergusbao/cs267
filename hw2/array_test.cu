#include "appendable_array.cuh"
#include <rlib/stdio.hpp>

int main() {
    rlib::appendable_stdlayout_array<int> ar;
    ar.push_back(1);
    ar.push_back(2);
    ar.push_back(3);
    ar.push_back(6);
    ar.push_back(5);
    ar.push_back(4);
    ar.push_back(1);

    for(auto cter = 0; cter < ar.size(); ++cter) {
        rlib::println(ar[cter]);
    }
}

