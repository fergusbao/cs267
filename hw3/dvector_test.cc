#include "upcxx_dist_vector.hpp"

#include <upcxx/upcxx.hpp>

#include <stdexcept>
#define dynamic_assert(b) do { \
    if(not (b)) \
        throw std::runtime_error(std::string("Assertion failed at ") + __FILE__ + ":" + std::to_string(__LINE__)); \
} while(false) 

struct dataType {
    uint64_t padding;
    int data;
};

int main() {
    upcxx::init();
    upcxx_matrix<dataType> mat(1024);
    size_t rows = 1024;
    mat.set_rows(rows);
    try {
        dynamic_assert(mat.get_cols_of_row(32) == 0);
        mat.get(64, 0);
        dynamic_assert(("Unwanted success! get should fail", false));
    }
    catch(std::out_of_range &o) {

    }
    upcxx::barrier(); // without this barrier, the access above may success at rank1.
   
    if(upcxx::rank_me() == 0) {
        mat.push_to_row(100, dataType{123, 666});
        mat.push_to_row(100, dataType{124, 666});
        mat.push_to_row(100, dataType{125, 666});
        mat.push_to_row(100, dataType{126, 666});
        mat.push_to_row(100, dataType{127, 666});
        mat.push_to_row(101, dataType{777, 555});
        dynamic_assert(mat.get_cols_of_row(100) == 5);
        rlib::println("rank0 pushed!");
    }
    if(upcxx::rank_me() == 1) {
        mat.push_to_row(100101, dataType{123, 666});
        mat.push_to_row(100101, dataType{124, 666});
        mat.push_to_row(100101, dataType{125, 666});
        mat.push_to_row(100101, dataType{126, 666});
        mat.push_to_row(100101, dataType{127, 666});
        mat.push_to_row(101, dataType{777, 555});
        mat.push_to_row(101, dataType{777, 5525});
        mat.push_to_row(101, dataType{777, 5525});
        dynamic_assert(mat.get_cols_of_row(100101) == 5);
        rlib::println("rank0 pushed!");
    }
     
    upcxx::barrier();
    dynamic_assert(mat.get_cols_of_row(101) == 4);

    auto val = mat.get(100101, 4);
    dynamic_assert(val.data == 666 and val.padding == 127);
    val = mat.get(101, 3);
    dynamic_assert(val.padding == 777);

        
    upcxx::finalize();

}
