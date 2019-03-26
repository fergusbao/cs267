#include "dist_kv_store.hpp"

#include <upcxx/upcxx.hpp>

int main() {
    upcxx::init();
    kv_store<double, int> kvs(upcxx::rank_me(), upcxx::rank_n());
    try {
        auto [succ, val] = kvs[1.23];
        if(succ)
            rlib::println("FUCK! succ! val is", val);
    }
    catch(std::out_of_range &o) {

    }
    upcxx::barrier(); // without this barrier, the access above may success at rank1.
   
    if(upcxx::rank_me() == 0) {
        kvs.push(1.23, 666);
        kvs.push(6.666, 123);
        kvs.push(0.01, 99);
        kvs.push(1., 111);
        rlib::println("pushed!");
    }
   
    upcxx::barrier();

    auto [succ, val] = kvs[6.666];
    if(not succ)
        rlib::println("not succ!");
    rlib::println(upcxx::rank_me(), val);

    upcxx::finalize();

}
