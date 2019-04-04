#include "dist_kv_store.hpp"

#include <upcxx/upcxx.hpp>

int main() {
    upcxx::init();
    kv_store<double, int> kvs(upcxx::rank_me(), upcxx::rank_n());
    try {
        auto [succ, val] = kvs.get(1.23);
        if(succ)
            throw std::runtime_error("get empty db failed.");
    }
    catch(std::out_of_range &o) {

    }
    upcxx::barrier(); // without this barrier, the access above may success at rank1.
   
    if(upcxx::rank_me() == 0) {
        kvs.set(1.23, 666);
        kvs.set(6.666, 123);
        kvs.set(0.01, 99);
        kvs.set(1., 111);
        rlib::println("pushed!");
    }
   
    upcxx::barrier();

    auto [succ, val] = kvs.get(6.666);
    if(not succ)
        throw std::runtime_error("not succ!");
    if(val != 123)
        throw std::runtime_error("not succ!");

    upcxx::finalize();

}
