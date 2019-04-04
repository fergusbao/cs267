#ifndef UPCXX_DIST_VECTOR_HPP
#define UPCXX_DIST_VECTOR_HPP

#include <upcxx/upcxx.hpp>

namespace fucking {
    struct key_type {
        uint64_t d1, d2;
        bool operator==(const key_type &another) const {
            return d1 == another.d1 and d2 == another.d2;
        }
    };
}
namespace std {
    template <>
    struct hash <fucking::key_type> {
        hash() {}
        size_t operator()(const fucking::key_type &p) {
            // This function is REALLY REALLY IMPORTANT to minimize the communication cost
            //     in vector operations of `contigs`.
            if(p.d2 + 1 == 0) return 0;
            else return std::hash<uint64_t>{}(p.d2>>4) ^ p.d1;
        }
    };
}

#include "dist_kv_store.hpp"
#include "kmer_t.hpp"

//template <typename T>
//class dist_vector {
//public:
//    dist_vector() : cap(0), size(0) {
//
//    }
//
//private:
//    void grow() {
//        cap += 1;
//
//    }
//
//    size_t cap, size;
//    upcxx::global_ptr<T> data;
//};

#include <numeric>
#include <cstdint>
using std::uint64_t;

template <typename T>
class upcxx_matrix {
    using key_type = fucking::key_type;
public:
    upcxx_matrix(size_t hash_table_size) : buf(upcxx::rank_me(), upcxx::rank_n(), hash_table_size) {}
    static_assert(sizeof(T) >= sizeof(uint64_t), "T should larger than 64b");
    void set_rows(uint64_t val) {
        // set size of col 0
        set_size_ele(SIZE_SLOT, 0, val);
    }
    uint64_t get_rows() {
        return get_size_ele(SIZE_SLOT, 0);
    }
    void push_to_row(uint64_t row_index, const T &data) {
        std::cout << "rank" << upcxx::rank_me() << "push_to:" << row_index << std::endl;
        // WARNING WARNING WARNING!!! NOT PROCESS SAFE!!! NOT THREAD SAFE!!!
        auto curr_size = get_cols_of_row(row_index);
        set(row_index, curr_size, data);
        // Warning: possible corruption here. 
        set_cols_of_row(row_index, curr_size + 1);
    }
    uint64_t get_cols_of_row(uint64_t row_index) const {
        return get_size_ele(row_index, SIZE_SLOT);
    }
    void set_cols_of_row(uint64_t row_index, uint64_t cols) {
        set_size_ele(row_index, SIZE_SLOT, cols);
    }
    auto back_of_row(uint64_t row_index) const {
        return get(row_index, get_cols_of_row(row_index)-1);
    }

    auto get(uint64_t row, uint64_t col) const {
        rlib::println("get:", row, col);
        auto res = buf.get(index2key(row, col));
        if(res.first == false)
            throw std::out_of_range("matrix::get index out of range");
        return res.second;
    }
    void set(uint64_t row, uint64_t col, const T &data) {
        buf.set(index2key(row, col), data);
    }
    uint64_t get_size_ele(uint64_t row, uint64_t col) const {
        try {
            auto res = get(row, col);
            auto converted = *(const uint64_t *)&res;
            rlib::println("get_size_ele at", row, col, "yields", converted);
            return converted;
        }
        catch(std::out_of_range &) {
            rlib::println("get_size_ele at", row, col, "yields", 0);
            return 0;
        }
    }
    void set_size_ele(uint64_t row, uint64_t col, uint64_t val) {
        set(row, col, *(T *)&val);
    }

private:
    kv_store<key_type, T> buf;

    static constexpr uint64_t SIZE_SLOT = (uint64_t)0xffffffffffffffff;
    __attribute__((const)) key_type index2key(uint64_t x, uint64_t y) const {
        return key_type{x, y};
    }
};


#endif