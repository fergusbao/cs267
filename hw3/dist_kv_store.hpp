#ifndef _R267_DIST_KV_STORE_HPP
#define _R267_DIST_KV_STORE_HPP

//! WARNING: This distributed kv-store is not fault-tolrence. 
//! It should ONLY be used in scientific computing task,
//!     RATHER THAN ANY project in production!!!

//! NO NODE FAILURE IS ALLOWED!!!
//! Recolic.

#include <upcxx/upcxx.hpp>
#include <functional>
#include <stdexcept>
#include <mutex>
#include "rlib.stdio.min.hpp"
#include "rlib_concurrent_list.hpp"

#ifndef R267_KVS_DEF_SLOT_PER_NODE
#define R267_KVS_DEF_SLOT_PER_NODE 1024
#endif

template <typename KeyType, typename ValueType, typename HashEngineType = std::hash<KeyType>, typename EqualEngineType = std::equal_to<KeyType>>
class kv_store {
public:
    using this_type = kv_store<KeyType, ValueType, HashEngineType, EqualEngineType>;
    using key_type = KeyType;
    using value_type = ValueType;
    using hash_engine_type = HashEngineType;
    using hash_type = decltype(hash_engine_type{}(key_type{}));
    //using hash_type = std::size_t;
    static_assert(std::is_same<hash_type, std::size_t>(), "Invalid hash function: ISO C++17 standard requires that hash_type is always std::size_t");
    using equal_engine_type = EqualEngineType;
    // kv_type should be default_constructable.

private:
    //using slot_type = std::list<std::pair<key_type, value_type>>;
    using slot_type = rlib::concurrency::single_list<std::pair<key_type, value_type>>;

public:
    kv_store(size_t my_rank, size_t n_rank, size_t slot_per_node = R267_KVS_DEF_SLOT_PER_NODE)
        : local_buf(slot_per_node), my_rank(my_rank), n_rank(n_rank) {
    }

    void set(const key_type &k, const value_type &v) {
        auto target_rank = get_rank_for_hash(get_hash_for_ele(k));
        if(my_rank == target_rank) {
            return do_set(k, v);
        }
        else {
            bool succ = upcxx::rpc(target_rank, std::bind(&this_type::do_rpc_set, this, k, v)).wait();
            if(not succ)
                throw std::runtime_error("RPC set failed.");
        }
    }
    std::pair<bool, value_type> get(const key_type &k) const {
        auto target_rank = get_rank_for_hash(get_hash_for_ele(k));
        if(my_rank == target_rank) {
            auto res = do_get(k, true);
            if(res == nullptr)
                return std::make_pair(false, value_type{});
            else
                return std::make_pair(true, *res);
        }
        else {
            auto res = upcxx::rpc(target_rank, std::bind(&this_type::do_rpc_get, this, k)).wait();
            if(not res.success)
                throw std::runtime_error("RPC get failed.");
            return std::make_pair(res.found, res.val);
        }
    }

    bool set_if_is_mine(const key_type &k, const value_type &v) {
        auto target_rank = get_rank_for_hash(get_hash_for_ele(k));
        if(my_rank == target_rank) {
            do_set(k, v);
        }
        return my_rank == target_rank;
    }
    std::pair<bool, value_type> get_if_is_mine(const key_type &k) {
        auto target_rank = get_rank_for_hash(get_hash_for_ele(k));
        if(my_rank == target_rank) {
            auto res = do_get(k, true);
            if(res == nullptr)
                return std::make_pair(false, value_type{});
            else
                return std::make_pair(true, *res);
        }
        else
            return std::make_pair(false, value_type{});
    }

private:

    bool do_rpc_set(key_type k, value_type v) {
        try {
            do_set(k, v);
            return true;
        }
        catch(std::exception &e) {
            rlib::println(std::cerr, "Error: exception while executing rpc set: ", e.what());
            return false;
        }
    }
    auto do_rpc_get(key_type k) const {
        try {
            const auto *res = do_get(k, true);
            if(res)
                return rpc_get_result{true, true, *res};
            else
                return rpc_get_result{false, true, value_type{}};
        }
        catch(std::exception &e) {
            rlib::println(std::cerr, "Error: exception while executing rpc get: ", e.what());
            return rpc_get_result{false, false, value_type{}};
        }
    }

private:
    struct rpc_get_result {
        bool found, success;
        value_type val;
    };

    void do_set(const key_type &k, const value_type &v) {
        auto &target_ls = get_slot(k);
        {
            for(auto &ele : target_ls) {
                if(equal_engine_type{}(ele.first, k)) {
                    ele.second = v;
                    return; // Done.
                }
            }
            // duplicate element not found. Insert it.
            // target_ls.push_front(std::make_pair(k, v));
            target_ls.push_back(std::make_pair(k, v));
        }
    }

    const value_type *do_get(const key_type &k, bool no_throw = false) const {
        const auto &target_ls = get_slot(k);
        {
            for(const auto &ele : target_ls) {
                if(equal_engine_type{}(ele.first, k))
                    return &ele.second;
            }
        }
        if(not no_throw)
            throw std::out_of_range("Element not found.");
        return nullptr;
    }
    value_type *do_get(const key_type &k, bool no_throw = false) {
        auto &target_ls = get_slot(k);
        {
            for(auto &ele : target_ls) {
                if(equal_engine_type{}(ele.first, k))
                    return &ele.second;
            }
        }
        if(not no_throw)
            throw std::out_of_range("Element not found.");
        return nullptr;
    }

    const auto &get_slot(const key_type &k) const {
        auto hash = get_hash_for_ele(k);
        if(my_rank != get_rank_for_hash(hash)) {
            throw std::invalid_argument("This key doesn't belong to me.");
        }
        auto pos = get_local_slot_num_for_hash(hash);
        return local_buf.at(pos);
    }
    auto &get_slot(const key_type &k) {
        auto hash = get_hash_for_ele(k);
        if(my_rank != get_rank_for_hash(hash)) {
            throw std::invalid_argument("This key doesn't belong to me.");
        }
        auto pos = get_local_slot_num_for_hash(hash);
        return local_buf.at(pos);
    }

private:
    inline auto get_rank_for_hash(hash_type h) const {
        return h % n_rank;
    }
    inline auto get_local_slot_num_for_hash(hash_type h) const {
        // The result is the same in all nodes.
        const auto slot_per_node = local_buf.size();
        return h / n_rank % slot_per_node;
    }
    inline auto get_hash_for_ele(const key_type &k) const {
        hash_type h = hash_engine_type{}(k);
        return h;
    }

    std::vector<slot_type> local_buf;
    size_t my_rank, n_rank;
};

#endif


