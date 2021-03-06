#pragma once

#include <upcxx/upcxx.hpp>
#include "kmer_t.hpp"
#include "dist_kv_store.hpp"

namespace std {
  template <>
  struct hash <pkmer_t> {
    size_t operator()(const pkmer_t &p) const {
      return p.hash();
    }
  };
}

#ifndef HASHMAP_SIZE_HINT_FACTOR
#define HASHMAP_SIZE_HINT_FACTOR 8
#endif

struct HashMap {
  HashMap(std::size_t size_hint) : real_db(upcxx::rank_me(), upcxx::rank_n(), (float)size_hint/HASHMAP_SIZE_HINT_FACTOR) {

  }

  bool insert(const kmer_pair &kmer) {
    ++counter;
    rlib::println("inserting hash", std::hash<pkmer_t>{}(kmer.kmer));
    return real_db.set_if_is_mine(kmer.kmer, kmer);
  }
  bool find(const pkmer_t &key_kmer, kmer_pair &val_kmer) {
    rlib::println("finding hash", std::hash<pkmer_t>{}(key_kmer));
    auto res = real_db.get_if_is_mine(key_kmer);
    if(res.first)
      val_kmer = res.second;
    return res.first;
  }
  size_t _debug_get_owner(const pkmer_t &key_kmer) {
    return real_db._debug_get_owner(key_kmer);
  }

  size_t counter = 0;
private:
  kv_store<pkmer_t, kmer_pair> real_db;
};

/*
struct HashMap {
  std::vector <kmer_pair> data;
  std::vector <int> used;

  size_t my_size;

  size_t size() const noexcept;

  HashMap(size_t size);

  // Most important functions: insert and retrieve
  // k-mers from the hash table.
  bool insert(const kmer_pair &kmer);
  bool find(const pkmer_t &key_kmer, kmer_pair &val_kmer);

  // Helper functions

  // Write and read to a logical data slot in the table.
  void write_slot(uint64_t slot, const kmer_pair &kmer);
  kmer_pair read_slot(uint64_t slot);

  // Request a slot or check if it's already used.
  bool request_slot(uint64_t slot);
  bool slot_used(uint64_t slot);
};

inline HashMap::HashMap(size_t size) {
  my_size = size;
  data.resize(size);
  used.resize(size, 0);
}

inline bool HashMap::insert(const kmer_pair &kmer) {
  uint64_t hash = kmer.hash();
  uint64_t probe = 0;
  bool success = false;
  do {
    uint64_t slot = (hash + probe++) % size();
    success = request_slot(slot);
    if (success) {
      write_slot(slot, kmer);
    }
  } while (!success && probe < size());
  return success;
}

inline bool HashMap::find(const pkmer_t &key_kmer, kmer_pair &val_kmer) {
  uint64_t hash = key_kmer.hash();
  uint64_t probe = 0;
  bool success = false;
  do {
    uint64_t slot = (hash + probe++) % size();
    if (slot_used(slot)) {
      val_kmer = read_slot(slot);
      if (val_kmer.kmer == key_kmer) {
        success = true;
      }
    }
  } while (!success && probe < size());
  return success;
}

inline bool HashMap::slot_used(uint64_t slot) {
  return used[slot] != 0;
}

inline void HashMap::write_slot(uint64_t slot, const kmer_pair &kmer) {
  data[slot] = kmer;
}

inline kmer_pair HashMap::read_slot(uint64_t slot) {
  return data[slot];
}

inline bool HashMap::request_slot(uint64_t slot) {
  if (used[slot] != 0) {
    return false;
  } else {
    used[slot] = 1;
    return true;
  }
}

inline size_t HashMap::size() const noexcept {
  return my_size;
}
*/