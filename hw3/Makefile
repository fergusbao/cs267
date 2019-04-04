# source `modules.sh` before attempting to build or run.

# The hashtable slots_per_node will be int(PASSED_SIZE/SIZE_HINT_FACTOR).
# Make it larger if you have more nodes!
# If SIZEHINT is smaller, the program is always quicker, and always consumes more RAM.
SIZEHINT ?= 3.0 # Single process
CXX = upcxx -O3 -std=c++14 -DHASHMAP_SIZE_HINT_FACTOR=$(SIZEHINT) -g

# upcxx-meta PPFLAGS are really CFLAGS to be used during compilation
# upcxx-meta LDFLAGS are really CFLAGS to be used during linking
# upcxx-meta LIBFLAGS are really a combination of LDLIBS and LDFLAGS

CXXFLAGS = `upcxx-meta PPFLAGS` `upcxx-meta LDFLAGS`
LDFLAGS = `upcxx-meta LIBFLAGS`

all: kmer_hash kv_naive_test dvector_naive_test

.PHONY: kmer_hash all

kmer_hash: kmer_hash.cpp kmer_t.hpp pkmer_t.hpp packing.hpp read_kmers.hpp hash_map.hpp butil.hpp
	$(CXX) kmer_hash.cpp -o kmer_hash $(CXXFLAGS) $(LDFLAGS)

clean:
	@rm -fv kmer_hash

kv_naive_test:
	upcxx kv_test.cc -o kv_test -std=c++17 -z muldefs -g

dvector_naive_test:
	upcxx dvector_test.cc -o dvector_test -std=c++17 -z muldefs -g
