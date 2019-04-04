#include <cstdio>
#include <cstdlib>
#include <vector>
#include <list>
#include <set>
#include <numeric>
#include <cstddef>
#include <chrono>
#include <thread>
#include <upcxx/upcxx.hpp>

#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "hash_map.hpp"
#include "upcxx_dist_vector.hpp"

#include "butil.hpp"
using namespace std::chrono_literals;

int main(int argc, char **argv) {
  upcxx::init();

  // TODO: remove this, when you start writing
  // parallel implementation.
  //if (upcxx::rank_n() > 1) {
  //  throw std::runtime_error("Error: parallel implementation not started yet!"
  //    " (remove this when you start working.)");
  //}

  if (argc < 2) {
    BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test]\n");
    upcxx::finalize();
    exit(1);
  }

  std::string kmer_fname = std::string(argv[1]);
  std::string run_type = "";

  if (argc >= 3) {
    run_type = std::string(argv[2]);
  }

  int ks = kmer_size(kmer_fname);

  if (ks != KMER_LEN) {
    throw std::runtime_error("Error: " + kmer_fname + " contains " +
      std::to_string(ks) + "-mers, while this binary is compiled for " +
      std::to_string(KMER_LEN) + "-mers.  Modify packing.hpp and recompile.");
  }

  size_t n_kmers = line_count(kmer_fname);

  // Load factor of 0.5
  size_t hash_table_size = n_kmers * (1.0 / 0.5);
  HashMap hashmap(hash_table_size);

  if (run_type == "verbose") {
    BUtil::print("Initializing hash table of size %d for %d kmers.\n",
      hash_table_size, n_kmers);
  }

  std::vector <kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

  if (run_type == "verbose") {
    BUtil::print("Finished reading kmers.\n");
  }

  auto start = std::chrono::high_resolution_clock::now();

  std::vector <kmer_pair> start_nodes;

  for (auto &kmer : kmers) {
    bool success = hashmap.insert(kmer);
    if (!success) {
      // Recolic: this insert is set_if_is_mine now.
      //throw std::runtime_error("Error: HashMap is full!");
    }

    if (upcxx::rank_me()==0 and kmer.backwardExt() == 'F') {
      start_nodes.push_back(kmer);
    }
  }
  auto end_insert = std::chrono::high_resolution_clock::now();
  upcxx::barrier();

  double insert_time = std::chrono::duration <double> (end_insert - start).count();
  if (run_type != "test") {
    BUtil::print("Finished inserting in %lf\n", insert_time);
  }
  upcxx::barrier();

  auto start_read = std::chrono::high_resolution_clock::now();

  upcxx_matrix<kmer_pair> contigs(hash_table_size);
  if (upcxx::rank_me()==0) {
    contigs.set_rows(start_nodes.size());
    for(auto cter = 0; cter < start_nodes.size(); ++cter) {
      contigs.push_to_row(cter, start_nodes[cter]);
      rlib::println("debug");
    }
    //now broadcast contigs, and broadcast the number of contig inside contigs to every processor
  }
  upcxx::barrier();
  uint64_t contigs_size = contigs.get_rows();
  rlib::println("rank", upcxx::rank_me(), "contigs_size is", contigs_size, "hash_table size", hashmap.counter);
  
  if(upcxx::rank_me() == 0)
    std::this_thread::sleep_for(1h); // for debug
  // the following will be done by every processor
  bool all_done = false;
  while (all_done == false) {
    all_done = true;
    rlib::println("rank", upcxx::rank_me(), "enter loop");
    for (auto row = 0; row < contigs_size; ++row){
      rlib::println("rank", upcxx::rank_me(), "fuckA");
      kmer_pair this_contig_end = contigs.back_of_row(row);
      rlib::println("rank", upcxx::rank_me(), "fuckB");
      if(this_contig_end.forwardExt() != 'F') {
        all_done = false;
        kmer_pair next;
        rlib::println("rank", upcxx::rank_me(), "debug: owner of next kmer is ", hashmap._debug_get_owner(this_contig_end.next_kmer()));
        // TODO FIXME BUG: SHOULD FOUND BUT DOESN"T at rank 1
        bool isMine = hashmap.find(this_contig_end.next_kmer(), next);
        rlib::println("rank", upcxx::rank_me(), "isMine=", isMine);
        if (isMine)
          contigs.push_to_row(row, next);
      }
    }
    rlib::println("All contig iterated. next round...");
  }
  rlib::println("rank", upcxx::rank_me(), "passed main loop");


  auto end_read = std::chrono::high_resolution_clock::now();
  upcxx::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  if (upcxx::rank_me()==0){
    std::chrono::duration <double> read = end_read - start_read;
    std::chrono::duration <double> insert = end_insert - start;
    std::chrono::duration <double> total = end - start;

    int numKmers = 0;
    for(auto cter = 0; cter < contigs_size; ++cter) {
      numKmers += contigs.get_cols_of_row(cter);
    }
    //int numKmers = std::accumulate(contigs.begin(), contigs.end(), 0,
    //  [] (int sum, const std::list <kmer_pair> &contig) {
    //    return sum + contig.size();
    //  });

    if (run_type != "test") {
      BUtil::print("Assembled in %lf total\n", total.count());
    }

    if (run_type == "verbose") {
      printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
        " (%lf read, %lf insert, %lf total)\n", upcxx::rank_me(), contigs_size,
        numKmers, start_nodes.size(), read.count(), insert.count(), total.count());
    }

    if (run_type == "test") {
      std::ofstream fout("test_" + std::to_string(upcxx::rank_me()) + ".dat");
      //for (const auto &contig : contigs) {
      for(auto cter = 0; cter < contigs_size; ++cter) {
        std::list<kmer_pair> real_contig;
        auto len = contigs.get_cols_of_row(cter);
        for(auto c = 0; c < len; ++c)
          real_contig.emplace_back(contigs.get(cter, c));
        fout << extract_contig(real_contig) << std::endl;
      }
      fout.close();
    }
  }
  upcxx::finalize();
  return 0;
}
