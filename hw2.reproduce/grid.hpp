#ifndef R267_HW2_GRID_HPP_
#define R267_HW2_GRID_HPP_ 1

// I have 24 hours to finish it. FIGHT!

// NOW I have 24 hours to finish mpi. FIGHT AGAIN!

#include "common.h"
#include <vector>
#include <array>
#include <cmath>
#include <functional>
#include <list>
#include <array>
#include <cassert>
#include <memory>
#include <utility>

#include <thread>
#include <chrono>
#include <algorithm>
using namespace std::literals;

#ifndef DISABLE_MPI
#include "mpi_ass.hpp"
#endif

#if defined(_OPENMP)
#if _OPENMP < 201307
#error OPENMP 4.0 or newer is required. (openmp parallel iter)
#endif
#include <omp.h>
#endif

using std::size_t;

//#include <rlib/stdio.hpp>
//#include <iostream>

namespace r267 {
    static particle_t shit;
    struct grid_info {
        grid_info(std::vector<std::reference_wrapper<particle_t>> &another) : particles(another) {}
        std::vector<std::reference_wrapper<particle_t>> particles;
    };

    using gridded_buffer_t = std::vector<grid_info>; // should use Eigen::Matrix but maybe its not quick enough.
    // The outter vector is fixed-length. The inner vector contains all particle in this grid.
    using buffer_t = std::vector<particle_t>; // All particles in one buffer.

    __attribute__((const)) static inline size_t XY(size_t x, size_t y) {
        // access the matrix `grids` with x,y
        return x*grid_size + y;
    }

    namespace mpi {
        struct grid_info_2 {
            std::vector<size_t> particles_by_offset;
        };

        struct particle_gridded_buffer_t {
            std::vector<grid_info_2> myBuffer;
            size_t global_x, global_y, rank;
        };

        static inline void move_and_reown(const int how_many_proc, const int myRank) {
            // : Report all particle that leaving my area to his new owner.
            // : receive all coming particles!
            const auto recv_thread_func = [](volatile bool &flagStopThread){
                while(not flagStopThread) {
                    int have_msg_flag = 0;
                    MPI_Status stat;
                    rlib::mpi_assert(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &have_msg_flag, &stat));
                    if(have_msg_flag) {
                        double received_msg;
                        rlib::mpi_assert(MPI_Recv(&received_msg, 1, MPI_DOUBLE, stat.MPI_SOURCE, stat.MPI_TAG, MPI_COMM_WORLD, &stat));
                    }
                }
            };
 
            volatile bool flag_stop_recv_thread = false;
            std::thread recv_thread(recv_thread_func, std::ref(flag_stop_recv_thread));

            const size_t new_rank = (myRank + 1) % how_many_proc;

            double packet = 1.1;
            MPI_Status stat;
            rlib::mpi_assert(MPI_Send(&packet, 1, MPI_DOUBLE, new_rank, 7, MPI_COMM_WORLD));


            rlib::mpi_assert(MPI_Barrier(MPI_COMM_WORLD));
            //printf("debug- %u: Killing recv thread\n", buf.rank);
            flag_stop_recv_thread = true;
            recv_thread.join();
        }

        namespace impl {
            static inline auto do_grid_for_mpi(const buffer_t &particles, std::vector<grid_info_2> &grids, size_t grid_x_begin, size_t grid_y_begin, size_t grid_xy_range) {
                auto x_range_begin = cutoff * grid_x_begin,
                     y_range_begin = cutoff * grid_y_begin,
                     x_range_end = cutoff * (grid_x_begin + grid_xy_range),
                     y_range_end = cutoff * (grid_y_begin + grid_xy_range);

                for(auto cter = 0; cter < particles.size(); ++cter) {
                    const auto &particle = particles.at(cter);
                    if(not(particle.x > x_range_begin and particle.y > y_range_begin and particle.x < x_range_end and particle.y < y_range_end))
                        continue;
                    const auto x = std::floor(particle.x / cutoff) - grid_x_begin;
                    const auto y = std::floor(particle.y / cutoff) - grid_y_begin;
                    auto &grid = grids.at(x*grid_xy_range + y);
                    grid.particles_by_offset.emplace_back(cter);
                }
            }
        }

        static inline auto init_my_buffer(const int rank, const int how_many_proc, const buffer_t &particles) {
            static const size_t buffer_global_size = std::sqrt(how_many_proc);
            static const size_t grid_xy_range = std::ceil((float)grid_size / (float)buffer_global_size); 

            particle_gridded_buffer_t res;
            res.rank = rank;
            res.global_x = rank % buffer_global_size;
            res.global_y = rank / buffer_global_size;

            res.myBuffer.resize(grid_xy_range * grid_xy_range);
            impl::do_grid_for_mpi(particles, res.myBuffer, res.global_x * grid_xy_range, res.global_y * grid_xy_range, grid_xy_range);


            return res;
        }

        namespace impl {
            static inline const grid_info_2 *access_fake_mmap(int x, int y, const int buffer_global_size, const int grid_xy_range, const particle_gridded_buffer_t &buf) {
                    if(x==-1) x=0;
                    if(y==-1) y=0;
                    if(x==grid_xy_range) x-=1;
                    if(y==grid_xy_range) y-=1;

                    return &buf.myBuffer.at(x*grid_xy_range+y);

            }

        } // end namespace impl

        static inline void compute_forces(const int how_many_proc, particle_gridded_buffer_t &buf, buffer_t &particles, double *dmin, double *davg, int *navg) {
            for(auto &particle : particles) {
                particle.ax = particle.ay = 0;
            }

            static const size_t buffer_global_size = std::sqrt(how_many_proc);
            static const size_t grid_xy_range = std::ceil((float)grid_size / (float)buffer_global_size); 

            // TODO: communicate and fill buf.neighbors[i].hisShare_data
            // should validate corresponding element in particles.
            //impl::mpi_exchange_shares(buf, particles, grid_xy_range);

            for(auto y = 0; y < grid_xy_range; ++y) {
                if(grid_xy_range*buf.global_y + y >= grid_size)
                    continue;
                for(auto x = 0; x < grid_xy_range; ++x) {
                    if(grid_xy_range*buf.global_x + x >= grid_size)
                        continue;
                    auto &working_grid = buf.myBuffer.at(grid_xy_range*x + y);

                    #define _RLIB_IMPL_R267_AUTOGEN_FUCK_NEIGHBOR(_x, _y) \
                    { \
                        auto ptr_neighbor_grid_info = impl::access_fake_mmap(_x, _y, buffer_global_size, grid_xy_range, buf); \
                            if(ptr_neighbor_grid_info != nullptr) { \
                            for(auto &particle_offset : working_grid.particles_by_offset) { \
                                for(const auto &neighbor_particle_offset : ptr_neighbor_grid_info->particles_by_offset) { \
                                    apply_force(particles.at(particle_offset), particles.at(neighbor_particle_offset), dmin, davg, navg); \
                                } \
                            } \
                        } \
                    }

                    _RLIB_IMPL_R267_AUTOGEN_FUCK_NEIGHBOR(x-1,y)
                    _RLIB_IMPL_R267_AUTOGEN_FUCK_NEIGHBOR(x  ,y)
                    _RLIB_IMPL_R267_AUTOGEN_FUCK_NEIGHBOR(x+1,y)
                    _RLIB_IMPL_R267_AUTOGEN_FUCK_NEIGHBOR(x-1,y-1)
                    _RLIB_IMPL_R267_AUTOGEN_FUCK_NEIGHBOR(x  ,y-1)
                    _RLIB_IMPL_R267_AUTOGEN_FUCK_NEIGHBOR(x+1,y-1)
                    _RLIB_IMPL_R267_AUTOGEN_FUCK_NEIGHBOR(x-1,y+1)
                    _RLIB_IMPL_R267_AUTOGEN_FUCK_NEIGHBOR(x  ,y+1)
                    _RLIB_IMPL_R267_AUTOGEN_FUCK_NEIGHBOR(x+1,y+1)
                }
            }
        }

    } // end namespace mpi
} // end namespace r267

#endif