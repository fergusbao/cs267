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

#if defined(_OPENMP)
#if _OPENMP < 201307
#error OPENMP 4.0 or newer is required. (openmp parallel iter)
#endif
#include <omp.h>
#endif

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

    static inline gridded_buffer_t do_grid(buffer_t &particles) {
        std::vector<std::reference_wrapper<particle_t>> _shit(0, std::ref(shit));
        gridded_buffer_t grids(grid_size * grid_size, grid_info(_shit));

        // memory bound. DO NOT apply omp please.
        for(auto &particle : particles) {
            const auto x = std::floor(particle.x / cutoff);
            const auto y = std::floor(particle.y / cutoff);
            auto &grid = grids[XY(x, y)];
            grid.particles.emplace_back(std::ref(particle));
        }

        return grids;
    }

    namespace serial {
        static inline void compute_forces(const gridded_buffer_t &grids, buffer_t &particles, double *dmin, double *davg, int *navg) {
            for(auto &particle : particles) {
                particle.ax = particle.ay = 0;
            }
            // I'm not sure if grid is sparse. I assume that it's dense.
            for(auto y = 0; y < grid_size; ++y) {
                for(auto x = 0; x < grid_size; ++x) {
                    auto &grid = grids[XY(x, y)];
                    // do_apply_force
                    // too lazy to write a new function.
                    //rlib::println("debug: particle in grid =", grid.particles.size());
                    #define FUCK_NEIGHBOR(_x, _y) \
                        if(_x >= 0 and _y >= 0 and _x < grid_size and _y < grid_size) { \
                            for(const auto &neighbor : grids[XY(_x, _y)].particles) {   \
                                for(auto &particle : grid.particles)                    \
                                    apply_force(particle, neighbor, dmin, davg, navg);  \
                            }                                                           \
                        }

                    FUCK_NEIGHBOR(x, y-1)
                    FUCK_NEIGHBOR(x, y)
                    FUCK_NEIGHBOR(x, y+1)
                    FUCK_NEIGHBOR(x+1, y-1)
                    FUCK_NEIGHBOR(x+1, y)
                    FUCK_NEIGHBOR(x+1, y+1)
                    FUCK_NEIGHBOR(x-1, y-1)
                    FUCK_NEIGHBOR(x-1, y)
                    FUCK_NEIGHBOR(x-1, y+1)
                    // do_apply_force end
                }
            }
        }

        static inline void move_them(buffer_t &particles) {
            // gridded_buffer will be invalidated!
            for(auto &particle : particles)
                ::move(particle);
        }
    } // end namespace serial

#if defined(_OPENMP)
    namespace omp {
        static inline void compute_forces(const gridded_buffer_t &grids, buffer_t &particles, double *dmin, double *davg, int *navg) {
            for(auto &particle : particles) {
                particle.ax = particle.ay = 0;
            }
            // I'm not sure if grid is sparse. I assume that it's dense.
            #pragma omp parallel for schedule(dynamic, 1)
            for(size_t y = 0; y < grid_size; ++y) {
                for(auto x = 0; x < grid_size; ++x) {
                    auto &grid = grids[XY(x, y)];
                    // do_apply_force
                    // too lazy to write a new function.
                    //rlib::println("debug: particle in grid =", grid.particles.size());
                    #define FUCK_NEIGHBOR(_x, _y) \
                        if(_x >= 0 and _y >= 0 and _x < grid_size and _y < grid_size) { \
                            for(const auto &neighbor : grids[XY(_x, _y)].particles) {   \
                                for(auto &particle : grid.particles)                    \
                                    apply_force(particle, neighbor, dmin, davg, navg);  \
                            }                                                           \
                        }

                    FUCK_NEIGHBOR(x, y-1)
                    FUCK_NEIGHBOR(x, y)
                    FUCK_NEIGHBOR(x, y+1)
                    FUCK_NEIGHBOR(x+1, y-1)
                    FUCK_NEIGHBOR(x+1, y)
                    FUCK_NEIGHBOR(x+1, y+1)
                    FUCK_NEIGHBOR(x-1, y-1)
                    FUCK_NEIGHBOR(x-1, y)
                    FUCK_NEIGHBOR(x-1, y+1)
                    // do_apply_force end
                }
            }
        }

        static inline void move_them(buffer_t &particles) {
            // gridded_buffer will be invalidated!
            #pragma omp parallel for
            for(size_t cter = 0; cter < particles.size(); ++cter)
                ::move(particles[cter]);
        }
    } // end namespace omp
#endif // defined _OPENMP

    namespace mpi {
        struct grid_info_2 {
            std::vector<size_t> particles_by_offset;
        };
        struct neighbor_t {
            int rank;
            size_t myShare_begin_index;
            size_t myShare_step;
            // MyShare := begin_index + step * [0,1,2,...,grid_size]
            int hisShare_begin_x; // grid_x
            int hisShare_begin_y; // grid_y
            bool hisShare_step_direction; // true for x, false for y
            std::vector<particle_t> hisShare_data;
        };
        struct partical_gridded_buffer_t {
            std::vector<grid_info_2> myBuffer;
            size_t global_x, global_y, rank;
            std::list<neighbor_t> neighbors;
        };

        static inline auto do_grid_for_mpi(const buffer_t &particles, std::vector<grid_info_2> &grids, size_t grid_x_begin, size_t grid_y_begin, size_t grid_xy_range) {
            auto x_range_begin = cutoff * grid_x_begin,
                 y_range_begin = cutoff * grid_y_begin,
                 x_range_end = cutoff * (grid_x_begin + grid_xy_range),
                 y_range_end = cutoff * (grid_y_begin + grid_xy_range);

            // memory bound. DO NOT apply omp please.
            for(auto cter = 0; cter < particles.size(); ++cter) {
                const auto &particle = particles[cter];
                if(not(particle.x > x_range_begin and particle.y > y_range_begin and particle.x < x_range_end and particle.y < y_range_end))
                    continue;
                const auto x = std::floor(particle.x / cutoff) - grid_x_begin;
                const auto y = std::floor(particle.y / cutoff) - grid_y_begin;
                auto &grid = grids[x*grid_xy_range + y];
                grid.particles_by_offset.emplace_back(cter);
            }
        }

        static inline auto init_my_buffer(const int rank, const int how_many_proc, const buffer_t &particles) {
            static const size_t buffer_global_size = std::sqrt(how_many_proc);
            partical_gridded_buffer_t res;
            res.rank = rank;
            res.global_x = rank % buffer_global_size;
            res.global_y = rank / buffer_global_size;

            static const size_t grid_xy_range = std::ceil((float)grid_size / (float)buffer_global_size); 
            res.myBuffer.resize(grid_xy_range * grid_xy_range);
            do_grid_for_mpi(particles, res.myBuffer, res.global_x * grid_xy_range, res.global_y * grid_xy_range, grid_xy_range);

            //TODO: add fucking neighbors.
            if(res.global_x != 0) {
                // left
                neighbor_t n {
                    .rank = rank - 1,
                    .myShare_begin_index = 0,
                    .myShare_step = grid_xy_range,
                    .hisShare_begin_x = -1,
                    .hisShare_begin_y = 0,
                    .hisShare_step_direction = false
                };
                res.neighbors.push_back(n);
            }
            if(res.global_x != buffer_global_size - 1) {
                // right
                neighbor_t n {
                    .rank = rank + 1,
                    .myShare_begin_index = grid_xy_range - 1,
                    .myShare_step = grid_xy_range,
                    .hisShare_begin_x = (int)grid_xy_range,
                    .hisShare_begin_y = 0,
                    .hisShare_step_direction = false
                };
                res.neighbors.push_back(n);
            }
            if(res.global_y != 0) {
                // up
                neighbor_t n {
                    .rank = rank - (int)buffer_global_size,
                    .myShare_begin_index = 0,
                    .myShare_step = 1,
                    .hisShare_begin_x = 0,
                    .hisShare_begin_y = -1,
                    .hisShare_step_direction = true
                };
                res.neighbors.push_back(n);
            }
            if(res.global_y != buffer_global_size - 1) {
                // down
                neighbor_t n {
                    .rank = rank + (int)buffer_global_size,
                    .myShare_begin_index = grid_xy_range * grid_xy_range - 1,
                    .myShare_step = 1,
                    .hisShare_begin_x = 0,
                    .hisShare_begin_y = (int)grid_xy_range,
                    .hisShare_step_direction = true
                };
                res.neighbors.push_back(n);
            }

            return res;
        }

        static inline void compute_forces(const int rank, const partical_gridded_buffer_t &grids, buffer_t &particles, double *dmin, double *davg, int *navg) {
            for(auto &particle : particles) {
                particle.ax = particle.ay = 0;
            }

        }
    }
} // end namespace r267

#endif