#ifndef R267_HW2_GRID_HPP_
#define R267_HW2_GRID_HPP_ 1

// I have 24 hours to finish it. FIGHT!

#include "common.h"
#include <vector>
#include <array>
#include <cmath>
#include <functional>

#if defined(_OPENMP)
#if _OPENMP < 201307
#error OPENMP 4.0 or newer is required. (openmp parallel iter)
#endif
#include <omp.h>
#endif

//#include <rlib/stdio.hpp>
#include <iostream>

namespace r267 {
    static particle_t shit;
    struct grid_info {
        grid_info(std::vector<std::reference_wrapper<particle_t>> &another) : particles(another) {}
        std::vector<std::reference_wrapper<particle_t>> particles;
    };

    using gridded_buffer_t = std::vector<grid_info>; // should use Eigen::Matrix but maybe its not quick enough.
    // The outter vector is fixed-length. The inner vector contains all particle in this grid.
    using buffer_t = std::vector<particle_t>; // All particles in one buffer.

    static inline size_t XY(size_t x, size_t y) {
        static const auto grid_size = std::ceil(size / cutoff);
        // access the matrix `grids` with x,y
        return x*grid_size + y;
    }

    static inline gridded_buffer_t do_grid(buffer_t &particles) {
        const auto grid_size = std::ceil(size / cutoff);
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
            static const auto grid_size = std::ceil(size / cutoff);
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
            #pragma omp parallel for
            for(size_t cter = 0; cter < particles.size(); ++cter) {
                auto &particle = particles[cter];
                particle.ax = particle.ay = 0;
            }
            // I'm not sure if grid is sparse. I assume that it's dense.
            const size_t grid_size = std::ceil(size / cutoff);
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
} // end namespace r267

#endif