#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <cstdio>
#include <cuda.h>
#include <stdexcept>

#if __cplusplus > 201103L
#define RLIB_CONSTEXPR constexpr
#else
#define RLIB_CONSTEXPR const
#endif

//
//  saving parameters
//
RLIB_CONSTEXPR int NSTEPS = 1000;
RLIB_CONSTEXPR int SAVEFREQ = 10;

// Recolic: I assume your GPU support 1024 thread per block. If not, edit this constant.
RLIB_CONSTEXPR int CUDA_MAX_THREAD_PER_BLOCK = 1024;

//
// particle data structure
//
typedef struct {
    double x;
    double y;
    double vx;
    double vy;
    double ax;
    double ay;
} particle_t;

//
//  timing routines
//
double read_timer();

//
//  simulation routines
//
void set_size(int n);
void init_particles(int n, particle_t *p);
void apply_force(particle_t &particle, const particle_t &neighbor, double *dmin,
                 double *davg, int *navg);
__device__ void move(particle_t *, double size);

//
//  I/O routines
//
FILE *open_save(char *filename, int n);
void save(FILE *f, int n, particle_t *p);

//
//  argument processing routines
//
int find_option(int argc, char **argv, const char *option);
int read_int(int argc, char **argv, const char *option, int default_value);
char *read_string(int argc, char **argv, const char *option,
                  char *default_value);

// template <typename T> static constexpr auto sq(T val) {return val * val;}

///////////////////////// var in common.cpp
//
//  tuned constants
//
constexpr double density = 0.0005;
constexpr double mass = 0.01;
constexpr double cutoff = 0.01;
constexpr double min_r = (cutoff / 100);
constexpr double dt = 0.0005;

double size;
///////////////////////// var in common.cpp

namespace rlib {
    inline void cuda_assert(cudaError_t err) {
        if(err != cudaError::cudaSuccess)
            throw std::runtime_error("CUDA runtime error: err code is " + std::to_string(err) + ", please refer to https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038");
    }


    //template <typename CudaErrorT, typename StringLike>
    //inline void cuda_assert(CudaErrorT err, StringLike msg) {
    //    if(err != cudaError.cudaSuccess)
    //        throw std::runtime_error("CUDA runtime error: err code is " + std::to_string(err) + ", " + msg + ", please see https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038");
    //}

    //template <typename CudaErrorT>
    //inline void cuda_assert(CudaErrorT err) {
    //    cuda_assert(err, "");
    //}
}

namespace r267 {
    __global__ inline void move_helper(particle_t *particles, double size, size_t buffer_size) {
        int index = threadIdx.x + blockIdx.x * CUDA_MAX_THREAD_PER_BLOCK;
        if(index < buffer_size)
            ::move(particles + index, size);
    }
}

#endif
