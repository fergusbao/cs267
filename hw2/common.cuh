#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <cstdio>
#include <cuda.h>
#include <stdexcept>

#include "cuda_ass.cuh"

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
__device__ void apply_force(particle_t &particle, const particle_t &neighbor, double * __restrict__  dmin, double * __restrict__ davg, int * __restrict__  navg);
__device__ void move(particle_t * __restrict__ , double size);

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


#define RLIB_IMPL_CUDA_FOR(counter_var, counter_var_begin, counter_var_end) \
    do { \
        auto _rlib_impl_range_size = counter_var_end - counter_var_begin; \
\
    } while(false)

#define RLIB_CUDA_FOR(counter_var_name, counter_var_begin, counter_var_end) RLIB_IMPL_CUDA_FOR(counter_var_name, RLIB_MACRO_DECAY(counter_var_begin), RLIB_MACRO_DECAY(counter_var_end))

__device__ double fatomicMin(double *addr, double value)
{
    static_assert(sizeof(double) == sizeof(unsigned long long), "fuck");
    double _old = *addr;
    unsigned long long old = (unsigned long long)_old;
    unsigned long long assumed;
    if(_old <= value) return _old;
    do
    {
        assumed = old;
        old = atomicCAS((unsigned long long *)addr, assumed, value);
    } while(old!=assumed);
    return (double)old;
}

#endif
