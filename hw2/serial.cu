#include "common.cuh"
#include <cassert>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <utility>
#include <memory>


// RECOLIC: In cuda, we should take special care to compile extern functions.
//  To make things easy, I'm willing to place all of them in one source.
//  THIS IS NOT AN ERROR!!
#include "common.cu"
// NOTE END

#include "appendable_array.cuh"

using dict_element_type = rlib::appendable_stdlayout_array<int>; // Fake vector.


namespace r267 {
  __global__ void move_helper(particle_t * __restrict__  particles, double size, size_t buffer_size) {
    int index = threadIdx.x + blockIdx.x * CUDA_MAX_THREAD_PER_BLOCK;
    if(index < buffer_size)
        ::move(particles + index, size);
  }

  __device__ void apply_force_single_thread(particle_t * __restrict__ particles, size_t my_index, 
      dict_element_type * __restrict__ _dict_buf_ptr, int grid_size,
      double * __restrict__  dmin, double * __restrict__ davg, int * __restrict__  navg) {
    auto i = my_index;
    auto &particle = particles[i];

    int a = floor(particle.x / cutoff);
    int b = floor(particle.y / cutoff);

    particle.ax = particle.ay = 0;

#define RLIB_MACRO_ACCESS_2D_DICT(_x, _y) (_dict_buf_ptr[(_x)*grid_size+(_y)])

    for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a, b).m_size; j++) {
      apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a, b).mem[j]], dmin, davg,
                  navg);
    }
    if (b > 0) {
      for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a, b - 1).m_size; j++) {
        apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a, b - 1).mem[j]], dmin, davg,
                    navg);
      }
    }
    if (b < grid_size - 1) {
      for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a, b + 1).m_size; j++) {
        apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a, b + 1).mem[j]], dmin, davg,
                    navg);
      }
    }
    if (a > 0) {
      for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a - 1, b).m_size; j++) {
        apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a - 1, b).mem[j]], dmin, davg,
                    navg);
      }
      if (b > 0) {
        for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a - 1, b - 1).m_size; j++) {
          apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a - 1, b - 1).mem[j]], dmin,
                      davg, navg);
        }
      }
      if (b < grid_size - 1) {
        for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a - 1, b + 1).m_size; j++) {
          apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a - 1, b + 1).mem[j]], dmin,
                      davg, navg);
        }
      }
    }
    if (a < grid_size - 1) {
      for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a + 1, b).m_size; j++) {
        apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a + 1, b).mem[j]], dmin, davg,
                    navg);
      }
      if (b > 0) {
        for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a + 1, b - 1).m_size; j++) {
          apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a + 1, b - 1).mem[j]], dmin,
                      davg, navg);
        }
      }
      if (b < grid_size - 1) {
        for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a + 1, b + 1).m_size; j++) {
          apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a + 1, b + 1).mem[j]], dmin,
                      davg, navg);
        }
      }
    }
  }

  __global__ void apply_force_helper(particle_t * __restrict__  particles, size_t buffer_size,
      dict_element_type * __restrict__ _dict_buf_ptr, int grid_size, 
      float * __restrict__ _dmin, double * __restrict__ _davg, int * __restrict__ _navg) {
    int navg = 0;
    double dmin = 1.0; double davg = 0;
    int index = threadIdx.x + blockIdx.x * CUDA_MAX_THREAD_PER_BLOCK;
    if(index < buffer_size) {
      apply_force_single_thread(particles, index, _dict_buf_ptr, grid_size, &dmin, &davg, &navg);
      ffatomicMin(_dmin, dmin);
      atomicAdd(_davg, davg);
      atomicAdd(_navg, navg);
    }
    // else return
  }

  __global__ void kernel_clear_dict(dict_element_type * __restrict__ _dict_buf_ptr, int grid_size) {
    // WARNING: index should be 0-grid_size**2, rather than 0-n
    int index = threadIdx.x + blockIdx.x * CUDA_MAX_THREAD_PER_BLOCK;
    RLIB_MACRO_ACCESS_2D_DICT(0, index).clear();
  }

  __global__ void kernel_fill_dicts(dict_element_type * __restrict__ _dict_buf_ptr, int grid_size, particle_t * __restrict__ particles, int n) {
    for(int index = 0; index < n; ++index) {
      int a = floor(particles[index].x / cutoff);
      int b = floor(particles[index].y / cutoff);
      RLIB_MACRO_ACCESS_2D_DICT(a, b).thread_safe_push_back(index);
    }
    return;

    int index = threadIdx.x + blockIdx.x * CUDA_MAX_THREAD_PER_BLOCK;
    int a = floor(particles[index].x / cutoff);
    int b = floor(particles[index].y / cutoff);
    RLIB_MACRO_ACCESS_2D_DICT(a, b).thread_safe_push_back(index);
  }
}

struct _r267_stats {
  int navg;
  double davg;
  float dmin;
};

//
//  benchmarking program
//
int main(int argc, char **argv) {
  int nabsavg = 0;
  double absmin = 1.0, absavg = 0.0;
  if (find_option(argc, argv, "-h") >= 0) {
    printf("Options:\n");
    printf("-h to see this help\n");
    printf("-n <int> to set the number of particles\n");
    printf("-o <filename> to specify the output file name\n");
    printf("-s <filename> to specify a summary file name\n");
    printf("-no turns off all correctness checks and particle output\n");
    return 0;
  }

  int n = read_int(argc, argv, "-n", 1000);

  char *savename = read_string(argc, argv, "-o", NULL);
  char *sumname = read_string(argc, argv, "-s", NULL);

  FILE *fsave = savename ? fopen(savename, "w") : NULL;
  FILE *fsum = sumname ? fopen(sumname, "a") : NULL;

  _r267_stats *r267_stats = nullptr;
  rlib::cuda_assert(cudaMallocManaged(&r267_stats, sizeof(_r267_stats)));

  particle_t *_cuda_managed_particles = nullptr;
  rlib::cuda_assert(cudaMallocManaged(&_cuda_managed_particles, n * sizeof(particle_t)));
  particle_t *particles = new(_cuda_managed_particles) particle_t[n]();

  set_size(n);
  init_particles(n, particles);
  double density = 0.0005;
  double cutoff = 0.01;
  double size = sqrt(density * n);
  int grid_size = floor(size / cutoff + 2);
  std::printf("RDEBUG> grid_size = %d\n", grid_size);
  //std::vector<int> dict[sx][sy];
  //using dict_element_type = std::vector<int>;
  // RECOLIC: FUCKING BRIDGE IS USING GCC 4.8.5 which doesn't support c++14
  //auto _dict_buf_ptr = std::make_unique<dict_element_type[]>(sx*sy);
  auto _dict_buf_ptr = std::unique_ptr<dict_element_type[]>(new dict_element_type[grid_size * grid_size]());


  //
  //  simulate a number of time steps
  //
  double simulation_time = read_timer();

  for (int step = 0; step < NSTEPS; step++) {
    r267_stats->navg = 0;
    r267_stats->davg = 0.0;
    r267_stats->dmin = 1.0;
    //
    //  Update bins
    //
    {
      const auto buffer_size = grid_size * grid_size;
      const auto threads = std::min(buffer_size, CUDA_MAX_THREAD_PER_BLOCK);
      const auto blocks = buffer_size / CUDA_MAX_THREAD_PER_BLOCK + 1;
      r267::kernel_clear_dict<<<blocks, threads>>>(_dict_buf_ptr.get(), grid_size);
    }
    //for (int i = 0; i < grid_size; i++) {
    //  for (int j = 0; j < grid_size; j++) {
    //    RLIB_MACRO_ACCESS_2D_DICT(i, j).clear();
    //  }
    //}

    //for (int i = 0; i < n; i++) {
    //  int a = floor(particles[i].x / cutoff);
    //  int b = floor(particles[i].y / cutoff);
    //  RLIB_MACRO_ACCESS_2D_DICT(a, b).push_back(i);
    //}
    //
    //  compute forces
    //

    //
    //  move particles
    //
    rlib::cuda_assert(cudaDeviceSynchronize());
    r267::kernel_fill_dicts<<<1, 1>>>(_dict_buf_ptr.get(), grid_size, particles, n);
    rlib::cuda_assert(cudaDeviceSynchronize());
    const auto buffer_size = n;
    const auto threads = std::min(buffer_size, CUDA_MAX_THREAD_PER_BLOCK);
    const auto blocks = buffer_size / CUDA_MAX_THREAD_PER_BLOCK + 1;
    //printf("debug: blocks=%d, threads=%d\n", blocks, threads);
    //r267::kernel_fill_dicts<<<blocks, threads>>>(_dict_buf_ptr.get(), grid_size, particles);
    r267::apply_force_helper<<<blocks, threads>>>(particles, buffer_size, _dict_buf_ptr.get(), grid_size, &r267_stats->dmin, &r267_stats->davg, &r267_stats->navg);
    rlib::cuda_assert(cudaDeviceSynchronize());
    //printf("in-kernel debug: dmin=%f\n", r267_stats->dmin);
    r267::move_helper<<<blocks, threads>>>(particles, size, buffer_size);
    rlib::cuda_assert(cudaDeviceSynchronize());
    //rlib::cuda_assert(cudaDeviceSynchronize());
    //for (int i = 0; i < n; i++)
    //  ::move(particles[i]);

    if (find_option(argc, argv, "-no") == -1) {
      //
      // Computing statistical data
      //
      if (r267_stats->navg) {
        absavg += r267_stats->davg / r267_stats->navg;
        nabsavg++;
      }
      if (r267_stats->dmin < absmin)
        absmin = r267_stats->dmin;

    }
  }
  simulation_time = read_timer() - simulation_time;

  printf("n = %d, simulation time = %g seconds", n, simulation_time);

  if (find_option(argc, argv, "-no") == -1) {
    if (nabsavg)
      absavg /= nabsavg;
    //
    //  -The minimum distance absmin between 2 particles during the run of the
    //  simulation -A Correct simulation will have particles stay at greater
    //  than 0.4 (of cutoff) with typical values between .7-.8 -A simulation
    //  where particles don't interact correctly will be less than 0.4 (of
    //  cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting
    //  correctly and ~.66 when no particles are interacting
    //
    printf(", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4)
      printf("\nThe minimum distance is below 0.4 meaning that some particle "
             "is not interacting");
    if (absavg < 0.8)
      printf("\nThe average distance is below 0.8 meaning that most particles "
             "are not interacting");
  }
  printf("\n");

  //
  // Printing summary data
  //
  if (fsum)
    fprintf(fsum, "%d %g\n", n, simulation_time);

  //
  // Clearing space
  //
  if (fsum)
    fclose(fsum);
  rlib::cuda_assert(cudaFree(_cuda_managed_particles));
  rlib::cuda_assert(cudaFree(r267_stats));
  if (fsave)
    fclose(fsave);

  return 0;
}
