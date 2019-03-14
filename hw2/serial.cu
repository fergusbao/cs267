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

//
//  benchmarking program
//
int main(int argc, char **argv) {
  int navg, nabsavg = 0;
  double davg, dmin, absmin = 1.0, absavg = 0.0;

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

  particle_t *_cuda_managed_particles = nullptr;
  rlib::cuda_assert((cudaError_t)cudaMallocManaged(&_cuda_managed_particles, n * sizeof(particle_t)));
  particle_t *particles = new(_cuda_managed_particles) particle_t[n]();

  set_size(n);
  init_particles(n, particles);
  double density = 0.0005;
  double cutoff = 0.01;
  double size = sqrt(density * n);
  int sx = floor(size / cutoff + 2);
  int sy = sx;
  std::printf("RDEBUG> grid_size = %d\n", sx);
  //std::vector<int> dict[sx][sy];
  using dict_element_type = std::vector<int>;
  // RECOLIC: FUCKING BRIDGE IS USING GCC 4.8.5 which doesn't support c++14
  //auto _dict_buf_ptr = std::make_unique<dict_element_type[]>(sx*sy);
  auto _dict_buf_ptr = std::unique_ptr<dict_element_type[]>(new dict_element_type[sx*sy]());

#define RLIB_MACRO_ACCESS_2D_DICT(_x, _y) (_dict_buf_ptr[(_x)*sx+(_y)])

  //
  //  simulate a number of time steps
  //
  double simulation_time = read_timer();

  for (int step = 0; step < NSTEPS; step++) {
    navg = 0;
    davg = 0.0;
    dmin = 1.0;
    //
    //  Update bins
    //
    for (int i = 0; i < sx; i++) {
      for (int j = 0; j < sy; j++) {
        RLIB_MACRO_ACCESS_2D_DICT(i, j).clear();
      }
    }

    for (int i = 0; i < n; i++) {
      int a = floor(particles[i].x / cutoff);
      int b = floor(particles[i].y / cutoff);
      RLIB_MACRO_ACCESS_2D_DICT(a, b).push_back(i);
    }
    //
    //  compute forces
    //
    for (int i = 0; i < n; i++) {
      auto &particle = particles[i];

      int a = floor(particle.x / cutoff);
      int b = floor(particle.y / cutoff);

      particle.ax = particle.ay = 0;

      for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a, b).size(); j++) {
        apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a, b)[j]], &dmin, &davg,
                    &navg);
      }
      if (b > 0) {
        for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a, b - 1).size(); j++) {
          apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a, b - 1)[j]], &dmin, &davg,
                      &navg);
        }
      }
      if (b < sy - 1) {
        for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a, b + 1).size(); j++) {
          apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a, b + 1)[j]], &dmin, &davg,
                      &navg);
        }
      }
      if (a > 0) {
        for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a - 1, b).size(); j++) {
          apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a - 1, b)[j]], &dmin, &davg,
                      &navg);
        }
        if (b > 0) {
          for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a - 1, b - 1).size(); j++) {
            apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a - 1, b - 1)[j]], &dmin,
                        &davg, &navg);
          }
        }
        if (b < sy - 1) {
          for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a - 1, b + 1).size(); j++) {
            apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a - 1, b + 1)[j]], &dmin,
                        &davg, &navg);
          }
        }
      }
      if (a < sx - 1) {
        for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a + 1, b).size(); j++) {
          apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a + 1, b)[j]], &dmin, &davg,
                      &navg);
        }
        if (b > 0) {
          for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a + 1, b - 1).size(); j++) {
            apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a + 1, b - 1)[j]], &dmin,
                        &davg, &navg);
          }
        }
        if (b < sy - 1) {
          for (int j = 0; j < RLIB_MACRO_ACCESS_2D_DICT(a + 1, b + 1).size(); j++) {
            apply_force(particle, particles[RLIB_MACRO_ACCESS_2D_DICT(a + 1, b + 1)[j]], &dmin,
                        &davg, &navg);
          }
        }
      }
    }

    //
    //  move particles
    //
    r267::move_helper<<<1, n>>>(particles, size);
    //for (int i = 0; i < n; i++)
    //  ::move(particles[i]);

    if (find_option(argc, argv, "-no") == -1) {
      //
      // Computing statistical data
      //
      if (navg) {
        absavg += davg / navg;
        nabsavg++;
      }
      if (dmin < absmin)
        absmin = dmin;

      //
      //  save if necessary
      //
      if (fsave && (step % SAVEFREQ) == 0)
        save(fsave, n, particles);
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
  if (fsave)
    fclose(fsave);

  return 0;
}
