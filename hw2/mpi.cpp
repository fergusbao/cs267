#include "common.h"
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "grid.hpp"
#include <cmath>

//#include <rlib/stdio.hpp>
//#include <rlib/string.hpp>
//using namespace rlib::literals;

//
//  benchmarking program
//
int main(int argc, char **argv) {
    int navg, nabsavg = 0;
    double dmin, absmin = 1.0, davg, absavg = 0.0;
    double rdavg, rdmin;
    int rnavg;

    //
    //  process command line parameters
    //
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

    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    {
        int magic = std::sqrt(n_proc);
        if(magic * magic != n_proc)
            throw std::invalid_argument("n_proc must be someInt^2. Or it will be automatically cut to an available number.");
    }

    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen(sumname, "a") : NULL;

    //particle_t *particles = (particle_t *)malloc(n * sizeof(particle_t));
    r267::buffer_t real_buffer(n);
    particle_t *particles = real_buffer.data();

    MPI_Datatype PARTICLE;
    MPI_Type_contiguous(6, MPI_DOUBLE, &PARTICLE);
    static_assert(sizeof(particle_t) == 6*sizeof(double));
    MPI_Type_commit(&PARTICLE);
    //
    //  initialize and distribute the particles (that's fine to leave it
    //  unoptimized)
    //
    set_size(n);
    if (rank == 0)
        init_particles(n, particles);
    
    //rlib::println("DEBUG> n_proc={}, rank={}, hello."_format(n_proc, rank));
    rlib::mpi_assert(MPI_Bcast(real_buffer.data(), n, PARTICLE, 0, MPI_COMM_WORLD), "mpi_bcast");
    rlib::mpi_assert(MPI_Barrier(MPI_COMM_WORLD));

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();
    for (int step = 0; step < NSTEPS; step++) {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        auto myBuffer = r267::mpi::init_my_buffer(rank, n_proc, real_buffer);

        r267::mpi::compute_forces(n_proc, myBuffer, real_buffer, &dmin, &davg, &navg);

        r267::mpi::move_and_reown(n_proc, myBuffer, real_buffer);

        ////
        ////  collect all global data locally (not good idea to do)
        ////
        //MPI_Allgatherv(local, nlocal, PARTICLE, particles, partition_sizes,
        //               partition_offsets, PARTICLE, MPI_COMM_WORLD);

        ////
        ////  save current step if necessary (slightly different semantics than in
        ////  other codes)
        ////
        //if (find_option(argc, argv, "-no") == -1)
        //    if (fsave && (step % SAVEFREQ) == 0)
        //        save(fsave, n, particles);

        ////
        ////  compute all forces
        ////
        //for (int i = 0; i < nlocal; i++) {
        //    local[i].ax = local[i].ay = 0;
        //    for (int j = 0; j < n; j++)
        //        apply_force(local[i], particles[j], &dmin, &davg, &navg);
        //}

        if (find_option(argc, argv, "-no") == -1) {

            MPI_Reduce(&davg, &rdavg, 1, MPI_DOUBLE, MPI_SUM, 0,
                       MPI_COMM_WORLD);
            MPI_Reduce(&navg, &rnavg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dmin, &rdmin, 1, MPI_DOUBLE, MPI_MIN, 0,
                       MPI_COMM_WORLD);

            if (rank == 0) {
                //
                // Computing statistical data
                //
                if (rnavg) {
                    absavg += rdavg / rnavg;
                    nabsavg++;
                }
                if (rdmin < absmin)
                    absmin = rdmin;
            }
        }

        ////
        ////  move particles
        ////
        //for (int i = 0; i < nlocal; i++)
        //    move(local[i]);
    }
    simulation_time = read_timer() - simulation_time;

    if (rank == 0) {
        printf("n = %d, simulation time = %g seconds", n, simulation_time);

        if (find_option(argc, argv, "-no") == -1) {
            if (nabsavg)
                absavg /= nabsavg;
            //
            //  -The minimum distance absmin between 2 particles during the run
            //  of the simulation -A Correct simulation will have particles stay
            //  at greater than 0.4 (of cutoff) with typical values between
            //  .7-.8 -A simulation where particles don't interact correctly
            //  will be less than 0.4 (of cutoff) with typical values between
            //  .01-.05
            //
            //  -The average distance absavg is ~.95 when most particles are
            //  interacting correctly and ~.66 when no particles are interacting
            //
            printf(", absmin = %lf, absavg = %lf", absmin, absavg);
            if (absmin < 0.4)
                printf("\nThe minimum distance is below 0.4 meaning that some "
                       "particle is not interacting");
            if (absavg < 0.8)
                printf("\nThe average distance is below 0.8 meaning that most "
                       "particles are not interacting");
        }
        printf("\n");

        //
        // Printing summary data
        //
        if (fsum)
            fprintf(fsum, "%d %d %g\n", n, n_proc, simulation_time);
    }

    //
    //  release resources
    //
    if (fsum)
        fclose(fsum);
    if (fsave)
        fclose(fsave);

    MPI_Finalize();

    return 0;
}
