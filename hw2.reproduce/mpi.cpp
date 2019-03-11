#include "common.h"
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "grid.hpp"
#include <cmath>


//
//  benchmarking program
//
int main(int argc, char **argv) {
    int n = 1000;

    int n_proc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    {
        int magic = std::sqrt(n_proc);
        if(magic * magic != n_proc)
            throw std::invalid_argument("n_proc must be someInt^2. Or it will be automatically cut to an available number.");
    }


    printf("nproc=%d, rank=%d\n", n_proc, rank);
    //rlib::mpi_assert(MPI_Bcast(real_buffer.data(), n, PARTICLE, 0, MPI_COMM_WORLD), "mpi_bcast");
    rlib::mpi_assert(MPI_Barrier(MPI_COMM_WORLD));

    //for (int step = 0; step < NSTEPS; step++) {

        //rlib::mpi_assert(MPI_Barrier(MPI_COMM_WORLD));
        //    //auto myBuffer = r267::mpi::init_my_buffer(rank, n_proc, real_buffer);

        //rlib::mpi_assert(MPI_Barrier(MPI_COMM_WORLD));
        //    //r267::mpi::compute_forces(n_proc, myBuffer, real_buffer, &dmin, &davg, &navg);
        //rlib::mpi_assert(MPI_Barrier(MPI_COMM_WORLD));

            r267::mpi::move_and_reown(n_proc, rank);
        //rlib::mpi_assert(MPI_Barrier(MPI_COMM_WORLD));

    //}

    MPI_Finalize();

    return 0;
}
