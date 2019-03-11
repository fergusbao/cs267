// mpicxx -o mpi -O3 -std=c++14 -g -fsanitize=address -fno-omit-frame-pointer -O0  mpi.cpp

#include <mpi.h>
#include <stdio.h>

#include <thread>
namespace rlib {
    void mpi_assert(int b) {
        if(b != MPI_SUCCESS)
            throw std::runtime_error("fuck");
    }
}

static inline void move_and_reown(const int n_proc, const int rank) {
    const auto recv_thread_func = [](volatile bool &flagStopThread) {
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

    const int new_rank = (rank + 1) % n_proc;

    double packet = 1.1;
    MPI_Status stat;
    rlib::mpi_assert(MPI_Send(&packet, 1, MPI_DOUBLE, new_rank, 7, MPI_COMM_WORLD));


    rlib::mpi_assert(MPI_Barrier(MPI_COMM_WORLD));
    flag_stop_recv_thread = true;
    recv_thread.join();
    printf("%d: recv thread exited.\n", rank);
}

int main(int argc, char **argv) {
    int n_proc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("nproc=%d, rank=%d\n", n_proc, rank);

    //for (int step = 0; step < 1000; step++) {
        move_and_reown(n_proc, rank);
        printf("%d: Loop success\n", rank);
    //}

    MPI_Finalize();

    return 0;
}
