#ifndef RLIB_IMPL_MPI_ASS
#define RLIB_IMPL_MPI_ASS

#include <mpi.h>

namespace rlib {
    template <typename IntLike, typename StringLike>
    void mpi_assert(IntLike b, StringLike msg) {
        if(b != MPI_SUCCESS) {
            throw std::runtime_error("mpi_assert: assertion error: returns " + std::to_string(b) + ", " + msg);
        }
    }
    template <typename IntLike>
    void mpi_assert(IntLike b) {
        mpi_assert(b, "");
    }


}
#endif
