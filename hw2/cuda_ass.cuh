#ifndef RLIB_CUDA_ASSERT_CUH
#define RLIB_CUDA_ASSERT_CUH

namespace rlib {
    inline void cuda_assert(cudaError_t err) {
        if(err != cudaError::cudaSuccess)
            throw std::runtime_error("CUDA runtime error: err code is " + std::to_string(err) + ", please refer to https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038");
    }
    __device__ bool dev_cuda_assert(cudaError_t err) {
        if(err != cudaError::cudaSuccess) {
            printf("FUCK! CUDA runtime error: %d, %s\n", err, cudaGetErrorString(err));
            return false;
        }
        return true;
    }
}


#endif