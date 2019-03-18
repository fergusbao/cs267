#ifndef RLIB_CUDA_APPENDABLE_ARRAY
#define RLIB_CUDA_APPENDABLE_ARRAY

#include <stdexcept>
#include <cstring>

#include "cuda_ass.cuh"
#include <cstdio>

namespace rlib {
    template <typename T>
    struct appendable_stdlayout_array {
        T * mem;
        // WARNING: NO CPU MEM ACCESS NOW!!!
        unsigned long long m_size, cap;
        int *pLock;

        appendable_stdlayout_array()
            : m_size(0), cap(0), mem(nullptr) {
                rlib::cuda_assert(cudaMallocManaged(&pLock, sizeof(int)));
                *pLock = 0;
            }

        // host-only function. we can write the device-only version easily but not necessary.
        // WARNING: NOT THREAD SAFE! You can make it thread-safe by simply wrap every member
        //  with std::atomic and use atomic fetch_add
        void push_back(T &&ele) {
            // NOT THREAD SAFE!
            if(m_size >= cap) {
                cap *= 2;
                cap += 3;
                apply_new_cap();
            }

            mem[m_size] = ele;
            ++m_size;
        }
        void push_back(const T &ele) {
            // NOT THREAD SAFE!
            if(m_size >= cap) {
                cap *= 2;
                cap += 3;
                apply_new_cap();
            }

            mem[m_size] = ele;
            ++m_size;
        }

#define RLIB_IMPL_CUDA_ENLOCK(pLock) {while(atomicCAS(pLock, 0, 1) != 0);}
#define RLIB_IMPL_CUDA_TRYLOCK(pLock) (!atomicCAS(pLock, 0, 1)) // yields 1 (true) if success, 0 (false) if fail.
#define RLIB_IMPL_CUDA_UNLOCK(pLock) atomicExch(pLock, 0);

        __device__ void thread_safe_push_back(T ele) {
            // THREAD SAFE.
        _rlib_impl_cuda_check_again:
            if(m_size >= cap) {
                if(RLIB_IMPL_CUDA_TRYLOCK(pLock)) {
                    cap *= 2;
                    cap += 4;
                    bool res = dev_apply_new_cap();
                    if(res == false) printf("FUCK!\n");
                    RLIB_IMPL_CUDA_UNLOCK(pLock);
                }
                else {
                    RLIB_IMPL_CUDA_ENLOCK(pLock);
                    RLIB_IMPL_CUDA_UNLOCK(pLock);
                    goto _rlib_impl_cuda_check_again;
                }

            }

            auto snapped_size = atomicAdd(&m_size, 1);
            mem[snapped_size] = ele;
        }

        size_t size() const {
            return m_size;
        }

        T * data() {
            return mem;
        }
        const T *data() const {
            return mem;
        }

        __device__ T &operator[](size_t index) {
            return mem[index];
        }
        __device__ const T &operator[](size_t index) const {
            return mem[index];
        }

        void reserve(size_t m_size) {
            if(m_size > cap) {
                cap = m_size;
                apply_new_cap();
            }
        }

        __host__ __device__ void clear() {
            m_size = 0;
        }

        ~appendable_stdlayout_array() {
            if(mem != nullptr)
                rlib::cuda_assert(cudaFree(mem));
            // Never free memory to make program faster.
        }

    private:
        __host__ void apply_new_cap() {
            void *new_mem;
            rlib::cuda_assert(cudaMalloc(&new_mem, cap * sizeof(T)));
            if(new_mem == nullptr)
                throw std::runtime_error("Failed to allocate memory.");
            if(mem != nullptr) {
                rlib::cuda_assert(cudaMemcpy(new_mem, mem, m_size * sizeof(T)));
                rlib::cuda_assert(cudaFree(mem));
            }
            mem = (T *)new_mem;
        }

        __device__ bool dev_apply_new_cap() {
            void *new_mem = nullptr;
            printf("Allocating %llu b mem\n", cap);
            rlib::dev_cuda_assert(cudaMalloc(&new_mem, cap * sizeof(T)));
            if(new_mem == nullptr)
                return false;
            if(mem != nullptr) {
                memcpy(new_mem, mem, m_size * sizeof(T));
                // hope it success.......
                //(cudaFree(mem)); // let it leak!!! It doesn't matter now.
            }
            mem = (T *)new_mem;
            return true;
        }


    public:
        // Make this class available in cudaManaged memory automatically.
        static void *operator new(size_t size) {
            void *ptr = nullptr;
            rlib::cuda_assert(cudaMallocManaged(&ptr, size));
            return ptr;
        }
        static void *operator new[](size_t size) {
            return operator new(size);
        }
        static void operator delete(void *ptr) {
            rlib::cuda_assert(cudaFree(ptr));
        }
        static void operator delete[](void *ptr) {
            return operator delete(ptr);
        }
    };

}

#endif


