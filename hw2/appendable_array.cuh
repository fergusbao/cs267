#ifndef RLIB_CUDA_APPENDABLE_ARRAY
#define RLIB_CUDA_APPENDABLE_ARRAY

#include <stdexcept>
#include <cstring>

#include "cuda_ass.cuh"
#include <cstdio>

using std::size_t;

namespace rlib {
    template <typename T>
    struct appendable_stdlayout_array {
        T * mem;
        size_t m_size, cap;

        appendable_stdlayout_array()
            : m_size(0), cap(0), mem(nullptr) {}

        // host-only function. we can write the device-only version easily but not necessary.
        // WARNING: NOT THREAD SAFE! You can make it thread-safe by simply wrap every member
        //  with std::atomic and use atomic fetch_add
        void push_back(T &&ele) {
            // NOT THREAD SAFE!
            if(m_size >= cap) {
                cap *= 2;
                ++cap;
                apply_new_cap();
            }

            mem[m_size] = ele;
            ++m_size;
        }
        void push_back(const T &ele) {
            // NOT THREAD SAFE!
            if(m_size >= cap) {
                cap *= 2;
                ++cap;
                apply_new_cap();
            }

            mem[m_size] = ele;
            ++m_size;
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

        T &operator[](size_t index) {
            return mem[index];
        }
        const T &operator[](size_t index) const {
            return mem[index];
        }

        void reserve(size_t m_size) {
            if(m_size > cap) {
                cap = m_size;
                apply_new_cap();
            }
        }

        void clear() {
            m_size = 0;
        }

        ~appendable_stdlayout_array() {
            if(mem != nullptr)
                rlib::cuda_assert(cudaFree(mem));
        }

    private:
        void apply_new_cap() {
            void *new_mem;
            rlib::cuda_assert(cudaMallocManaged(&new_mem, cap));
            if(new_mem == nullptr)
                throw std::runtime_error("Failed to allocate memory.");
            if(mem != nullptr)
                std::memcpy(new_mem, mem, m_size * sizeof(T));
            mem = (T *)new_mem;
        }
    };
}

#endif


