#ifndef RLIB_CUDA_APPENDABLE_ARRAY
#define RLIB_CUDA_APPENDABLE_ARRAY

#include <stdexcept>
#include <memory>

using std::size_t;

namespace rlib {
    template <typename T>
    struct appendable_stdlayout_array {
        size_t size, cap;
        T * mem;

        appendable_stdlayout_array()
            : size(0), cap(0), mem(nullptr) {}

        // host-only function. we can write the device-only version easily but not necessary.
        // WARNING: NOT THREAD SAFE! You can make it thread-safe by simply wrap every member
        //  with std::atomic and use atomic fetch_add
        void push_back(T &&ele) {
            // NOT THREAD SAFE!
            if(size >= cap) {
                cap *= 2;
                ++cap;
                apply_new_cap();
            }

            mem[size] = ele;
            ++size;
        }
        void push_back(const T &ele) {
            // NOT THREAD SAFE!
            if(size >= cap) {
                cap *= 2;
                ++cap;
                apply_new_cap();
            }

            mem[size] = ele;
            ++size;
        }

        size_t size() const {
            return size;
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

        void reserve(size_t size) {
            if(size > cap) {
                cap = size;
                apply_new_cap();
            }
        }

        void clear() {
            size = 0;
            if(mem)
                std::free(mem);
        }

    private:
        void apply_new_cap() {
            void *new_mem = std::realloc(mem, cap);
            if(new_mem == nullptr)
                throw std::runtime_error("Failed to allocate memory.");
            mem = new_mem;
        }
    };
}

#endif


