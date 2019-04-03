#ifndef R267_GLOBAL_ALLOC_
#define R267_GLOBAL_ALLOC_



// See StackOverflow replies to this answer for important commentary about inheriting from std::allocator before replicating this code.
template <typename T>
class mmap_allocator: public std::allocator<T>
{
public:
    typedef size_t size_type;
    typedef T* pointer;
    typedef const T* const_pointer;

    template<typename _Tp1>
    struct rebind
    {   
        typedef mmap_allocator<_Tp1> other;
    };  

    pointer allocate(size_type size, const void *hint=0)
    {   
        void *ptr = create_shared_memory(size + sizeof(size));
        size_t *size_ptr = reinterpret_cast<size_t *>(ptr);
        *size_ptr = size;
        return (pointer)++size_ptr;
    }   

    void deallocate(pointer ptr, size_type n)
    {   
        size_t *size_ptr = reinterpret_cast<size_t *>(ptr);
        --size_ptr;
        auto res = munmap(size_ptr, *size_ptr);
        if(res == -1) 
            throw std::runtime_error("munmap failed. system error: {}"_format(strerror(errno)));
    }   

    mmap_allocator() throw(): std::allocator<T>() {}
    mmap_allocator(const mmap_allocator &a) throw(): std::allocator<T>(a) { } 
    template <class U>
    mmap_allocator(const mmap_allocator<U> &a) throw(): std::allocator<T>(a) { }
    ~mmap_allocator() throw() { }
};


#endif

