#ifndef INCLUDE_TRT_CONVERTER_MEMORY_ALLOCATOR_
#define INCLUDE_TRT_CONVERTER_MEMORY_ALLOCATOR_

class Allocator {
    virtual void* Alloc() = 0;
    virtual void Free()  = 0;
};

class CpuAllocator: public Allocator {
    
};

#endif /* INCLUDE_TRT_CONVERTER_MEMORY_ALLOCATOR_ */
