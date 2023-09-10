#pragma once

#include <cstddef>
#include <utility>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "Noncopyable.h"

template<typename T>
struct CudaAllocator {
    using value_type = T;
    T *allocate(size_t size) {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr));
        return ptr;
    }

    void deallocate(T *ptr, size_t size=0) {
        checkCudaErrors(cudaFree(ptr));
    }

    template<typename ...Args>
    void construct(T *p, Args &&...args) {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))::new((void*)p)
            T(std::forward<Args>(args)...);
    }

    constexpr bool operator==(const CudaAllocator<T> &other) const { return this == &other; }
};

