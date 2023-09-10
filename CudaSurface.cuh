#pragma once

#include "CudaArray.cuh"

template <class T>
struct CudaSurfaceAccessor {
    cudaSurfaceObject_t m_cuSuf;

    template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
    __device__ __forceinline__ T read(int x, int y, int z) const {
        return surf3Dread<T>(m_cuSuf, x * sizeof(T), y, z, mode);
    }

    template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
    __device__ __forceinline__ void write(T val, int x, int y, int z) const {
        surf3Dwrite<T>(val, m_cuSuf, x * sizeof(T), y, z, mode);
    }
};

template <class T>
struct CudaSurface : CudaArray<T> {
    cudaSurfaceObject_t m_cuSuf{};

    explicit CudaSurface(uint3 const &_dim)
        : CudaArray<T>(_dim) {
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;

        resDesc.res.array.array = CudaArray<T>::getArray();
        checkCudaErrors(cudaCreateSurfaceObject(&m_cuSuf, &resDesc));
    }

    cudaSurfaceObject_t getSurface() const {
        return m_cuSuf;
    }

    CudaSurfaceAccessor<T> accessSurface() const {
        return {m_cuSuf};
    }

    ~CudaSurface() {
        checkCudaErrors(cudaDestroySurfaceObject(m_cuSuf));
    }
};
