#pragma once

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "Noncopyable.h"

template <class T>
struct CudaArray : Noncopyable {
    cudaArray *m_cuArray{};
    uint3 m_dim{};

    explicit CudaArray(uint3 const &_dim)
        : m_dim(_dim) {
        cudaExtent extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        checkCudaErrors(cudaMalloc3DArray(&m_cuArray, &channelDesc, extent, cudaArraySurfaceLoadStore));
    }

    void copyIn(T const *_data) {
        cudaMemcpy3DParms copy3DParams{};
        copy3DParams.srcPtr = make_cudaPitchedPtr((void *)_data, m_dim.x * sizeof(T), m_dim.x, m_dim.y);
        copy3DParams.dstArray = m_cuArray;
        copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copy3DParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copy3DParams));
    }

    void copyOut(T *_data) {
        cudaMemcpy3DParms copy3DParams{};
        copy3DParams.srcArray = m_cuArray;
        copy3DParams.dstPtr = make_cudaPitchedPtr((void *)_data, m_dim.x * sizeof(T), m_dim.x, m_dim.y);
        copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copy3DParams.kind = cudaMemcpyDeviceToHost;
        //printf("%p %d %d %d\n", _data, m_dim.x, m_dim.y, m_dim.z);
        checkCudaErrors(cudaMemcpy3D(&copy3DParams));
    }

    cudaArray *getArray() const {
        return m_cuArray;
    }

    ~CudaArray() {
        checkCudaErrors(cudaFreeArray(m_cuArray));
    }
};
