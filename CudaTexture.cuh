#pragma once

#include "CudaArray.cuh"
#include "CudaSurface.cuh"

template <class T>
struct CudaTextureAccessor {
    cudaTextureObject_t m_cuTex;

    __device__ __forceinline__ T sample(float x, float y, float z) const {
        return tex3D<T>(m_cuTex, x, y, z);
    }
};

template <class T>
struct CudaTexture : CudaSurface<T> {
    struct Parameters {
        cudaTextureAddressMode addressMode{cudaAddressModeBorder};
        cudaTextureFilterMode filterMode{cudaFilterModeLinear};
        cudaTextureReadMode readMode{cudaReadModeElementType};
        bool normalizedCoords{false};
    };

    cudaTextureObject_t m_cuTex{};

    explicit CudaTexture(uint3 const &_dim, Parameters const &_args = {})
        : CudaSurface<T>(_dim) {
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = CudaSurface<T>::getArray();

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = _args.addressMode;
        texDesc.addressMode[1] = _args.addressMode;
        texDesc.addressMode[2] = _args.addressMode;
        texDesc.filterMode = _args.filterMode;
        texDesc.readMode = _args.readMode;
        texDesc.normalizedCoords = _args.normalizedCoords;

        checkCudaErrors(cudaCreateTextureObject(&m_cuTex, &resDesc, &texDesc, NULL));
    }

    cudaTextureObject_t getTexture() const {
        return m_cuTex;
    }

    CudaTextureAccessor<T> accessTexture() const {
        return {m_cuTex};
    }

    ~CudaTexture() {
        checkCudaErrors(cudaDestroyTextureObject(m_cuTex));
    }
};
