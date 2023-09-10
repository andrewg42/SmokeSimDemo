#pragma once

#include "CudaArray.cuh"
#include "CudaTexture.cuh"
#include "CudaSurface.cuh"
#include "helper_math.h"


#define calxyz(n)                                    \
    int x = threadIdx.x + blockDim.x * blockIdx.x;   \
    int y = threadIdx.y + blockDim.y * blockIdx.y;   \
    int z = threadIdx.z + blockDim.z * blockIdx.z;   \
    if (x >= n || y >= n || z >= n) return

__global__ void advect_kernel(CudaTextureAccessor<float4> texVel, CudaSurfaceAccessor<float4> sufLoc, CudaSurfaceAccessor<char> sufBound, unsigned int n) {
    calxyz(n);

    auto sample = [] (CudaTextureAccessor<float4> tex, float3 loc) -> float3 {
        float4 vel = tex.sample(loc.x, loc.y, loc.z);
        return make_float3(vel.x, vel.y, vel.z);
    };

    float3 loc = make_float3(x + 0.5f, y + 0.5f, z + 0.5f);
    if (sufBound.read(x, y, z) >= 0) {
        float3 vel1 = sample(texVel, loc);
        float3 vel2 = sample(texVel, loc - 0.5f * vel1);
        float3 vel3 = sample(texVel, loc - 0.75f * vel2);
        loc -= (2.f / 9.f) * vel1 + (1.f / 3.f) * vel2 + (4.f / 9.f) * vel3;
    }
    sufLoc.write(make_float4(loc.x, loc.y, loc.z, 0.f), x, y, z);
}

template <class T>
__global__ void resample_kernel(CudaSurfaceAccessor<float4> sufLoc, CudaTextureAccessor<T> texClr, CudaSurfaceAccessor<T> sufClrNext, unsigned int n) {
    calxyz(n);

    float4 loc = sufLoc.read(x, y, z);
    T clr = texClr.sample(loc.x, loc.y, loc.z);
    sufClrNext.write(clr, x, y, z);
}

__global__ void decay_kernel(CudaSurfaceAccessor<float> sufTmp, CudaSurfaceAccessor<float> sufTmpNext, CudaSurfaceAccessor<char> sufBound, float ambientRate, float decayRate, unsigned int n) {
    calxyz(n);
    if (sufBound.read(x, y, z) < 0) return;

    float txp = sufTmp.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float typ = sufTmp.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float tzp = sufTmp.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float txn = sufTmp.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float tyn = sufTmp.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float tzn = sufTmp.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float tmpAvg = (txp + typ + tzp + txn + tyn + tzn) * (1 / 6.f);
    float tmpNext = sufTmp.read(x, y, z);
    tmpNext = (tmpNext * ambientRate + tmpAvg * (1.f - ambientRate)) * decayRate;
    sufTmpNext.write(tmpNext, x, y, z);
}

__global__ void divergence_kernel(CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<float> sufDiv, CudaSurfaceAccessor<char> sufBound, unsigned int n) {
    calxyz(n);
    if (sufBound.read(x, y, z) < 0) {
        sufDiv.write(0.f, x, y, z);
        return;
    }

    float vxp = sufVel.read<cudaBoundaryModeClamp>(x + 1, y, z).x;
    float vyp = sufVel.read<cudaBoundaryModeClamp>(x, y + 1, z).y;
    float vzp = sufVel.read<cudaBoundaryModeClamp>(x, y, z + 1).z;
    float vxn = sufVel.read<cudaBoundaryModeClamp>(x - 1, y, z).x;
    float vyn = sufVel.read<cudaBoundaryModeClamp>(x, y - 1, z).y;
    float vzn = sufVel.read<cudaBoundaryModeClamp>(x, y, z - 1).z;
    float div = (vxp - vxn + vyp - vyn + vzp - vzn) * 0.5f;
    sufDiv.write(div, x, y, z);
}

__global__ void sumloss_kernel(CudaSurfaceAccessor<float> sufDiv, float *sum, unsigned int n) {
    calxyz(n);

    float div = sufDiv.read(x, y, z);
    atomicAdd(sum, div * div);
}

__global__ void heatup_kernel(CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<float> sufTmp, CudaSurfaceAccessor<float> sufClr, CudaSurfaceAccessor<char> sufBound, float tmpAmbient, float heatRate, float clrRate, unsigned int n) {
    calxyz(n);
    if (sufBound.read(x, y, z) < 0) return;

    float4 vel = sufVel.read(x, y, z);
    float tmp = sufTmp.read(x, y, z);
    float clr = sufClr.read(x, y, z);
    vel.z += heatRate * (tmp - tmpAmbient);
    vel.z -= clrRate * clr;
    sufVel.write(vel, x, y, z);
}

__global__ void subgradient_kernel(CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<char> sufBound, unsigned int n) {
    calxyz(n);
    if (sufBound.read(x, y, z) < 0) return;

    float pxn = sufPre.read<cudaBoundaryModeZero>(x - 1, y, z);
    float pyn = sufPre.read<cudaBoundaryModeZero>(x, y - 1, z);
    float pzn = sufPre.read<cudaBoundaryModeZero>(x, y, z - 1);
    float pxp = sufPre.read<cudaBoundaryModeZero>(x + 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeZero>(x, y + 1, z);
    float pzp = sufPre.read<cudaBoundaryModeZero>(x, y, z + 1);
    float4 vel = sufVel.read(x, y, z);
    vel.x -= (pxp - pxn) * 0.5f;
    vel.y -= (pyp - pyn) * 0.5f;
    vel.z -= (pzp - pzn) * 0.5f;
    sufVel.write(vel, x, y, z);
}

template <int phase>
__global__ void rbgs_kernel(CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float> sufDiv, unsigned int n) {
    calxyz(n);
    if ((x + y + z) % 2 != phase) return;

    float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float div = sufDiv.read(x, y, z);
    float preNext = (pxp + pxn + pyp + pyn + pzp + pzn - div) * (1.f / 6.f);
    sufPre.write(preNext, x, y, z);
}

__global__ void residual_kernel(CudaSurfaceAccessor<float> sufRes, CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float> sufDiv, unsigned int n) {
    calxyz(n);

    float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float pre = sufPre.read(x, y, z);
    float div = sufDiv.read(x, y, z);
    float res = pxp + pxn + pyp + pyn + pzp + pzn - 6.f * pre - div;
    sufRes.write(res, x, y, z);
}

__global__ void restrict_kernel(CudaSurfaceAccessor<float> sufPreNext, CudaSurfaceAccessor<float> sufPre, unsigned int n) {
    calxyz(n);

    float ooo = sufPre.read<cudaBoundaryModeClamp>(x*2, y*2, z*2);
    float ioo = sufPre.read<cudaBoundaryModeClamp>(x*2+1, y*2, z*2);
    float oio = sufPre.read<cudaBoundaryModeClamp>(x*2, y*2+1, z*2);
    float iio = sufPre.read<cudaBoundaryModeClamp>(x*2+1, y*2+1, z*2);
    float ooi = sufPre.read<cudaBoundaryModeClamp>(x*2, y*2, z*2+1);
    float ioi = sufPre.read<cudaBoundaryModeClamp>(x*2+1, y*2, z*2+1);
    float oii = sufPre.read<cudaBoundaryModeClamp>(x*2, y*2+1, z*2+1);
    float iii = sufPre.read<cudaBoundaryModeClamp>(x*2+1, y*2+1, z*2+1);
    float preNext = (ooo + ioo + oio + iio + ooi + ioi + oii + iii);
    sufPreNext.write(preNext, x, y, z);
}

__global__ void fillzero_kernel(CudaSurfaceAccessor<float> sufPre, unsigned int n) {
    calxyz(n);

    sufPre.write(0.f, x, y, z);
}

__global__ void prolongate_kernel(CudaSurfaceAccessor<float> sufPreNext, CudaSurfaceAccessor<float> sufPre, unsigned int n) {
    calxyz(n);

    float preDelta = sufPre.read(x, y, z) * (0.5f / 8.f);
#pragma unroll
    for (int dz = 0; dz < 2; dz++) {
#pragma unroll
        for (int dy = 0; dy < 2; dy++) {
#pragma unroll
            for (int dx = 0; dx < 2; dx++) {
                float preNext = sufPreNext.read<cudaBoundaryModeZero>(x*2+dx, y*2+dy, z*2+dz);
                preNext += preDelta;
                sufPreNext.write<cudaBoundaryModeZero>(preNext, x*2+dx, y*2+dy, z*2+dz);
            }
        }
    }
}