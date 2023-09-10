#pragma once

#include "Noncopyable.h"
#include "OpKernels.cuh"


struct SmokeSim : Noncopyable {
    unsigned int n;
    std::unique_ptr<CudaSurface<float4>> loc;
    std::unique_ptr<CudaTexture<float4>> vel;
    std::unique_ptr<CudaTexture<float4>> velNext;
    std::unique_ptr<CudaTexture<float>> clr;
    std::unique_ptr<CudaTexture<float>> clrNext;
    std::unique_ptr<CudaTexture<float>> tmp;
    std::unique_ptr<CudaTexture<float>> tmpNext;

    std::unique_ptr<CudaSurface<char>> bound;
    std::unique_ptr<CudaSurface<float>> div;
    std::unique_ptr<CudaSurface<float>> pre;
    std::vector<std::unique_ptr<CudaSurface<float>>> res;
    std::vector<std::unique_ptr<CudaSurface<float>>> res2;
    std::vector<std::unique_ptr<CudaSurface<float>>> err2;
    std::vector<unsigned int> sizes;

    explicit SmokeSim(unsigned int _n, unsigned int _n0 = 16)
    : n(_n)
    , loc(std::make_unique<CudaSurface<float4>>(uint3{n, n, n}))
    , vel(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
    , velNext(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
    , clr(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
    , clrNext(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
    , tmp(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
    , tmpNext(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
    , div(std::make_unique<CudaSurface<float>>(uint3{n, n, n}))
    , pre(std::make_unique<CudaSurface<float>>(uint3{n, n, n}))
    , bound(std::make_unique<CudaSurface<char>>(uint3{n, n, n}))
    {
        fillzero_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(pre->accessSurface(), n);

        unsigned int tn;
        for (tn = n; tn >= _n0; tn /= 2) {
            res.push_back(std::make_unique<CudaSurface<float>>(uint3{tn, tn, tn}));
            res2.push_back(std::make_unique<CudaSurface<float>>(uint3{tn/2, tn/2, tn/2}));
            err2.push_back(std::make_unique<CudaSurface<float>>(uint3{tn/2, tn/2, tn/2}));
            sizes.push_back(tn);
        }
    }

    void smooth(CudaSurface<float> *v, CudaSurface<float> *f, unsigned int lev, int times = 4) {
        unsigned int tn = sizes[lev];
        for (int step = 0; step < times; step++) {
            rbgs_kernel<0><<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(v->accessSurface(), f->accessSurface(), tn);
            rbgs_kernel<1><<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(v->accessSurface(), f->accessSurface(), tn);
        }
    }

    void vcycle(unsigned int lev, CudaSurface<float> *v, CudaSurface<float> *f) {
        if (lev >= sizes.size()) {
            unsigned int tn = sizes.back() / 2;
            smooth(v, f, lev);
            return;
        }
        auto *r = res[lev].get();
        auto *r2 = res2[lev].get();
        auto *e2 = err2[lev].get();
        unsigned int tn = sizes[lev];
        smooth(v, f, lev);
        residual_kernel<<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(r->accessSurface(), v->accessSurface(), f->accessSurface(), tn);
        restrict_kernel<<<dim3((tn/2 + 7) / 8, (tn/2 + 7) / 8, (tn/2 + 7) / 8), dim3(8, 8, 8)>>>(r2->accessSurface(), r->accessSurface(), tn/2);
        fillzero_kernel<<<dim3((tn/2 + 7) / 8, (tn/2 + 7) / 8, (tn/2 + 7) / 8), dim3(8, 8, 8)>>>(e2->accessSurface(), tn/2);
        vcycle(lev + 1, e2, r2);
        prolongate_kernel<<<dim3((tn/2 + 7) / 8, (tn/2 + 7) / 8, (tn/2 + 7) / 8), dim3(8, 8, 8)>>>(v->accessSurface(), e2->accessSurface(), tn/2);
        smooth(v, f, lev);
    }

    void projection() {
        heatup_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(vel->accessSurface(), tmp->accessSurface(), clr->accessSurface(), bound->accessSurface(), 0.05f, 0.018f, 0.004f, n);
        divergence_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(vel->accessSurface(), div->accessSurface(), bound->accessSurface(), n);
        vcycle(0, pre.get(), div.get());
        subgradient_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(pre->accessSurface(), vel->accessSurface(), bound->accessSurface(), n);
    }

    void advection() {
        advect_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(vel->accessTexture(), loc->accessSurface(), bound->accessSurface(), n);

        resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(loc->accessSurface(), vel->accessTexture(), velNext->accessSurface(), n);
        resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(loc->accessSurface(), clr->accessTexture(), clrNext->accessSurface(), n);
        resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(loc->accessSurface(), tmp->accessTexture(), tmpNext->accessSurface(), n);
        std::swap(vel, velNext);
        std::swap(clr, clrNext);
        std::swap(tmp, tmpNext);

        decay_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(tmp->accessSurface(), tmpNext->accessSurface(), bound->accessSurface(), std::exp(-0.5f), std::exp(-0.0003f), n);
        decay_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(clr->accessSurface(), clrNext->accessSurface(), bound->accessSurface(), std::exp(-0.05f), std::exp(-0.003f), n);
        std::swap(tmp, tmpNext);
        std::swap(clr, clrNext);
    }

    void step(int times = 16) {
        for (int step = 0; step < times; step++) {
            projection();
            advection();
        }
    }

    float calc_loss() {
        divergence_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(vel->accessSurface(), div->accessSurface(), bound->accessSurface(), n);
        float *sum;
        checkCudaErrors(cudaMalloc(&sum, sizeof(float)));
        sumloss_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(div->accessSurface(), sum, n);
        float cpu;
        checkCudaErrors(cudaMemcpy(&cpu, sum, sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(sum));
        return cpu;
    }
};
