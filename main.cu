#include <cstdio>
#include <cmath>
#include <vector>
#include <memory>
#include <thread>
#include "Noncopyable.h"
#include "OpKernels.cuh"
#include "SmokeSim.h"
#include "writevdb.h"


int main() {
    unsigned int n = 128;
    SmokeSim sim(n);

    {
        std::vector<char> cpu(n * n * n);
        for (int z = 0; z < n; z++) {
            for (int y = 0; y < n; y++) {
                for (int x = 0; x < n; x++) {
                    char sdf1 = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n / 4) < n / 12 ? -1 : 1;
                    char sdf2 = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n * 3 / 4) < n / 6 ? -1 : 1;
                    cpu[x + n * (y + n * z)] = sdf1;
                }
            }
        }
        sim.bound->copyIn(cpu.data());
    }

    {
        std::vector<float> cpu(n * n * n);
        for (int z = 0; z < n; z++) {
            for (int y = 0; y < n; y++) {
                for (int x = 0; x < n; x++) {
                    float den = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n / 4) < n / 12 ? 1.f : 0.f;
                    cpu[x + n * (y + n * z)] = den;
                }
            }
        }
        sim.clr->copyIn(cpu.data());
    }

    {
        std::vector<float> cpu(n * n * n);
        for (int z = 0; z < n; z++) {
            for (int y = 0; y < n; y++) {
                for (int x = 0; x < n; x++) {
                    float tmp = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n / 4) < n / 12 ? 1.f : 0.f;
                    cpu[x + n * (y + n * z)] = tmp;
                }
            }
        }
        sim.tmp->copyIn(cpu.data());
    }

    {
        std::vector<float4> cpu(n * n * n);
        for (int z = 0; z < n; z++) {
            for (int y = 0; y < n; y++) {
                for (int x = 0; x < n; x++) {
                    float vel = std::hypot(x - (int)n / 2, y - (int)n / 2, z - (int)n / 4) < n / 12 ? 0.9f : 0.f;
                    cpu[x + n * (y + n * z)] = make_float4(0.f, 0.f, vel * 0.1f, 0.f);
                }
            }
        }
        sim.vel->copyIn(cpu.data());
    }

    std::vector<std::thread> tpool;
    for (int frame = 1; frame <= 200; frame++) {
        std::vector<float> cpuClr(n * n * n);
        std::vector<float> cpuTmp(n * n * n);
        sim.clr->copyOut(cpuClr.data());
        sim.tmp->copyOut(cpuTmp.data());
        tpool.push_back(std::thread([cpuClr = std::move(cpuClr), cpuTmp = std::move(cpuTmp), frame, n] {
            VDBWriter writer;
            writer.addGrid<float, 1>("density", cpuClr.data(), n, n, n);
            writer.addGrid<float, 1>("temperature", cpuTmp.data(), n, n, n);
            writer.write("/tmp/a" + std::to_string(1000 + frame).substr(1) + ".vdb");
        }));

        printf("frame=%d, loss=%f\n", frame, sim.calc_loss());
        sim.step();
    }

    for (auto &t: tpool) t.join();
    return 0;
}
