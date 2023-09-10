#pragma once

#include <utility>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct VDBWriter {
private:
    struct Impl;
    template<typename T, std::size_t N> struct AddGridImpl {
        VDBWriter *p_grid;
        std::string name;
        const void *base;
        uint32_t size_x, size_y, size_z;
        int32_t min_x, min_y, min_z;
        uint32_t pitch_x, pitch_y, pitch_z;

        void operator() () const;
    };

    const std::unique_ptr<Impl> p_Impl;

public:
    VDBWriter();
    VDBWriter(VDBWriter const &) = delete;
    VDBWriter &operator= (VDBWriter const &) = delete;
    VDBWriter(VDBWriter &&) = delete;
    VDBWriter &operator= (VDBWriter &&) = delete;
    ~VDBWriter();

    template<typename T, std::size_t N, bool normalizedCoords = true>
    void addGrid(const std::string &name, void const *base,
        uint32_t size_x, uint32_t size_y, uint32_t size_z,
        uint32_t pitch_x = 0, uint32_t pitch_y = 0, uint32_t pitch_z = 0) {
        if(!pitch_x) pitch_x = sizeof(T) * N;
        if(!pitch_y) pitch_y = pitch_x * size_x;
        if(!pitch_z) pitch_z = pitch_y * size_y;
        int32_t min_x = normalizedCoords ? -(int32_t)size_x/2 :0;
        int32_t min_y = normalizedCoords ? -(int32_t)size_y/2 :0;
        int32_t min_z = normalizedCoords ? -(int32_t)size_z/2 :0;

        AddGridImpl<T, N>{this, name, base,
            size_x, size_y, size_z,
            min_x, min_y, min_z,
            pitch_x, pitch_y, pitch_z
        }();
    }

    void write(const std::string &path);
};
