#include "writevdb.h"
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>

struct VDBWriter::Impl {
    openvdb::GridPtrVec grids;
};

VDBWriter::VDBWriter(): p_Impl(std::make_unique<Impl>()) {
}

VDBWriter::~VDBWriter() = default;

void VDBWriter::write(const std::string &path) {
    openvdb::io::File(path).write(p_Impl->grids);
}

namespace {
template<typename T, std::size_t N> struct vdbtraits {};

template<> struct vdbtraits<float, 1> {
    using type = openvdb::FloatGrid;
};

template<> struct vdbtraits<float, 3> {
    using type = openvdb::Vec3fGrid;
};

template<typename VecT, typename T, std::size_t ...Is>
VecT help_make_vec(const T *ptr, std::index_sequence<Is...>) {
    return VecT(ptr[Is]...);
}
}

template<typename T, std::size_t N>
void VDBWriter::AddGridImpl<T, N>::operator() () const {
    using GridT = typename vdbtraits<T, N>::type;
    using GridVT = typename GridT::ValueType;

    openvdb::tools::Dense<GridVT> dense(openvdb::Coord(size_x, size_y, size_z), openvdb::Coord(min_x, min_y, min_z));
    for(uint32_t z=0; z<size_z; z++) {
        for(uint32_t y=0; y<size_y; y++) {
            for(uint32_t x=0; x<size_x; x++) {
                auto ptr = reinterpret_cast<const T*>(reinterpret_cast<const char*>(base) + pitch_x*x + pitch_y*y + pitch_z*z);
                dense.setValue(x, y, z, help_make_vec<GridVT>(ptr, std::make_index_sequence<N>{}));
            }
        }
    }

    auto grid = GridT::create();
    GridVT tolerance{0};
    openvdb::tools::copyFromDense(dense, grid->tree(), tolerance);
    openvdb::MetaMap &meta = *grid;
    meta.insertMeta(openvdb::Name("name"), openvdb::TypedMetadata<std::string>(name));
    p_grid->p_Impl->grids.push_back(grid);
}

template struct VDBWriter::AddGridImpl<float, 1>;
template struct VDBWriter::AddGridImpl<float, 3>;

