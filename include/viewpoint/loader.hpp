// File: viewpoint/loader.hpp

#ifndef VIEWPOINT_LOADER_HPP
#define VIEWPOINT_LOADER_HPP

#include "viewpoint/provider.hpp"
#include <string>
#include <vector>
#include "core/view.hpp"

namespace viewpoint {

    class Loader : public Provider {
    public:
        explicit Loader(std::string  filepath);
        std::vector<core::View> provision() override;

    private:
        std::string filepath_;
    };

}

#endif // VIEWPOINT_LOADER_HPP

