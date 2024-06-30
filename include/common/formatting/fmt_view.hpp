// File: common/formatting/fmt_view.hpp

#ifndef FMT_VIEW_HPP
#define FMT_VIEW_HPP

#include <fmt/core.h>
#include <fmt/format.h>
#include <Eigen/Core>

#include "core/view.hpp"

template<>
struct fmt::formatter<core::View> {
    // Parses format specifications, none needed here
    constexpr auto parse(format_parse_context &ctx) {
        return ctx.begin();
    }

    // Formats the View object
    template<typename FormatContext>
    auto format(const core::View &view, FormatContext &ctx) {
        Eigen::Vector3f position = view.getPosition();
        Eigen::Vector3f object_center = view.getObjectCenter();
        Eigen::Matrix4f pose = view.getPose();
        Eigen::Matrix3f rotation = pose.block<3, 3>(0, 0);

        return fmt::format_to(
                ctx.out(),
                "View:\n"
                "Position: ({:.2f}, {:.2f}, {:.2f})\n"
                "Object Center: ({:.2f}, {:.2f}, {:.2f})\n"
                "Rotation:\n"
                "[{:.2f}, {:.2f}, {:.2f}]\n"
                "[{:.2f}, {:.2f}, {:.2f}]\n"
                "[{:.2f}, {:.2f}, {:.2f}]\n"
                "Pose:\n"
                "[{:.2f}, {:.2f}, {:.2f}, {:.2f}]\n"
                "[{:.2f}, {:.2f}, {:.2f}, {:.2f}]\n"
                "[{:.2f}, {:.2f}, {:.2f}, {:.2f}]\n"
                "[{:.2f}, {:.2f}, {:.2f}, {:.2f}]",
                position.x(), position.y(), position.z(),
                object_center.x(), object_center.y(), object_center.z(),
                rotation(0, 0), rotation(0, 1), rotation(0, 2),
                rotation(1, 0), rotation(1, 1), rotation(1, 2),
                rotation(2, 0), rotation(2, 1), rotation(2, 2),
                pose(0, 0), pose(0, 1), pose(0, 2), pose(0, 3),
                pose(1, 0), pose(1, 1), pose(1, 2), pose(1, 3),
                pose(2, 0), pose(2, 1), pose(2, 2), pose(2, 3),
                pose(3, 0), pose(3, 1), pose(3, 2), pose(3, 3)
                );
    }
};

#endif // FMT_VIEW_HPP

