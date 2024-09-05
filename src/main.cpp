// File: main.cpp

#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "ftv.hpp"

int main(const int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  const auto node = std::make_shared<FTVNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
