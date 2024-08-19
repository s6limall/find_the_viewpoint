#!/usr/bin/env bash
set -euo pipefail

# Detect if using X11 or Wayland
if [ -n "${DISPLAY:-}" ]; then
  echo "Configuring X11..."
  xhost +local:root
  export DISPLAY=$DISPLAY
  export QT_QPA_PLATFORM=xcb
elif [ -n "${WAYLAND_DISPLAY:-}" ]; then
  echo "Configuring Wayland..."
  export WAYLAND_DISPLAY=$WAYLAND_DISPLAY
  export QT_QPA_PLATFORM=wayland
  export XDG_RUNTIME_DIR="/run/user/${UID:-$(id -u)}"
else
  echo "No display server detected. GUI applications may not work."
fi

# Build and start container
docker compose down --remove-orphans
docker rm -f ros2_${ROS_DISTRO:-humble}_xarm7 || true
docker compose up --build --detach

# Run bash in the container
docker compose run --rm ros2 bash
