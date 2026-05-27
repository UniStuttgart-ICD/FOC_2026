FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
ARG UR_ROS2_DRIVER_REF=fb74630777c3e16bdb0ac04fd0ef9186d972fb57
ARG UR_CLIENT_LIBRARY_REF=627cf205cec22a727b73e7c317d93fc088f68ce4

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-humble-ur \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-control-msgs \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-ros2controlcli \
    ros-humble-controller-manager \
    iproute2 \
    netcat-openbsd \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /ur_ws/src \
    && git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git /ur_ws/src/Universal_Robots_ROS2_Driver \
    && git -C /ur_ws/src/Universal_Robots_ROS2_Driver checkout "${UR_ROS2_DRIVER_REF}" \
    && git clone https://github.com/UniversalRobots/Universal_Robots_Client_Library.git /ur_ws/src/Universal_Robots_Client_Library \
    && git -C /ur_ws/src/Universal_Robots_Client_Library checkout "${UR_CLIENT_LIBRARY_REF}" \
    && grep -F "std::chrono::milliseconds timeout(10000)" /ur_ws/src/Universal_Robots_Client_Library/src/ur/ur_driver.cpp \
    && source /opt/ros/humble/setup.bash \
    && cd /ur_ws \
    && rosdep update \
    && rosdep install \
        --rosdistro humble \
        --ignore-src \
        --from-paths src \
        --dependency-types build \
        --dependency-types build_export \
        --dependency-types buildtool \
        --dependency-types buildtool_export \
        --dependency-types exec \
        -y \
    && colcon build \
        --packages-up-to ur_robot_driver \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf /ur_ws/build /ur_ws/log

CMD ["bash"]
