FROM cxy201/noetic-vizor

ARG ROBOTIQ_2F85_COMMIT=6247f868520ae0452628da6faf059342298e7db6

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:1
ENV NOVNC_PORT=6080
ENV VNC_PORT=5901

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    fluxbox \
    x11vnc \
    xvfb \
    novnc \
    websockify \
    xterm \
    dbus-x11 \
    python3-arrow \
    python3-numpy \
    python3-pymodbus \
    python3-serial \
    ros-noetic-control-msgs \
    ros-noetic-moveit-task-constructor-capabilities \
    ros-noetic-moveit-task-constructor-core \
    ros-noetic-moveit-task-constructor-msgs \
    ros-noetic-moveit-task-constructor-visualization \
    ros-noetic-rviz \
    ros-noetic-moveit-ros-visualization \
    ros-noetic-py-binding-tools \
    ros-noetic-rviz-marker-tools \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/KevinGalassi/Robotiq-2f-85.git /tmp/robotiq-2f-85 \
    && cd /tmp/robotiq-2f-85 \
    && git fetch --depth 1 origin "${ROBOTIQ_2F85_COMMIT}" \
    && git checkout "${ROBOTIQ_2F85_COMMIT}" \
    && cp -a robotiq_2f_85_gripper_visualization robotiq_2f_gripper_control robotiq_2f_gripper_msgs robotiq_modbus_rtu /root/catkin_ws/src/ \
    && rm -rf /tmp/robotiq-2f-85

COPY start-vizor-desktop.sh /usr/local/bin/start-vizor-desktop.sh
COPY vizor_current_pose_service.py /usr/local/bin/vizor_current_pose_service.py
COPY vizor_mtc_pick_server.py /usr/local/bin/vizor_mtc_pick_server.py
COPY patch-vizor-robot.py /usr/local/bin/patch-vizor-robot.py
COPY rviz_ur10.launch /root/catkin_ws/src/vizor_package/launch/rviz_ur10.launch
COPY ur10_robot.rviz /root/catkin_ws/src/vizor_package/launch/ur10_robot.rviz
RUN python3 /usr/local/bin/patch-vizor-robot.py
RUN bash -lc "source /opt/ros/noetic/setup.bash && cd /root/catkin_ws && catkin_make"
RUN chmod +x /usr/local/bin/start-vizor-desktop.sh /usr/local/bin/vizor_current_pose_service.py /usr/local/bin/vizor_mtc_pick_server.py

EXPOSE 6080 5901
ENTRYPOINT ["/usr/local/bin/start-vizor-desktop.sh"]
