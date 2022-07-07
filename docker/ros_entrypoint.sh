#!/bin/bash
 
set -e

# Ros build
source "/opt/ros/melodic/setup.bash"

# Fix error related to GTSAM 
# ref: https://github.com/borglab/gtsam/issues/380
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH


echo "==============LVI-SAM Docker Env Ready================"

cd /home/catkin_ws

exec "$@"
