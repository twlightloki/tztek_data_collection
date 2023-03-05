target_dir=$1
set -ex
sudo LD_LIBRARY_PATH=/opt/ros/noetic/lib/:$LD_LIBRARY_PATH ./build/data_collect ./config.ini test ${target_dir}
