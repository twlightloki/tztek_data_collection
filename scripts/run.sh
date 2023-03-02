#!/bin/bash
set -e
target_dir=$1
module_name=$2
if [ $(expr length "${module_name}") == 0  ]; then
    module_name=test
fi
if [ $(expr length "${test_dir}") == 0  ]; then
    log_dir="./"
else
    log_dir=${target_dir}
fi
sudo LD_LIBRARY_PATH=/opt/ros/noetic/lib/:$LD_LIBRARY_PATH ./build/data_collect /home/nvidia/tztek_data_collection/config_${module_name}.ini "${module_name}" ${target_dir} 2>&1 | tee ${log_dir}/log.$(date +%s)
