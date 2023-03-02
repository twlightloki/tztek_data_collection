#!/bin/bash
set -e
collect_case=$1
echo_cam_config (){
    echo "摄像头初始化可能未完成"
    echo "请执行 bash /home/nvidia/scripts/env_configure/cam_configure.sh [配置方案]"
    echo "配置方案有：cam6 fisheye test"
    exit
}

echo_build (){
    echo "采集工具未找到，请重新构建"
    echo "请执行 bash /home/nvidia/scripts/compile.sh"
    exit
}

if [ ! -f "/etc/configure-camera/current_module" ]; then
    echo_cam_config
fi
module_name=$(</etc/configure-camera/current_module)
echo "目前配置方案: ${module_name}"
echo "请确认 y/n"
read opt
if [ ${opt} != "y" ]; then
    echo_cam_config
fi

sudo systemctl restart rslidar_sdk.service
cd /home/nvidia/tztek_data_collection/scripts
source ./prepare.sh

if [ ! -d "/home/nvidia/tztek_data_collection/scripts/build" ]; then
    echo_build
fi
output_dir=${DEVICE_ROOT}/${collect_case}/
mkdir -p ${output_dir}
bash run.sh ${output_dir} ${module_name}

