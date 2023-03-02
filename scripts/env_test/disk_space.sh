#!/bin/bash
set -e
source /home/nvidia/tztek_data_collection/scripts/env_configure/mount_device.sh
free_space=$(df -lh | grep ${DEVICE_ROOT} | awk -F " " '{print $4}')
echo 挂载硬盘剩余空间 ${free_space}
