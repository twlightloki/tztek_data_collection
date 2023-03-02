#!/bin/bash
set -e
module=$1
sudo systemctl restart rslidar_sdk.service
cd /home/nvidia/tztek_data_collection/scripts
source ./prepare.sh
bash run.sh ${DEVICE_ROOT} ${module}

