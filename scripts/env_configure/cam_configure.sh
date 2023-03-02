#!/bin/bash
set -e
module_name=$1
sudo echo ${module_name} > /etc/configure-camera/current_module
sudo cp /home/nvidia/tztek_data_collection/configs/cam_cfg_${module_name}.ini /etc/configure-camera/cam_cfg.ini
sudo systemctl restart jetson-cam-cfg.service
