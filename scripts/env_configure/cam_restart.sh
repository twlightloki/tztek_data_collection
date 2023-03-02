#!/bin/bash
set -e
config_suffix=$1
sudo cp /home/nvidia/tztek_data_collection/configs/cam_cfg_${config_suffix}.ini /etc/configure-camera/cam_cfg.ini
sudo systemctl restart jetson-cam-cfg.service
