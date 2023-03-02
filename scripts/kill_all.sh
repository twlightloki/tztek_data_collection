#!/bin/bash
set -e
sudo systemctl stop rslidar_sdk.service
./build/network_pub 5557 "exit"

