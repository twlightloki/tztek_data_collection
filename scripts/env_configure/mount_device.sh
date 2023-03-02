#!/bin/bash
mount_source=$(lsblk -l -o NAME | grep -e sda1 -e sdb1 -e sdc1 | head -n 1)
if [ $(expr length "${mount_source}") == 0  ]; then
    echo "没有检测到硬盘挂载" 
    exit 1
else
    sudo umount /dev/${mount_source}
fi
echo "挂载硬盘 ${mount_source} to /media/nvidia/record/"
sudo mount /dev/${mount_source} /media/nvidia/record
export DEVICE_ROOT=/media/nvidia/record/
