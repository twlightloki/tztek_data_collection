#!/bin/bash
mount_source=$(lsblk -l -o NAME | grep -e sda1 -e sdb1 -e sdc1 | head -n 1)
device_root=$(lsblk -l -o NAME,MOUNTPOINT | grep -e sda1 -e sdb1 -e sdc1 | head -n 1| rev | cut -d " " -f1 | rev)
if [[ ${device_root} != "/media/nvidia/record" ]]; then
    if [ $(expr length "${mount_source}") == 0  ]; then
        echo "没有检测到硬盘挂载" 
        exit 1
    else
        if [ $(expr length "${device_root}") != 0  ]; then
            sudo umount /dev/${mount_source}
        fi
    fi
    echo "挂载硬盘 ${mount_source} to /media/nvidia/record/"
    sudo mount /dev/${mount_source} /media/nvidia/record
fi
export DEVICE_ROOT=/media/nvidia/record
