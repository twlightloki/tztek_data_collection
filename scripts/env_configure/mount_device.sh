#!/bin/bash
device_root=$(lsblk -l -o NAME,MOUNTPOINT | grep -e sda1 -e sdb1 -e sdc1 | head -n 1| rev | cut -d " " -f1 | rev)
mount_source=$(lsblk -l -o NAME | grep -e sda1 -e sdb1 -e sdc1 | head -n 1)
if [ $(expr length "${mount_source}") == 0  ]; then
    echo "no device found" 
    exit 1
fi
if [ $(expr length "${device_root}") == 0  ]; then
    echo "mount ${mount_source} to /media/nvidia/record/"
    sudo mount /dev/${mount_source} /media/nvidia/record
fi
export DEVICE_ROOT=${device_root}
