set -e
device_root=$(lsblk -l -o NAME,MOUNTPOINT | grep -e sda1 -e sdb1 -e sdc1 | head -n 1| rev | cut -d " " -f1 | rev)
mount_source=$(lsblk -l -o NAME | grep -e sda1 -e sdb1 -e sdc1 | head -n 1)
if [ $(expr length "${mount_source}") == 0  ]; then
    echo "no device found" 
    exit
fi
if [ $(expr length "${device_root}") == 0  ]; then
    echo "mount ${mount_source} to /media/nvidia/record/"
    sudo mount /dev/${mount_source} /media/nvidia/record
else
    echo "already device mounted: ${device_root}"
fi
sudo tztek-jetson-tool-cpld-test -d /dev/ttyTHS1 -t 3 -l 232 -b 11520
sudo netplan apply
