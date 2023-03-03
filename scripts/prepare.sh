set -e
device_root=$(lsblk | grep sda1 | rev | cut -d " " -f1 | rev)
if [ $(expr length "${device_root}") == 0  ]; then
    echo "mount device to /media/nvidia/record/"
    sudo mount /dev/sda1 /media/nvidia/record
else
    echo "already device mounted: ${device_root}"
fi
sudo tztek-jetson-tool-cpld-test -d /dev/ttyTHS1 -t 3 -l 232 -b 11520
sudo netplan apply
