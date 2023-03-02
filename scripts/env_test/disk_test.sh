set -e
disk_num=$(lsusb -t -v | grep "Mass Storage" | wc -l)
if [ $disk_num -ne 1 ]; then
    echo 检测到 USB 储存装置数量：${disk_num} != 1, 不符合预期
    exit 1
fi
bandwidth=$(lsusb -t -v | grep "Mass Storage" | rev | cut -d " " -f1 | rev | cut -d "M" -f1)
if [ $bandwidth -lt 2000 ]; then
    echo 检测到 USB 储存带宽：${bandwidth} \< 2000, 请确认是否支持 USB 3.0
    exit 1
fi
source /home/nvidia/tztek_data_collection/scripts/env_configure/mount_device.sh
time dd if=/dev/zero of=${DEVICE_ROOT}/testfile bs=10M count=200
rm -f ${DEVICE_ROOT}/testfile
