set -ex
bash ./prepare.sh
device_root=$(lsblk | grep sda1 | rev | cut -d " " -f1 | rev)
source /home/nvidia/ws_rslidar_sdk/devel/setup.bash
nohup roslaunch rslidar_sdk start.launch&
bash run.sh ${device_root}

