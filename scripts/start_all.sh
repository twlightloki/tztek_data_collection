set -ex
cd /home/nvidia/ws_rslidar_sdk/
source devel/setup.bash
nohup roslaunch rslidar_sdk start.launch&
cd /home/nvidia/tztek_data_collection/scripts
bash ./prepare.sh
device_root=$(lsblk | grep sda1 | rev | cut -d " " -f1 | rev)
bash run.sh ${device_root}

