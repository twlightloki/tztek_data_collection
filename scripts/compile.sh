set -e
cd /home/nvidia/tztek_data_collection/scripts
rm -rf build
mkdir -p build
cd build
cmake ../../
make -j

