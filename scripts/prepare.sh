set -e
sudo tztek-jetson-tool-cpld-test -d /dev/ttyTHS1 -t 3 -l 232 -b 11520
sudo netplan apply
