#!/bin/bash

# Grant permission to all ttyACM ports
echo "Setting permissions for /dev/ttyACM* ports..."
for port in /dev/ttyACM*; do
    if [ -e "$port" ]; then
        sudo chmod 777 "$port"
        echo "  $port - OK"
    fi
done

python scripts/find_joint_limits.py --config robot_configs/robot/so101_robot2.yaml