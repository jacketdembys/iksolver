#!/bin/bash
# This script creates a config YAML file
# List all the config files
#layers=2
#neurons=1000
#robot_choice="7DoF-7R-Panda"
#scripts=$(ls *.yaml)

# Create the config file
#echo "$scripts";

#for script in $`scripts`; do
#    python ik-solver.py --config-path "$script" &
#done

python ik-solver.py --config-path train_14.yaml &
python ik-solver.py --config-path train_15.yaml &
#python ik-solver.py --config-path train_3.yaml &
#python ik-solver.py --config-path train_4.yaml &
#python ik-solver.py --config-path train_5.yaml &
wait


# Print a success message
echo "successfully running experiments based created yaml files!"