#!/bin/bash
# This script creates a config YAML file
# List all the config files
layers=5
neurons=1000
robot_choice="7DoF-7R-Panda"
scripts=$(ls configs/"$robot_choice"/config_layers_"$layers"_neurons_"$neurons"/*.yaml)

# Create the config file
echo "$scripts";

for script in $scripts; do
    python ik-solver.py --config-path "$script" &
done


# Print a success message
echo "successfully running experiments based created yaml files!"