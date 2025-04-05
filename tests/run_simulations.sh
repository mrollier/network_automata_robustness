#!/bin/bash

# filepath: /home/michiel/Documentos/Repos/network_automata_robustness/tests/state_vs-defect-density.py

# Define the Python script path
PYTHON_SCRIPT="/home/michiel/Documentos/Repos/network_automata_robustness/tests/state_vs-defect-density.py"

RESOLUTION=5
L=30
T=100
NUM_GRAPHS=30
NUM_INIT_CONF=30
REWIRING_PROB=0.2
NETWORK_TYPE="random"
OUTPUT_FILE="results_$NETWORK_TYPE.h5"

python3 "$PYTHON_SCRIPT" --network_type $NETWORK_TYPE --resolution $RESOLUTION --L $L --T $T --num_init_conf $NUM_INIT_CONF --num_graphs $NUM_GRAPHS --output_file $OUTPUT_FILE