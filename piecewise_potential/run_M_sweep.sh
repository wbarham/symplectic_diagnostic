#!/bin/bash

# Check for required arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <interpolation_order> <dt>"
    echo "Example: $0 2 0.125"
    exit 1
fi

INTERP_ORDER=$1
DT=$2

# Create output directory if it doesn't exist
mkdir -p simulation_results

# Generate timestamp for unique filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="simulation_results/sweep_results_M_${TIMESTAMP}.txt"

# Header for the output file
echo "=== Parameter Sweep Results ===" > $OUTPUT_FILE
echo "Start time: $(date)" >> $OUTPUT_FILE
echo "Sweeping M from 2^4 to 2^16" >> $OUTPUT_FILE
echo "==============================" >> $OUTPUT_FILE

# Loop over dt values
for i in {4..16}; do
    M=$(awk "BEGIN {print 2^$i}")
    echo "Running simulation with M = $M (2^$i)"
    
    # Run simulation and append to output file
    echo -e "\n=== M = $M ===" >> $OUTPUT_FILE
    python simulate.py --M $M --dt $DT --interp_order $INTERP_ORDER >> $OUTPUT_FILE
    
    # Progress indicator
    echo "Completed M = 2^$i"
done

# Final status
echo -e "\n=== Sweep Complete ===" >> $OUTPUT_FILE
echo "End time: $(date)" >> $OUTPUT_FILE
echo "Results saved to: $OUTPUT_FILE"

echo "Parameter sweep complete. Results saved to $OUTPUT_FILE"