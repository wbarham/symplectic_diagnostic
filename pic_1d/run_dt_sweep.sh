#!/bin/bash

# Check for required arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <interpolation_order> <filter> <N_traj>"
    echo "Example: $0 2 0 1024"
    exit 1
fi

INTERP_ORDER=$1
FILTER=$2
N_TRAJ=$3

# Create output directory if it doesn't exist
mkdir -p simulation_results

# Generate timestamp for unique filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="simulation_results/sweep_dt_${TIMESTAMP}_ORDER_${INTERP_ORDER}_FILTER_${FILTER}_M_${N_TRAJ}.txt"

# Header for the output file
echo "=== Parameter Sweep Results ===" > "$OUTPUT_FILE"
echo "Start time: $(date)" >> "$OUTPUT_FILE"
echo "Sweeping dt from 2^-1 to 2^-10" >> "$OUTPUT_FILE"
echo "==============================" >> "$OUTPUT_FILE"

# Loop over dt values
for i in {1..10}; do
    dt=$(awk "BEGIN {print 2^-$i}")
    echo "Running simulation with dt = $dt (2^-$i)"
    
    # Run simulation and append to output file
    echo -e "\n=== dt = $dt ===" >> "$OUTPUT_FILE"
    python simulate.py --dt "$dt" --order_space "$INTERP_ORDER" \
        --filter "$FILTER" --Ntheta "$N_TRAJ" >> "$OUTPUT_FILE"
    
    # Progress indicator
    echo "Completed dt = 2^-$i"
done

# Final status
echo -e "\n=== Sweep Complete ===" >> "$OUTPUT_FILE"
echo "End time: $(date)" >> "$OUTPUT_FILE"
echo "Results saved to: $OUTPUT_FILE"

echo "Parameter sweep complete. Results saved to $OUTPUT_FILE"
