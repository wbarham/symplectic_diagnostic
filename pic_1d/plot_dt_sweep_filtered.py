#!/usr/bin/env python3
import os
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_loopintegral_sweep(filename):
    """Parse a sweep file and extract dt values, symplectic and RK2 errors."""
    dt_values = []
    symp_errors = []
    rk2_errors = []
    
    with open(filename, 'r') as f:
        content = f.read()
        sections = re.split(r'=== dt = ([\d\.e+-]+) ===', content)[1:]

        for i in range(0, len(sections), 2):
            try:
                dt = float(sections[i])
                section_content = sections[i+1]
                
                symp_match = re.search(
                    r'Relative error \(Symplectic\):\s*([\d\.eE+-]+)', 
                    section_content, flags=re.IGNORECASE
                )
                rk2_match = re.search(
                    r'Relative error \(RK2\):\s*([\d\.eE+-]+)', 
                    section_content, flags=re.IGNORECASE
                )
                
                if symp_match and rk2_match:
                    dt_values.append(dt)
                    symp_errors.append(float(symp_match.group(1)))
                    rk2_errors.append(float(rk2_match.group(1)))
                    
            except (IndexError, ValueError) as e:
                print(f"Warning: Couldn't parse section: {e}")
                continue

    # Sort
    dt_values = np.array(dt_values)
    symp_errors = np.array(symp_errors)
    rk2_errors = np.array(rk2_errors)
    sort_idx = np.argsort(dt_values)

    return dt_values[sort_idx], symp_errors[sort_idx], rk2_errors[sort_idx]

def find_sweep_files(order, IC, M):
    """Find all sweep result files for given settings."""
    results_dir = Path('simulation_results')
    pattern = f'sweep_dt_*_ORDER_{order}_FILTER_*_IC_{IC}_M_{M}.txt'
    files = list(results_dir.glob(pattern))
    return sorted(files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loop integral errors across filters.")
    parser.add_argument("--order_space", type=int, required=True, choices=[1,2,3], help="Spatial interpolation order.")
    parser.add_argument("--IC", type=str, required=True, help="Initial condition name, e.g., two_stream.")
    parser.add_argument("--N_traj", type=int, required=True, help="Number of trajectories M.")
    args = parser.parse_args()

    # Output directory for plots
    plot_dir = f'plots_order{args.order_space}'
    os.makedirs(plot_dir, exist_ok=True)

    # Find all matching result files
    result_files = find_sweep_files(args.order_space, args.IC, args.N_traj)
    
    if not result_files:
        print(f"No results found for ORDER={args.order_space}, IC={args.IC}, M={args.N_traj}")
        exit(1)

    # Organize data by filter parameter
    data = {}
    for file in result_files:
        # Parse filter value from filename
        match = re.search(r'FILTER_(\d+)', str(file))
        if not match:
            print(f"Warning: Couldn't parse filter value from {file}")
            continue
        filter_val = int(match.group(1))
        
        dt, symp_err, rk2_err = parse_loopintegral_sweep(file)
        data[filter_val] = (dt, symp_err, rk2_err)

    # Plot
    plt.figure()
    plt.rcParams.update({'font.size': 12, 'lines.linewidth': 1.5})

    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'd', 'v', 'P']

    for idx, (filter_val, (dt, symp_err, rk2_err)) in enumerate(sorted(data.items())):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        plt.plot(dt, symp_err + 1e-16, 
                 marker + '-', 
                 color=color, 
                 label=f"Strang, s={filter_val}")

        plt.plot(dt, rk2_err + 1e-16, 
                 marker + '--', 
                 color=color, 
                 label=f"RK2, s={filter_val}")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Time Step $\Delta t$')
    plt.ylabel('Relative Error in Loop Integral')
    #plt.title(f'Loop Integral Errors (IC={args.IC}, Order={args.order_space}, M={args.N_traj})')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    # Clean up tick labels
    ax = plt.gca()

    # Collect all unique dt values across filters
    all_dt_values = sorted({d for dt_array, _, _ in data.values() for d in dt_array})

    # Set major ticks (maybe every other one if you have many points)
    ax.set_xticks(all_dt_values[::2])
    ax.set_xticklabels([f'$2^{{{-int(np.round(np.log2(1/d)))} }}$' for d in all_dt_values[::2]], fontsize=12)

    # Remove minor ticks
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

    plot_filename = os.path.join(plot_dir, f"{args.IC}_dt_sweep_order{args.order_space}_all_filters.png")
    plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
    plt.show()

    print(f"Plot saved to {plot_filename}")
