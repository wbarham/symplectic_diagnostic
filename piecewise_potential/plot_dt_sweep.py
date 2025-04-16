#!/usr/bin/env python3
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def parse_sweep_file(filename):
    """Improved parser for the parameter sweep output file"""
    dt_values = []
    strang_errors = []
    rk2_errors = []
    
    with open(filename, 'r') as f:
        content = f.read()
        
        # Find all dt sections
        sections = re.split(r'=== dt = ([\d\.e+-]+) ===', content)[1:]
        
        # Process in pairs (dt value, content)
        for i in range(0, len(sections), 2):
            try:
                dt = float(sections[i])
                section_content = sections[i+1]
                
                # Search for errors in this section
                strang_match = re.search(
                    r'Relative error \(Symplectic\):\s*([\d\.e+-]+)', 
                    section_content
                )
                rk2_match = re.search(
                    r'Relative error \(RK2\):\s*([\d\.e+-]+)', 
                    section_content
                )
                
                if strang_match and rk2_match:
                    dt_values.append(dt)
                    strang_errors.append(float(strang_match.group(1)))
                    rk2_errors.append(float(rk2_match.group(1)))
                    
            except (IndexError, ValueError) as e:
                print(f"Warning: Couldn't parse section: {e}")
                continue
    
    # Convert to numpy arrays and sort by dt
    dt_values = np.array(dt_values)
    strang_errors = np.array(strang_errors)
    rk2_errors = np.array(rk2_errors)
    
    # Sort by dt (in case they weren't in order)
    sort_idx = np.argsort(dt_values)
    
    return dt_values[sort_idx], strang_errors[sort_idx], rk2_errors[sort_idx]

def plot_results(dt_values, strang_errors, rk2_errors, interp_order):
    """Create clean log-log comparison plot with only Δt² reference"""
    plt.figure()
    
    # Plot data with small offset to avoid log(0)
    plt.loglog(dt_values, strang_errors + 1e-16, 'bo-', 
              label='Strang Splitting', markersize=8, linewidth=2)
    plt.loglog(dt_values, rk2_errors + 1e-16, 'rs--', 
              label='RK2', markersize=8, linewidth=2)
    
    # Add Δt^4 reference line
    y_ref = 1e-8 * (dt_values/dt_values[0])**4
    plt.loglog(dt_values, y_ref, 'k:', alpha=0.5, 
              label=r'$\Delta t^4$ reference')
    
    # Format plot
    plt.xlabel(r'$\Delta t$', fontsize=14)
    plt.ylabel('Relative error in loop integral', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)

    #order_print = f"Interpolation Order {interp_order}" if interp_order else "Non-interpolated"
    #plt.title(rf'Relative error vs $\Delta t$ ({order_print})', fontsize=14)
    
    # Clean x-axis with only powers of 2 as ticks
    ax = plt.gca()
    ax.set_xticks(dt_values[::2])
    ax.set_xticklabels([f'$2^{{{-int(np.round(np.log2(1/dt)))}}}$' 
                      for dt in dt_values[::2]], fontsize=12)
    
    # Remove minor ticks between the powers of 2
    ax.xaxis.set_minor_locator(plt.NullLocator())
    
    plt.legend(fontsize=12, framealpha=0.9)
    plt.tight_layout()
    filename = f'sweep_dt_order{interp_order}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

def find_latest_results(interpolation_order):
    """Find the most recent results file for a given interpolation order"""
    results_dir = Path('simulation_results')
    pattern = f'sweep_results_dt_*_ORDER_{interpolation_order}.txt'
    result_files = sorted(results_dir.glob(pattern), reverse=True)
    
    if not result_files:
        raise FileNotFoundError(f"No results files found for interpolation order {interpolation_order}.")
    
    # Verify file has content
    latest_file = result_files[0]
    if latest_file.stat().st_size == 0:
        raise ValueError(f"Empty results file: {latest_file}")
    
    return latest_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot dt sweep results for a given interpolation order.')
    parser.add_argument('interpolation_order', type=int,
                        help='Interpolation order to filter result files by (e.g., 2)')
    args = parser.parse_args()

    try:
        results_file = find_latest_results(args.interpolation_order)
        print(f"Processing results from: {results_file}")
        
        dt, strang_err, rk2_err = parse_sweep_file(results_file)
        
        if len(dt) == 0:
            raise ValueError("No valid data found in results file. Check the format.")
        
        print(f"Found {len(dt)} data points:")
        print("dt values:", dt)
        print("Strang errors:", strang_err)
        print("RK2 errors:", rk2_err)
        
        plot_results(dt, strang_err, rk2_err, args.interpolation_order)
        print("Successfully created convergence_plot.png")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify your sweep output file has the expected format")
        print("2. Check that lines contain 'Relative error (Symplectic)' and 'Relative error (RK2)'")
        print("3. Ensure the file isn't empty")
        print("\nExample expected format:")
        print("=== dt = 0.5 ===")
        print("Relative error (Symplectic): 1.23e-4")
        print("Relative error (RK2): 5.67e-5")
