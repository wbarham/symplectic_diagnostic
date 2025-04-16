#!/usr/bin/env python3
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def parse_sweep_file(filename):
    """Parse output file where M is varied"""
    M_values = []
    strang_errors = []
    rk2_errors = []
    
    with open(filename, 'r') as f:
        content = f.read()
        sections = re.split(r'=== M = (\d+) ===', content)[1:]
        
        for i in range(0, len(sections), 2):
            try:
                M = int(sections[i])
                section_content = sections[i+1]
                
                strang_match = re.search(
                    r'Relative error \(Symplectic\):\s*([\d\.e+-]+)', 
                    section_content
                )
                rk2_match = re.search(
                    r'Relative error \(RK2\):\s*([\d\.e+-]+)', 
                    section_content
                )
                
                if strang_match and rk2_match:
                    M_values.append(M)
                    strang_errors.append(float(strang_match.group(1)))
                    rk2_errors.append(float(rk2_match.group(1)))
                    
            except (IndexError, ValueError) as e:
                print(f"Warning: Couldn't parse section: {e}")
                continue
    
    # Convert to numpy arrays and sort
    M_values = np.array(M_values)
    strang_errors = np.array(strang_errors)
    rk2_errors = np.array(rk2_errors)
    
    sort_idx = np.argsort(M_values)
    return M_values[sort_idx], strang_errors[sort_idx], rk2_errors[sort_idx]

def plot_results(M_vals, strang_errors, rk2_errors, interp_order):
    """Plot relative errors vs M in log-log"""
    plt.figure()
    
    plt.loglog(M_vals, strang_errors + 1e-16, 'bo-', 
               label='Strang Splitting', markersize=8, linewidth=2)
    plt.loglog(M_vals, rk2_errors + 1e-16, 'rs--', 
               label='RK2', markersize=8, linewidth=2)
    
    # Add M^4 reference line
    if interp_order == 1:
        y_ref = 1e-3 * (M_vals/M_vals[0])**(-1)
        label = r'$N_s^{-1}$ reference'
    elif interp_order == 2:
        y_ref = 1e-3 * (M_vals/M_vals[0])**(-2)
        label = r'$N_s^{-2}$ reference'
    elif interp_order == 3:
        y_ref = 1e-3 * (M_vals/M_vals[0])**(-3)
        label = r'$N_s^{-3}$ reference'
    if interp_order != 0:
        plt.loglog(M_vals, y_ref, 'k:', alpha=0.5, 
                  label=label)

    plt.xlabel(r'$N_s$ (number of trajectories)', fontsize=14)
    plt.ylabel('Relative error in loop integral', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    
    ax = plt.gca()
    ax.set_xticks(M_vals[::2])
    ax.set_xticklabels([f'$2^{{{int(np.log2(m))}}}$' for m in M_vals[::2]], fontsize=12)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    plt.legend(fontsize=12, framealpha=0.9)
    order_print = f"Interpolation Order {interp_order}" if interp_order else "Non-interpolated"
    #plt.title(f'{order_print}', fontsize=14)
    plt.tight_layout()
    
    filename = f'sweep_M_order{interp_order}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {filename}")

def find_latest_results(interp_order):
    """Find the most recent M-sweep results file for a given order"""
    results_dir = Path('simulation_results')
    pattern = f'sweep_results_M_*_ORDER_{interp_order}.txt'
    result_files = sorted(results_dir.glob(pattern), reverse=True)
    
    if not result_files:
        raise FileNotFoundError(f"No results files found for interpolation order {interp_order}")
    
    latest_file = result_files[0]
    if latest_file.stat().st_size == 0:
        raise ValueError(f"Empty results file: {latest_file}")
    
    return latest_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot relative error vs M')
    parser.add_argument('interpolation_order', type=int,
                        help='Interpolation order used in the sweep')
    args = parser.parse_args()
    
    try:
        results_file = find_latest_results(args.interpolation_order)
        print(f"Processing results from: {results_file}")
        
        M_vals, strang_err, rk2_err = parse_sweep_file(results_file)
        
        if len(M_vals) == 0:
            raise ValueError("No valid data found in results file. Check the format.")
        
        print(f"Found {len(M_vals)} data points:")
        print("M values:", M_vals)
        print("Strang errors:", strang_err)
        print("RK2 errors:", rk2_err)
        
        plot_results(M_vals, strang_err, rk2_err, args.interpolation_order)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify your sweep output file has the expected format")
        print("2. Check that lines contain 'Relative error (Symplectic)' and 'Relative error (RK2)'")
        print("3. Ensure the file isn't empty")
        print("\nExample expected format:")
        print("=== M = 4096 ===")
        print("Relative error (Symplectic): 1.23e-4")
        print("Relative error (RK2): 5.67e-5")
