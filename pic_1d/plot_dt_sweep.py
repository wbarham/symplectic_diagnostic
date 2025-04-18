#!/usr/bin/env python3
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def parse_loopintegral_sweep(filename):
    """Parse the sweep file for the LoopIntegral sweep."""
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
                    section_content,
                    flags=re.IGNORECASE
                )
                rk2_match = re.search(
                    r'Relative error \(RK2\):\s*([\d\.eE+-]+)', 
                    section_content,
                    flags=re.IGNORECASE
                )

                if symp_match and rk2_match:
                    dt_values.append(dt)
                    symp_errors.append(float(symp_match.group(1)))
                    rk2_errors.append(float(rk2_match.group(1)))
                    
            except (IndexError, ValueError) as e:
                print(f"Warning: Couldn't parse section: {e}")
                continue

    # Convert to numpy arrays and sort
    dt_values = np.array(dt_values)
    symp_errors = np.array(symp_errors)
    rk2_errors = np.array(rk2_errors)
    sort_idx = np.argsort(dt_values)
    
    return dt_values[sort_idx], symp_errors[sort_idx], rk2_errors[sort_idx]

def plot_loopintegral_results(dt_values, symp_errors, rk2_errors, interp_order, filter_param, N_traj, IC):
    """Plot the loop integral sweep results."""
    plt.figure()
    
    plt.loglog(dt_values, symp_errors + 1e-16, 'bo-', 
               label='Strang', markersize=8, linewidth=2)
    plt.loglog(dt_values, rk2_errors + 1e-16, 'rs--',
               label='RK2', markersize=8, linewidth=2)
    
    # Add dt^2 reference line
    y_ref = symp_errors[0] * (dt_values / dt_values[0])**2
    plt.loglog(dt_values, y_ref, 'k--', alpha=0.6, label=r'$\Delta t^2$ reference')
    
    plt.xlabel(r'$\Delta t$', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    
    #title = f'Loop Integral Error\nOrder {interp_order}, Filter {filter_param}, IC={IC}, M={N_traj}'
    #plt.title(title, fontsize=14)
    
    ax = plt.gca()
    ax.set_xticks(dt_values[::2])
    ax.set_xticklabels([f'$2^{{{-int(np.round(np.log2(1/dt)))}}}$' for dt in dt_values[::2]], fontsize=12)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    filename = f'sweep_loopintegral_order{interp_order}_filter{filter_param}_IC{IC}_M{N_traj}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

def find_latest_loopintegral_file(order, filter_param, IC, N_traj):
    """Find the latest results file matching sweep pattern."""
    results_dir = Path('simulation_results')
    # Adjust pattern to expect IC in the filename
    pattern = f'sweep_dt_*_ORDER_{order}_FILTER_{filter_param}_IC_{IC}_M_{N_traj}.txt'
    result_files = sorted(results_dir.glob(pattern), reverse=True)
    
    # Now filter by IC appearing in the file content
    for file in result_files:
        with open(file, 'r') as f:
            content = f.read()
            if IC in content:
                return file
    
    raise FileNotFoundError(f"No sweep results matching IC={IC} found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot loop integral dt sweep results.')
    parser.add_argument('interpolation_order', type=int, help='Interpolation order (e.g., 2)')
    parser.add_argument('filter', type=int, help='Filter parameter (e.g., 0)')
    parser.add_argument('IC', type=str, help='Initial condition name (e.g., landau)')
    parser.add_argument('N_traj', type=int, help='Number of trajectories M (e.g., 4096)')
    args = parser.parse_args()

    try:
        results_file = find_latest_loopintegral_file(args.interpolation_order, args.filter, args.IC, args.N_traj)
        print(f"Processing results from: {results_file}")
        
        dt, symp_errors, rk2_errors = parse_loopintegral_sweep(results_file)

        if len(dt) == 0:
            raise ValueError("No valid data found in the results file.")

        print(f"Found {len(dt)} data points.")
        plot_loopintegral_results(dt, symp_errors, rk2_errors, args.interpolation_order, args.filter, args.N_traj, args.IC)
        
        print("Plot created successfully.")
        
    except Exception as e:
        print(f"Error: {e}")
