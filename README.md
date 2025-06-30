# 🧹 Loop Integral Symplectic Diagnostic

This repository contains a diagnostic tool to verify the symplecticity of time-advance maps for ODEs. Several examples have been included in the subdirectories to demonstrate basic usage. 

## 📦 Requirements

Install the required Python packages:

```bash
pip install numpy numba matplotlib scipy pyfftw
```

## 📄 Relation to Published Work

This repository serves as a companion to the numerical results presented in the corresponding paper, **Insert name here**. The structure and contents of this repository are designed to document and allow for the reproduction of the findings discussed in that publication, and to provide templates for using the diagnostic tool in other contexts. 

**Top directory**:

- `loop_integrals.py` and `loop_integrals_par.py`: Located in the top-level directory are serial and parallel implementations of the symplectic diagnostic tool. 

**Subdirectories**: Each subfolder within this repository corresponds to a set of numerical tests detailed in the paper.

- **Implementation Files**: Python scripts impelementing the numerical method proper to each test. These are named and commented such that their purpose in relation to the article is clear.

- **`.sh` Files**: Shell scripts responsible for executing the parameter sweeps. 
  
- **`plot*.py` Files**: Python scripts which generate plots from the data generated in the shell scripts. These files generate the plots found in the paper, and are included for completeness. 

- **`simulate.py` Files**: Utility scripts to organize the workflow. They parse command line arguements from the shell scripts, call simulation methods from the implementation files, and output the results in a formatted fashion to be easily read by the plot scripts. 
