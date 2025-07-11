This repository contains the  analysis scripts used to generate all figures and analyses for the manuscript. * The full data set for figure generation is deposited elsewhere. (Link added upon publication )

## Repository Structure

```
├── Fig_1/                   # Survival analysis and population dynamics
│   ├── Fig_1_a-b.py         # Mathematical survival models
│   └── Fig_1_c-d-e.py       # Simple simulations
├── Fig_2/                   # Probability of emergence analysis
│   └── Fig_2.py             # Contour plot generation
├── Fig_3/                   # MIC evolution simulations
│   ├── Fig_3_a/             # R-based simulation analysis
│   └── Fig_3_b/             # Python visualization
├── Fig_4/                   # Distribution analysis
│   └── Fig_4_a-b.R          # Density plots and bar charts
├── Fig_5/                   # Mutation analysis
│   ├── Fig_5_a-b/           # Mutation counting analysis
│   │   ├── Fig_5_a/         # Simulation data (sim)
│   │   └── Fig_5_b/         # Empirical data (emp)
│   └── Fig_5_c/             # Gene ontology analysis and functional
├── Fig_6/                   # Large-scale evolutionary simulations
│   ├── Fig_6_a-b/           # Heatmap visualizations
│   ├── Fig_6_c/             # MIC and extinction analysis
│   └── Fig_6_d/             # Population size effects
├── S1_figure/               # Supplementary experimental data
├── S2_figure/               # Supplementary frequency analysis
├── S3_figure/               # Supplementary probability analysis
├── scripts_simulations_cluster/ # Large scale cluster optimized simulations
└── complete_data/           # * The full datasheet is deposited elsewhere
```


## Script Types and Languages

### Python Scripts (`.py`)
- **Mathematical modeling**: Survival functions, probability calculations
- **Stochastic simulations**: Tau-leaping population dynamics
- **Data processing**: Mutation analysis, frequency calculations
- **Visualization**: Matplotlib/Seaborn-based plotting
- **Dependencies**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`

### R Scripts (`.R`)
- **Statistical analysis**: Distribution fitting, density plots
- **Advanced visualization**: ggplot2-based publication-quality figures
- **Data manipulation**: dplyr/tidyr workflows
- **Dependencies**: `dplyr`, `tidyr`, `ggplot2`, `readxl`, `cowplot`

## Data Requirements

### For Complete Analysis
Most figures require experimental or simulation data. The expected data format is an Excel file (`complete_data.xlsx`) with specific sheet names corresponding to each figure.


## Quick Start

### Self-Sufficient Figures (Fast Generation)
The first two figures can be generated quickly without external data:

```bash
# Figure 1: Survival analysis and simulations (1-2 minutes)
cd Fig_1
python Fig_1_a-b.py      # Mathematical models
python Fig_1_c-d-e.py    # Population simulations

# Figure 2: Probability of emergence (1-2 minutes)
cd ../Fig_2
python Fig_2.py          # Contour analysis
```

These scripts are completely self-contained and will generate both the analysis and output figures.

## Large Scale Simulations
The folder `scripts_simulations_cluster/` contains scripts optimized for running large-scale simulations on a cluster. They can be adapted for faster use on a laptop by reducing the number of replicates. The only added dependency for these scripts is `mpi4py`.


### Data Placement
1. **Drop data file**: Place `complete_data.xlsx` in the `complete_data/` folder
2. **Automatic detection**: Scripts will automatically locate and read the appropriate data sheets
3. **Sheet mapping**: Each script reads from its corresponding sheet (e.g., `Fig_3_b.py` reads from `Fig_3_b` sheet)

### Data Availability
The complete dataset will be deposited in a public data repository upon publication. The data includes:
- Experimental evolution results
- Large-scale simulation outputs  
- Mutation frequency measurements
- Survival fraction data


## System Requirements

### Python Environment
- Python 3.7+
- Required packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `mpi4py`
- Optional: `goatools` (for gene ontology analysis in Fig_5_c)

### R Environment  
- R 4.0+
- Required packages: `dplyr`, `tidyr`, `ggplot2`, `readxl`, `cowplot`

## Citation

If you use these scripts in your research, please cite the original manuscript:

[Citation information will be added upon publication]


