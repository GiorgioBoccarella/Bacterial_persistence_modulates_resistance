import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import os

# save plot in the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(script_dir, exist_ok=True)

# Set publication-quality figure parameters
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# The antibiotic concentrations (used for filtering data for the boxplot)
AB_concentrations = [12.5]

# Define colors for nutrient concentrations
COLOR_LOW_NUTRIENT = 'blue'  # For 0.25 nutrient concentration
COLOR_HIGH_NUTRIENT = 'green' # For 0.8 nutrient concentration

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(7, 6)) # Adjusted for a single plot

# --- Third plot: Boxplot (now the only plot) ---
# Load the dataset
try:
    # Try to use __file__ if available, otherwise use current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    complete_data_file = os.path.join(script_dir, "..", "complete_data", "complete_data.xlsx")
    df = pd.read_excel(complete_data_file, sheet_name='S1_figure')

    # Filter the dataset
    subset_df = df[
        df['AB_conc'].isin(AB_concentrations) &
        df['nutrient_conc'].isin([0.25, 0.8]) &
        df['time'].isin([2])
    ]

    # Create a figure-level groupby object
    # Pandas sorts group keys by default, so 0.25 will be first, then 0.8
    grouped = subset_df.groupby('nutrient_conc')
    
    # Define positions and width for the boxes
    positions = [0, 1]  # x positions for the boxes
    width = 0.75
    
    # Create manual boxplots using matplotlib directly
    for i, (group_name, group_data) in enumerate(grouped):
        box_data = group_data['surv_frac']
        
        # Determine color based on group index (0 for 0.25, 1 for 0.8)
        box_color = COLOR_LOW_NUTRIENT if i == 0 else COLOR_HIGH_NUTRIENT
        
        bp = ax.boxplot(
            box_data, 
            positions=[positions[i]], 
            widths=width,
            patch_artist=True,  # Fill boxes with color
            showfliers=False
        )
        
        # Style the box
        for box_item in bp['boxes']:
            box_item.set(facecolor=box_color, edgecolor='black', linewidth=1.5)
        
        # Style the whiskers
        for whisker in bp['whiskers']:
            whisker.set(color=box_color, linewidth=1.5)
        
        # Style the caps
        for cap in bp['caps']:
            cap.set(color=box_color, linewidth=1.5)
        
        # Style the median line
        for median in bp['medians']:
            median.set(color='black', linewidth=2)
    
    # Configure the plot
    ax.set_yscale('log')
    ax.set_xlabel('Nutrient Concentration')
    ax.set_ylabel('Survival Fraction')
    ax.set_title('Experimental Survival Data') # Simplified title
    
    # Set x tick positions and labels
    ax.set_xticks(positions)
    # Using actual newlines for the labels
    xtick_labels_text = ['25% MHB\n(High persistence level)', '80% MHB\n(Low persistence level)']
    ax.set_xticklabels(xtick_labels_text)
    
    # Removed color setting for tick labels as per previous request

except Exception as e:
    # If the CSV file doesn't exist or can't be loaded, add a note in the plot
    ax.text(0.5, 0.5, f"CSV data not available\\n{str(e)}", 
             horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes)
    ax.set_title('Experimental Survival Data')

# Add grid for better readability
ax.grid(True, which='both', linestyle='--', alpha=0.3)

# Ensure proper spacing
plt.tight_layout() # Simplified call

# Save the figure in high resolution with a properly descriptive filename
plt.savefig(os.path.join(script_dir, 'S1_figure.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, 'S1_figure.pdf'), format='pdf', bbox_inches='tight')