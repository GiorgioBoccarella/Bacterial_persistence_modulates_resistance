import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import Normalize
import os

# --------------------------------
# 1. Prepare the Data
# --------------------------------
sns.set_context("talk", font_scale=1.3)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute path to the complete data file
complete_data_file = os.path.join(script_dir, "..", "..", "complete_data", "complete_data.xlsx")

# Load data from Excel file
mic_data = pd.read_excel(complete_data_file, sheet_name='Fig_6_c_mic')
extinction_data = pd.read_excel(complete_data_file, sheet_name='Fig_6_c_extinction')
extinction_data['fraction_extinct'] = extinction_data['extinct_count'] / extinction_data['total_count']

manual_extinction_text = {
    (12.5, 'Low'):  "2/24",
    (25.0, 'Low'):  "22/24",
    (12.5, 'High'): "3/24",
    (25.0, 'High'): "2/24"
}

pivot_extinct = (
    extinction_data
    .pivot(index='AB_conc', columns='pers_label', values='fraction_extinct')
    [['High','Low']]
    .reindex([25, 12.5])
)

# Create a mapping for severity labels
severity_labels = {25: 'High severity', 12.5: 'Low severity'}

# --------------------------------
# 2. Color Schemes
# --------------------------------
# Use same colormap as in main_plot.py
extinction_cmap = sns.diverging_palette(225, 20, l=65, as_cmap=True).reversed()

# Using the same binning and colors as in main_plot.py
bins = [0, 8, 16, 32, np.inf]
bin_labels = ["0–8", "8–16", "16–32", ">32"]
bin_colors = sns.light_palette("red", n_colors=len(bin_labels), reverse=False)

# Convert MIC strings to numeric for binning
# Note: '32+' in the original data needs special handling
mic_numeric_map = {'2': 2, '4': 4, '8': 8, '16': 16, '32+': 40}  # Map '32+' to a value > 32
mic_data['MIC_numeric'] = mic_data['MIC'].map(mic_numeric_map)

# Map original MIC values to new bin categories
def assign_bin(mic_val):
    if mic_val < 8:
        return 0  # 0-8 bin
    elif mic_val < 16:
        return 1  # 8-16 bin
    elif mic_val <= 32:
        return 2  # 16-32 bin
    else:
        return 3  # >32 bin

mic_data['bin_index'] = mic_data['MIC_numeric'].apply(assign_bin)

# --------------------------------
# 3. Plot (with much tighter spacing!)
# --------------------------------
fig, axes = plt.subplots(
    ncols=2,
    figsize=(10, 6),
    sharey=True,
    gridspec_kw={'wspace': 0.3}   
)

for ax, pers_label in zip(axes, ['High', 'Low']):
    data = pivot_extinct[[pers_label]]
    sns.heatmap(
        data,
        ax=ax,
        cmap=extinction_cmap,
        cbar=(pers_label=='Low'),
        vmin=0, vmax=1,
        linewidths=1, linecolor='black',
        cbar_kws={'label': 'Fraction Extinct'},
        square=True,
        xticklabels=False,
        yticklabels=False
    )
    ax.set_title(f"{pers_label} persistence")

    # Remove x-axis labels completely
    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_yticks(np.arange(data.shape[0]) + 0.5)
    # Change y-axis labels to severity levels instead of AB_conc values
    ax.set_yticklabels([severity_labels[idx] for idx in data.index], rotation=0, fontsize=16)

    if pers_label == 'Low':
        ax.yaxis.set_visible(False)

    for i, ab_conc in enumerate(data.index):
        if data.loc[ab_conc, pers_label] >= 1:
            ax.add_patch(Rectangle((0, i), 1, 1,
                                   facecolor='lightgreen',
                                   edgecolor='black',
                                   linewidth=0.1,
                                   alpha=0.75))

    total_w, total_h = 0.5, 0.3
    for i, ab_conc in enumerate(data.index):
        x0, y0 = 0.5, i + 0.5
        txt = manual_extinction_text[(ab_conc, pers_label)]
        cell = mic_data[(mic_data.AB_conc==ab_conc)&(mic_data.pers_label==pers_label)]
        if data.loc[ab_conc, pers_label] >= 1 or cell.empty:
            ax.text(x0, y0, txt, ha='center', va='center',
                    fontsize=11, fontweight='bold')
        else:
            left = x0 - total_w/2
            bottom = y0 - total_h/2
            cur = left
            
            # Group MIC values by bin and sum frequencies
            bin_freqs = {}
            for bin_idx in range(len(bin_labels)):
                bin_freqs[bin_idx] = cell[cell['bin_index']==bin_idx]['rel_freq'].sum()
            
            # Plot each bin
            for bin_idx, freq in bin_freqs.items():
                if freq > 0:
                    w = freq * total_w
                    ax.add_patch(Rectangle((cur, bottom), w, total_h,
                                          facecolor=bin_colors[bin_idx],
                                          edgecolor='black',
                                          linewidth=0.5))
                    cur += w
                    
            ax.text(x0, bottom-0.15, txt, ha='center', va='center',
                    fontsize=11, fontweight='bold')

# --------------------------------
# 4. Legends + Labels
# --------------------------------
# Add a legend for MIC bins
mic_handles = [Patch(facecolor=bin_colors[i], edgecolor='black', label=bin_labels[i]) 
              for i in range(len(bin_labels))]
#fig.legend(handles=mic_handles, title="Evolved MIC", loc='upper right', bbox_to_anchor=(0.9, 0.9))

# Add legend for extinction indicator
extinction_handle = [Patch(facecolor='lightgreen', edgecolor='black', label='100% Extinct')]
#fig.legend(handles=extinction_handle, loc='lower right', bbox_to_anchor=(0.9, 0.8))

plt.tight_layout(rect=[0,0,0.85,0.95])

# Construct absolute path for the output plot
output_plot_path = os.path.join(script_dir, "Fig_6_c.png")
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')