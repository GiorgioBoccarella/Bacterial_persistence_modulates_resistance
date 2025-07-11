import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import Normalize
import os



def load_and_prepare_data():
    """Load and prepare the simulation data."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    complete_data_file = os.path.join(script_dir, "..", "..", "complete_data", "complete_data.xlsx")
    
    # Load the data
    df = pd.read_excel(complete_data_file, sheet_name='Fig_6_a-b')
    
    # Get unique persister levels and sort them
    plevels = sorted(df['initial_pers_level'].unique())
    first_plevel = plevels[0]
    
    # Create c_to_duration mapping (you may need to adjust this based on your data structure)
    c_to_duration = {}
    for plevel in plevels:
        c_to_duration[plevel] = {}
        subset = df[df['initial_pers_level'] == plevel]
        for _, row in subset.iterrows():
            if 'treatment_duration' in df.columns:
                c_to_duration[plevel][row['c']] = row['treatment_duration']
            else:
                # If treatment_duration column doesn't exist, use a default mapping
                c_to_duration[plevel][row['c']] = row['c'] * 2  # Adjust as needed
    
    return df, plevels, first_plevel, c_to_duration

# -----------------------------
# 1. Read in & Prepare the Data
# -----------------------------

sns.set_context("talk", font_scale=1.3)  # style for publication

# Load and prepare data using the function
df, plevels, first_plevel, c_to_duration = load_and_prepare_data()

final_mic_col = "most_common_mic_day_12"

# -----------------------------
# 2. Summarize by Group
# -----------------------------

def summarize_group(g):
    """
    Summarize each (initial_pers_level, c, growth_duration) group.
    """
    N = len(g)
    E = g['extinct'].sum()
    c_val = g['c'].iloc[0]
    # survivors
    survivors = g[g['extinct'] == 0]
    M = (survivors[final_mic_col] > c_val).sum()
    avg_MIC = survivors[final_mic_col].mean() if len(survivors) > 0 else 0
    return pd.Series({
        'N': N,
        'E': E,
        'M': M,
        'avg_MIC_survivors': avg_MIC
    })

# -- IMPORTANT: use `group_keys=False` to suppress the deprecation warning --
results = df.groupby(['initial_pers_level', 'c', 'growth_duration'], group_keys=False)\
            .apply(summarize_group)\
            .reset_index()

results['S'] = results['N'] - results['E']
results['fraction_extinct'] = results['E'] / results['N']
results['fraction_mic_above_c'] = np.where(results['S'] > 0, results['M'] / results['S'], 0)
results['mic_ratio'] = np.where(results['S'] > 0, results['avg_MIC_survivors'] / results['c'], 0)


# filter growth_duration different from 2 and 6
results = results[results['growth_duration'].isin([1, 2, 4, 6, 8, 16, 48, 72 ])]

# filter antibiotic concentration different from 0, 4, 8, 12, 16, 20, 24, 28, 32
results = results[results['c'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 50, 100])]

# -----------------------------
# 3. Set Up Color Schemes
# -----------------------------

# Extinction fraction colormap (diverging); reversed so that 0=dark, 1=light
extinction_cmap = sns.diverging_palette(225, 20, l=65, as_cmap=True).reversed()

# MIC bins (4 bins)
bins = [0, 8, 16, 32, np.inf]
bin_labels = ["0–8", "8–16", "16–32", ">32"]
bin_colors = sns.light_palette("red", n_colors=len(bin_labels), reverse=False)


# -----------------------------
# 4. Plotting
# -----------------------------

# MODIFICATION: Reverse the order of plevels to switch the plot order
plevels = plevels[::-1]  # This reverses the array

fig, axes = plt.subplots(
    1, len(plevels),
    figsize=(6 * len(plevels), 5.2),
    sharey=True
)

# If there's only one persister level, make axes a list of length 1
if len(plevels) == 1:
    axes = [axes]

for i, (ax, plevel) in enumerate(zip(axes, plevels)):
    # Subset for this persister level
    subset = results[results['initial_pers_level'] == plevel]
    
    # Pivot so that c is the row, growth_duration is the column
    pivot_extinct = subset.pivot(index='c', columns='growth_duration', values='fraction_extinct')
    
    # Sort descending (largest c at top, largest growth_duration on left)
    pivot_extinct = pivot_extinct.sort_index(ascending=False)
    pivot_extinct = pivot_extinct[pivot_extinct.columns.sort_values(ascending=False)]
    
    # Create the heatmap
    hm = sns.heatmap(
        pivot_extinct,
        ax=ax,
        cmap=extinction_cmap,
        annot=False,
        cbar=False,  # We'll add one colorbar on the right for the entire figure
        vmin=0,
        vmax=1
    )
    
    # Overlay green squares for fraction_extinct == 1
    for row_i, c_val in enumerate(pivot_extinct.index):
        for col_i, gd_val in enumerate(pivot_extinct.columns):
            if pivot_extinct.loc[c_val, gd_val] == 1:
                # draw a green rectangle
                rect = Rectangle(
                    (col_i, row_i), 1, 1,
                    fill=True,
                    facecolor='lightgreen',
                    edgecolor='black',
                    linewidth=0.1,
                    alpha=0.92
                )
                ax.add_patch(rect)
    
    # We will manually set the x- and y-tick locations to match the shape:
    x_values = pivot_extinct.columns.values
    y_values = pivot_extinct.index.values
    
    ax.set_xticks(np.arange(len(x_values)) + 0.5)
    ax.set_yticks(np.arange(len(y_values)) + 0.5)
    
    ax.set_xticklabels(x_values, rotation=90)
    
    # Create custom y-tick labels with both concentration and treatment_duration
    custom_y_labels = []
    for c_val in y_values:
        duration = c_to_duration.get(plevel, {}).get(c_val)
        if duration is not None:
            custom_y_labels.append(f"{c_val}, {duration}")
        else:
            custom_y_labels.append(f"{c_val}")
            
    # Add explicit rotation=0 to keep y-axis labels horizontal
    ax.set_yticklabels(custom_y_labels, rotation=0)
    
    # For histogram blocks in each cell where S>0
    total_width = 0.5   # The total width of the stacked hist bars
    total_height = 0.3  # The total height of the stacked hist bars
    
    # For each row in 'subset', we get the fraction of survivors in each MIC bin
    for row in subset.itertuples(index=False):
        # row has fields: initial_pers_level, c, growth_duration, N, E, M, ...
        c_val = row.c
        gd_val = row.growth_duration
        S = row.S
        
        if (c_val in y_values) and (gd_val in x_values) and S > 0:
            # Convert from c_val, gd_val to the cell center
            col_idx = np.where(x_values == gd_val)[0][0]
            row_idx = np.where(y_values == c_val)[0][0]
            x_center = col_idx + 0.5
            y_center = row_idx + 0.5
            
            # Get the *actual survivors* for that group from df
            condition = (
                (df['initial_pers_level'] == plevel) &
                (df['c'] == c_val) &
                (df['growth_duration'] == gd_val) &
                (df['extinct'] == 0)
            )
            survivors = df.loc[condition, final_mic_col]
            
            # Bin the survivors by their final MIC
            hist_counts, _ = np.histogram(survivors, bins=bins)
            total_counts = hist_counts.sum()
            fractions = hist_counts / total_counts if total_counts > 0 else np.zeros_like(hist_counts)
            
            # Compute the left/bottom of this stacked rectangle
            left = x_center - total_width / 2
            bottom = y_center - total_height / 2
            
            # Draw each bin as a horizontal segment of length (frac * total_width)
            current_x = left
            for bin_idx, frac in enumerate(fractions):
                if frac > 0:
                    block_width = frac * total_width
                    rect = Rectangle(
                        (current_x, bottom),
                        block_width,
                        total_height,
                        facecolor=bin_colors[bin_idx],
                        edgecolor='black',
                        linewidth=0.5
                    )
                    ax.add_patch(rect)
                    current_x += block_width
    
    # Axis labels
    if i == 0:
        ax.set_ylabel("Higher treatment severity →\nConcentration [µg/mL], Duration [h]", fontsize=14)
        # Add a note explaining the tick format
    else:
        ax.set_ylabel("")
    ax.set_xlabel("Time between treatments [h]", fontsize=13)
    if plevel == first_plevel:
        ax.set_title(f"Low persister level", fontsize=14)
    else:
        ax.set_title(f"High persister level", fontsize=14)

    # Increase tick label size
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

# Layout adjustments
plt.tight_layout()
plt.subplots_adjust(left=0.12, right=0.88)  # Increased left margin for longer y-tick labels

# Add one colorbar for the extinction fraction with hard-coded 0.99 value
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap=extinction_cmap, norm=Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Fraction Extinct", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Hard-code 0.99 at the end of the colorbar (replacing the default 1.0)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 0.99])
cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '0.99'])

# Create a legend for MIC bins next to the plots with smaller title
mic_handles = [
    Patch(facecolor=bin_colors[i], edgecolor='black', label=bin_labels[i])
    for i in range(len(bin_labels))
]
fig.legend(
    handles=mic_handles,
    loc='center left',
    bbox_to_anchor=(0.99, 0.8),  # Positioned above
    title="Evolved MIC",
    frameon=True,
    fontsize=12,
    title_fontsize=10  # Smaller title for MIC bins
)

# Create a separate legend for the extinction indicator
extinction_handle = [Patch(facecolor='lightgreen', edgecolor='black', label='100% Extinct')]
fig.legend(
    handles=extinction_handle,
    loc='center left',
    bbox_to_anchor=(0.99, 0.6),  # Positioned below the MIC bins legend
    title="",
    frameon=False,
    fontsize=12
)

# Construct the absolute path for the output plot
script_dir = os.path.dirname(os.path.abspath(__file__)) # Define script_dir here
output_plot_path = os.path.join(script_dir, "Fig_6_a-b.png")
plt.savefig(output_plot_path, dpi=400, bbox_inches='tight')