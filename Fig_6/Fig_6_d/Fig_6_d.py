import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a publication-ready plotting style.
sns.set_context("talk", font_scale=1.2)
#sns.set_style("whitegrid")

# ----------------------------
# Data Loading and Preparation
# ----------------------------
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute path to the complete data file
complete_data_file = os.path.join(script_dir, "..", "..", "complete_data", "complete_data.xlsx")

df = pd.read_excel(complete_data_file, sheet_name='Fig_6_d')



# ----------------------------
# Define Summarization Functions
# ----------------------------
def summarize_group_survivors(g):
    """Compute average MIC using only surviving replicates."""
    survivors = g[g['extinct'] == 0]
    avg_MIC = survivors['most_common_mic_day_15'].mean() if len(survivors) > 0 else 0
    return pd.Series({'avg_MIC': avg_MIC})

def summarize_group_extinct_as_zero(g):
    """Compute average MIC counting extinct replicates as having MIC = 0."""
    mic_values = np.where(g['extinct'] == 0, g['most_common_mic_day_15'], 0)
    avg_MIC = mic_values.mean()
    return pd.Series({'avg_MIC': avg_MIC})

def summarize_group_extinct_as_wild(g):
    """
    Compute average MIC counting extinct replicates as having wild type MIC = 2.
    
    This avoids inflating the MIC difference by assigning extinct populations
    the wild-type value instead of 0.
    """
    mic_values = np.where(g['extinct'] == 0, g['most_common_mic_day_15'], 2)
    avg_MIC = mic_values.mean()
    return pd.Series({'avg_MIC': avg_MIC})

# ----------------------------
# Process and Normalize Data
# ----------------------------
def process_summary(summary_func):
    results = (
        df.groupby(['K', 'initial_pers_level', 'c', 'growth_duration'])
          .apply(summary_func, include_groups=False)  # Added include_groups=False to fix deprecation warning
          .reset_index()
    )
    # Remove unwanted rows (e.g., K == 10000)
    results = results[results['K'] != 10000]
    
    # Determine lowest and highest initial persister levels.
    all_pers_levels = results['initial_pers_level'].unique()
    lowest_pers = np.min(all_pers_levels)
    highest_pers = np.max(all_pers_levels)
    
    # Create separate subsets for low and high persister levels.
    low_subset = results[results['initial_pers_level'] == lowest_pers][['K', 'c', 'growth_duration', 'avg_MIC']]\
                .rename(columns={'avg_MIC': 'MIC_low'})
    high_subset = results[results['initial_pers_level'] == highest_pers][['K', 'c', 'growth_duration', 'avg_MIC']]\
                 .rename(columns={'avg_MIC': 'MIC_high'})
    
    # Merge to obtain the difference (High - Low)
    merged = pd.merge(low_subset, high_subset, on=['K', 'c', 'growth_duration'], how='inner')
    merged['diff_MIC'] = merged['MIC_high'] - merged['MIC_low']
    
    # Normalize the difference within each population size (K) so diff_norm is between -1 and 1.
    def normalize_within_K(g):
        max_abs_diff = g['diff_MIC'].abs().max()
        g['diff_norm'] = g['diff_MIC'] / max_abs_diff if max_abs_diff > 0 else 0
        return g
    
    # Fix for the deprecation warning - select columns after groupby
    merged = merged.groupby('K', group_keys=False).apply(normalize_within_K)
    
    # Associate each treatment combination with an x-axis index.
    unique_gd = sorted(merged['growth_duration'].unique(), reverse=True)
    unique_c  = sorted(merged['c'].unique(), reverse=False)
    min_len = min(len(unique_gd), len(unique_c))
    paired = list(zip(unique_gd[:min_len], unique_c[:min_len]))
    
    pairs_df = pd.DataFrame({
        'growth_duration': [p[0] for p in paired],
        'c':               [p[1] for p in paired],
        'pair_index':      np.arange(min_len)
    })
    
    filtered_merged = pd.merge(merged, pairs_df, on=['growth_duration', 'c'], how='inner')
    filtered_merged = filtered_merged.sort_values('pair_index')
    return filtered_merged, pairs_df

# Process data using the new summarization approach (assigning extinct replicates MIC = 2).
filtered_wild, pairs_df = process_summary(summarize_group_extinct_as_wild)

# ----------------------------
# Plotting Function with Visual Cues
# ----------------------------
def plot_line(filtered_data, pairs_df, filename, title):
    # Fix: Added height parameter to figsize tuple
    plt.figure(figsize=(8, 6))
    
    # Set y-axis limits.
    plt.ylim(-1.1, 1.1)
    
    # Add background shading to indicate the two regimes.
    plt.axhspan(0, 1.1, facecolor='skyblue', alpha=0.2, zorder=0)
    plt.axhspan(-1.1, 0, facecolor='wheat', alpha=0.2, zorder=0)
    
    # Define a sequential color palette for population sizes.
    unique_k = sorted(filtered_data['K'].unique())
    custom_palette = sns.color_palette("plasma", len(unique_k))
    
    # Create the line plot.
    sns.lineplot(
        data=filtered_data,
        x='pair_index',
        y='diff_norm',
        hue='K',
        palette=custom_palette,
        marker='o',
        markersize=6,
        lw=2,
        alpha=1.0,
        zorder=3  # Ensure plot is above the background shading.
    )
    
    plt.xlabel("Higher treatment severity →\nTime between treatment [h] (↓), Concentration [µg/mL] (↑)", size=14.4)
    plt.ylabel("Difference in MIC\n (High persistence - Low persistence)  ", size=16.4)
    plt.title(title)
    
    # Set custom x-axis tick labels.
    pair_labels = pairs_df.sort_values('pair_index').apply(
        lambda row: f"{row['growth_duration']}, {row['c']}", axis=1
    ).tolist()
    plt.xticks(ticks=np.arange(len(pair_labels)), labels=pair_labels, rotation=45, ha='right', fontsize=12)
    
    # Draw a horizontal line at y=0.
    plt.axhline(0, color='black', linewidth=0.7, zorder=2)
    
    # Add annotation boxes.
    ax = plt.gca()
    ax.text(.05, 0.88, "  ↑ resistance \n   high persistence",
            transform=ax.transAxes, fontsize=10, color='navy',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    ax.text(.72, 0.05, "  ↑ resistance \n  low persistence",
            transform=ax.transAxes, fontsize=10, color='sienna',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    # Adjust the legend: display K values in scientific notation.
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        try:
            k_val = float(label)
            exponent = int(np.log10(k_val))
            new_label = f"$10^{{{exponent}}}$"
        except ValueError:
            new_label = label
        new_labels.append(new_label)
    plt.legend(handles, new_labels, title="Population Size", bbox_to_anchor=(1.05, 1),
               prop={'size': 10}, title_fontsize=10)
    
    plt.tight_layout()
    # Construct absolute path for the output plot
    output_plot_path = os.path.join(script_dir, filename)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

# ----------------------------
# Generate and Save the Plot
# ----------------------------
plot_line(filtered_wild,
          pairs_df,
          "Fig_6_d.png",
          "")