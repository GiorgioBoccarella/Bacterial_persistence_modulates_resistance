#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def main():
    # -------------------------------------------------------------------------
    # 1) SETUP & READ PROCESSED CSV
    # -------------------------------------------------------------------------
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    complete_data_file = os.path.join(script_dir, "..", "..", "..", "complete_data", "complete_data.xlsx")
    try:
        data = pd.read_excel(complete_data_file, sheet_name='Fig_5_a')
    except FileNotFoundError:
        print(f"Error: Complete data file not found at {complete_data_file}.")
        return

    # Ensure columns are numeric where needed (already done in preproc, but good for safety)
    if 'frequency' in data.columns:
        data['frequency'] = pd.to_numeric(data['frequency'], errors='coerce')
    if 'mutation_count' in data.columns:
        data['mutation_count'] = pd.to_numeric(data['mutation_count'], errors='coerce')

    # Print unique kappaS and c for reference (should reflect pre-filtered values)
    print("Unique kappaS values in loaded data (expected [2.]): ", data['kappaS'].unique())
    print("Unique c values in loaded data (expected [12.5]): ", data['c'].unique())

    # -------------------------------------------------------------------------
    # 2) FILTERING FOR SPECIFIC DAY
    # -------------------------------------------------------------------------
    day = 13
    data = data[data['day'] == day]
    if data.empty:
        print(f"No data available for day={day} after loading processed data. Exiting.")
        return

    # Define the frequency threshold (e.g., freq_thresh=0.95 for 95%)
    freq_thresh = 0.8

    # -------------------------------------------------------------------------
    # 3) BUILD DICTIONARIES FOR PLOTTING
    #    - For each persistence level, gather total number of mutations
    #      that reach freq >= freq_thresh in that simulation
    # -------------------------------------------------------------------------
    initial_pers_levels = data['pers_level'].unique()
    initial_pers_levels.sort()

    # We'll collect all mutation counts into a dictionary:
    # {pers_level: [mutation_count_for_sim1, mutation_count_for_sim2, ...], ...}
    plot_data_including_zeros = {}

    pers_levels = sorted(plot_data_including_zeros.keys(), reverse=True)

    for pers_level in initial_pers_levels:
        # Subset for this persistence level
        pers_data = data[data['pers_level'] == pers_level]
        sims = pers_data['simulation_id'].unique()

        mutation_counts_including_zeros = []

        for sim_id in sims:
            sim_data = pers_data[pers_data['simulation_id'] == sim_id]
            # Sum total mutations that have freq >= threshold
            mutation_count = sim_data[sim_data['frequency'] >= freq_thresh]['mutation_count'].sum()
            mutation_counts_including_zeros.append(mutation_count)

        plot_data_including_zeros[pers_level] = mutation_counts_including_zeros

    # -------------------------------------------------------------------------
    # 4) PRINT DISTRIBUTION TABLE (OPTIONAL)
    # -------------------------------------------------------------------------
    print(f"\n=== Number of simulations by total mutation count (≥ {int(freq_thresh * 100)}%), Day={day} ===")
    for pers_level in initial_pers_levels:
        # We'll see how many have exactly 0, 1, 2, etc. mutations
        counts = Counter(plot_data_including_zeros[pers_level])
        print(f"\nPersistence Level = {pers_level}")
        for m_count in sorted(counts.keys()):
            print(f"   {counts[m_count]} simulations have {m_count} mutation(s)")
    print("==============================================================\n")

    # -------------------------------------------------------------------------
    # 5) PLOT STACKED BAR CHART
    # -------------------------------------------------------------------------
    plot_stacked_bar(plot_data_including_zeros, freq_thresh, day)

def plot_stacked_bar(plot_data_including_zeros, freq_thresh, day):
    """
    Create and save a stacked bar plot showing the fraction of simulations
    having 0,1,2, or >=3 high-frequency mutations for each pers_level.
    """
    # 1) Extract and sort the persistence levels
    pers_levels = sorted(plot_data_including_zeros.keys(), reverse=True)

    # 2) For each pers_level, clamp mutation counts >3 to 3 and count
    categories = [0, 1, 2, 3]  # 3 stands for "3 or more"
    counts_dict = {}
    total_sims_dict = {}

    for pl in pers_levels:
        total_sims = len(plot_data_including_zeros[pl])
        total_sims_dict[pl] = total_sims

        # Clamp each sim's mutation count
        clamped_counts = [cnt if cnt <= 3 else 3 for cnt in plot_data_including_zeros[pl]]
        counter = Counter(clamped_counts)
        # Build the order [count(0), count(1), count(2), count(3)]
        cat_counts = [counter.get(cat, 0) for cat in categories]
        counts_dict[pl] = cat_counts

    # 3) Convert raw counts to fractions
    fraction_matrix = []
    for pl in pers_levels:
        cat_counts = counts_dict[pl]
        total_sims = total_sims_dict[pl]
        if total_sims == 0:
            fraction_matrix.append([0, 0, 0, 0])
        else:
            fraction_matrix.append([count / total_sims for count in cat_counts])

    fraction_matrix = np.array(fraction_matrix)  # shape (num_pers_levels, 4)

    # 4) Plot the stacked bar chart
    #sns.set(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 14,
    })

    fig, ax = plt.subplots(figsize=(5, 5), dpi=600)

    x_positions = np.arange(len(pers_levels))

    # Choose a different color palette and modify purple and green
    category_colors = sns.color_palette("Dark2", n_colors=len(categories))
    # Replace the default green (index 0) and purple (index 2) with new colors
    category_colors[0] = (0.1, 0.5, 0.6)  # new color for the green segment
    category_colors[2] = (0.8, 0.60, 0.1)  # new color for the purple segment
    category_colors[3] = (0.3, 0.7, 0.34)
    # Optionally reduce alpha
    category_colors = [(r, g, b, 0.85) for r, g, b in category_colors]

    bottoms = np.zeros(len(pers_levels))

    for j, cat in enumerate(categories):
        ax.bar(
            x_positions,
            fraction_matrix[:, j],
            bottom=bottoms,
            color=category_colors[j],
            edgecolor='none',
            width=0.6,
            label=f"Mutations = {cat}",
        )
        bottoms += fraction_matrix[:, j]

    # Tidy up
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["High\nPersistence", "Low\nPersistence"])
    ax.set_xlabel("Persistence Level")  # Match x-label from first script
    ax.set_ylabel("Fraction of Populations")
    ax.set_ylim([0, 1])
    ax.set_title("Simulations")  # Keep original title of second plot

    # Update legend to match first script
    # Rename the last label to "3+" for clarity as in first script
    labels = [f"{i}" for i in categories]
    labels[-1] = "3+"

    # Use the same legend positioning as first script
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Mutations")

    fig.tight_layout()
    
    # Get the directory where the script is located for saving the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_filename = f"Fig_5a.png"
    output_plot_path = os.path.join(script_dir, plot_filename)
    plt.savefig(output_plot_path, dpi=300)


if __name__ == "__main__":
    main()
