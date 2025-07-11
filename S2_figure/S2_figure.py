#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Fixed day for analysis for simulation data
DAY_TO_FILTER_SIM = 12
# Number of levels to pick from each dataset for the 2x2 plot
NUM_LEVELS_TO_PLOT = 2

def load_and_process_sim_data(base_script_dir, day_to_filter):
    """Loads and processes simulation data."""
    # Navigate to complete_data folder
    complete_data_file = os.path.join(os.path.dirname(base_script_dir), 'complete_data', 'complete_data.xlsx')
    try:
        data = pd.read_excel(complete_data_file, sheet_name='Fig_5_a')
    except FileNotFoundError:
        print(f"Error: Complete data file not found at {complete_data_file}.")
        return pd.DataFrame(), []

    if 'frequency' in data.columns:
        data['frequency'] = pd.to_numeric(data['frequency'], errors='coerce')
    if 'mutation_count' in data.columns:
        data['mutation_count'] = pd.to_numeric(data['mutation_count'], errors='coerce')

    data = data[data['day'] == day_to_filter]
    if data.empty:
        print(f"No simulation data available for day={day_to_filter} after loading.")
        return pd.DataFrame(), []

    initial_pers_levels = sorted(data['pers_level'].unique())
    frequency_thresholds_to_test = np.linspace(0.1, 1.0, 50)
    results_data = []

    for pers_level in initial_pers_levels:
        pers_data_subset = data[data['pers_level'] == pers_level]
        if pers_data_subset.empty:
            continue
        
        sim_ids_for_pers_level = pers_data_subset['simulation_id'].unique()
        if len(sim_ids_for_pers_level) == 0:
            continue

        for freq_thresh in frequency_thresholds_to_test:
            mutation_counts_at_this_thresh = []
            for sim_id in sim_ids_for_pers_level:
                sim_data_for_thresh = pers_data_subset[pers_data_subset['simulation_id'] == sim_id]
                current_sim_total_mutations = sim_data_for_thresh[sim_data_for_thresh['frequency'] >= freq_thresh]['mutation_count'].sum()
                mutation_counts_at_this_thresh.append(current_sim_total_mutations)

            clamped_counts = [cnt if cnt <= 3 else 3 for cnt in mutation_counts_at_this_thresh]
            category_counter = Counter(clamped_counts)
            total_sims = len(sim_ids_for_pers_level)

            results_data.append({
                'pers_level': pers_level,
                'freq_thresh': freq_thresh,
                'category_0_frac': category_counter.get(0, 0) / total_sims if total_sims > 0 else 0,
                'category_1_frac': category_counter.get(1, 0) / total_sims if total_sims > 0 else 0,
                'category_2_frac': category_counter.get(2, 0) / total_sims if total_sims > 0 else 0,
                'category_3plus_frac': category_counter.get(3, 0) / total_sims if total_sims > 0 else 0
            })
    
    plot_df = pd.DataFrame(results_data)
    return plot_df, initial_pers_levels

def load_and_process_emp_data(base_script_dir):
    """Loads and processes empirical data."""
    # Navigate to complete_data folder
    complete_data_file = os.path.join(os.path.dirname(base_script_dir), 'complete_data', 'complete_data.xlsx')
    try:
        data_empirical = pd.read_excel(complete_data_file, sheet_name='Fig_5_b')
    except FileNotFoundError:
        print(f"Error: Complete data file not found at {complete_data_file}.")
        return pd.DataFrame(), []

    data_empirical['AB'] = data_empirical['AB'].astype('category')
    data_empirical['nutrient'] = data_empirical['nutrient'].astype('category')
    data_empirical['pop'] = data_empirical['pop'].astype('category')
    
    try:
        unique_nutrient_levels = sorted(data_empirical['nutrient'].unique(), key=float)
    except ValueError:
        unique_nutrient_levels = sorted(data_empirical['nutrient'].unique())
        
    frequency_thresholds_to_test = np.linspace(0.1, 1.0, 50)
    results_data = []

    for nutrient_level in unique_nutrient_levels:
        all_pops_for_nutrient_df = data_empirical[data_empirical['nutrient'] == nutrient_level][['pop']].drop_duplicates()
        if all_pops_for_nutrient_df.empty:
            continue
        total_pops_for_this_nutrient = len(all_pops_for_nutrient_df)

        for freq_thresh in frequency_thresholds_to_test:
            # Empirical frequency is 0-100, threshold is 0-1
            high_freq_subset = data_empirical[data_empirical['frequency'] > (freq_thresh * 100)] 

            if high_freq_subset.empty:
                n_mutation_clamped_counts = Counter({0: total_pops_for_this_nutrient})
            else:
                union_highfreq_for_thresh = high_freq_subset.groupby(
                    ['nutrient', 'pop', 'mutation_id'], observed=True
                ).size().reset_index(name='dummy')
                
                mutation_counts_for_thresh = union_highfreq_for_thresh.groupby(
                    ['nutrient', 'pop'], observed=True
                )['mutation_id'].nunique().reset_index(name='n_mutation')

                current_nutrient_mutation_counts = mutation_counts_for_thresh[
                    mutation_counts_for_thresh['nutrient'] == nutrient_level
                ]
                merged_counts = pd.merge(
                    all_pops_for_nutrient_df,
                    current_nutrient_mutation_counts[['pop', 'n_mutation']],
                    on='pop',
                    how='left'
                )
                merged_counts['n_mutation'] = merged_counts['n_mutation'].fillna(0)
                merged_counts['n_mutation_clamped'] = merged_counts['n_mutation'].apply(lambda x: x if x <= 3 else 3)
                n_mutation_clamped_counts = Counter(merged_counts['n_mutation_clamped'])
            
            results_data.append({
                'nutrient': nutrient_level,
                'freq_thresh': freq_thresh,
                'category_0_frac': n_mutation_clamped_counts.get(0, 0) / total_pops_for_this_nutrient if total_pops_for_this_nutrient > 0 else 0,
                'category_1_frac': n_mutation_clamped_counts.get(1, 0) / total_pops_for_this_nutrient if total_pops_for_this_nutrient > 0 else 0,
                'category_2_frac': n_mutation_clamped_counts.get(2, 0) / total_pops_for_this_nutrient if total_pops_for_this_nutrient > 0 else 0,
                'category_3plus_frac': n_mutation_clamped_counts.get(3, 0) / total_pops_for_this_nutrient if total_pops_for_this_nutrient > 0 else 0
            })

    plot_df_empirical = pd.DataFrame(results_data)
    return plot_df_empirical, unique_nutrient_levels

def plot_subplot_content(ax, subset_df, level_value, data_type_label, level_label_value, is_empirical, column_specific_title_suffix, subplot_label):
    """Plots data on a given subplot axis."""
    category_colors_base = sns.color_palette("Dark2", n_colors=4)
    final_colors = [
        (0.1, 0.5, 0.6),    # Category 0
        category_colors_base[1], # Category 1
        (0.8, 0.60, 0.1),   # Category 2
        (0.3, 0.7, 0.34)    # Category 3+
    ]
    final_colors_with_alpha = [(r, g, b, 0.85) for r, g, b in final_colors]

    marker_style = 'o' if is_empirical else '.'
    
    ax.plot(subset_df['freq_thresh'], subset_df['category_0_frac'], label='0 Mutations', marker=marker_style, color=final_colors_with_alpha[0])
    ax.plot(subset_df['freq_thresh'], subset_df['category_1_frac'], label='1 Mutation', marker=marker_style, color=final_colors_with_alpha[1])
    ax.plot(subset_df['freq_thresh'], subset_df['category_2_frac'], label='2 Mutations', marker=marker_style, color=final_colors_with_alpha[2])
    ax.plot(subset_df['freq_thresh'], subset_df['category_3plus_frac'], label='3+ Mutations', marker=marker_style, color=final_colors_with_alpha[3])

    ax.set_xlabel('Frequency Threshold (Proportion)')
    ax.set_ylabel('Fraction of Populations')
    
    base_title = ""
    if is_empirical:
        base_title = f'{data_type_label} Data\nNutrient Level: {level_label_value}'
    else:
        base_title = f'{data_type_label} Data (Day {DAY_TO_FILTER_SIM})\nPersistence Level: {level_label_value:.5f}'
    final_title = f'{base_title}\n{column_specific_title_suffix}'
    ax.set_title(final_title)
    
    # Add subplot label (a, b, c, d) in the top-left corner
    ax.text(0.02, 0.98, subplot_label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.legend(title='Mutation Category', fontsize=9) # Smaller legend for subplots
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0.1, 1.1, 0.1))
    ax.set_xlim(0.3, 1.0) # Adjusted xlim


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load and process data
    sim_plot_df, sim_pers_levels = load_and_process_sim_data(script_dir, DAY_TO_FILTER_SIM)
    emp_plot_df, emp_nutrient_levels = load_and_process_emp_data(script_dir)

    if sim_plot_df.empty and emp_plot_df.empty:
        print("No data available from either simulation or empirical sources. Exiting.")
        return

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10, # Adjusted for subplots
        "axes.labelsize": 12,
        "axes.titlesize": 13, # Adjusted for subplots
        "legend.fontsize": 9, # Default, can be overridden in plot_subplot_content
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axs_flat = axs.flatten() # For easier iteration

    # Plot simulation data (top row, inverted)
    sim_levels_to_plot = sim_pers_levels[:NUM_LEVELS_TO_PLOT]
    # Target axes for sim data: top-right (index 1), then top-left (index 0)
    sim_target_axes_indices = [1, 0] 
    # Subplot labels for simulation data
    sim_subplot_labels = ['b)', 'a)']  # b) for top-right, a) for top-left

    for i, pers_level in enumerate(sim_levels_to_plot):
        if i < NUM_LEVELS_TO_PLOT: # Ensure we are within the number of levels to plot
            ax_idx = sim_target_axes_indices[i]
            subplot_label = sim_subplot_labels[i]
            column_suffix = "High persistence level" if ax_idx % 2 == 0 else "Low persistence level"
            subset_df = sim_plot_df[sim_plot_df['pers_level'] == pers_level]
            if not subset_df.empty:
                plot_subplot_content(axs_flat[ax_idx], subset_df, pers_level, "Simulated", pers_level, is_empirical=False, column_specific_title_suffix=column_suffix, subplot_label=subplot_label)
            else:
                base_empty_title = f"Simulated Data - Persistence Level {pers_level:.5f}\n(No data or not enough levels)"
                axs_flat[ax_idx].set_title(f"{base_empty_title}\n{column_suffix}")
                axs_flat[ax_idx].axis('off')
        else:
            break 
            
    # Turn off unused sim plot axes if fewer than NUM_LEVELS_TO_PLOT sim_levels exist
    # This needs to check the original target axes for emptiness if a level wasn't plotted
    plotted_sim_axes = []
    if len(sim_levels_to_plot) > 0:
        plotted_sim_axes.append(sim_target_axes_indices[0]) # First sim plot
    if len(sim_levels_to_plot) > 1:
        plotted_sim_axes.append(sim_target_axes_indices[1]) # Second sim plot

    for i in range(NUM_LEVELS_TO_PLOT): # Iterate up to NUM_LEVELS_TO_PLOT (0 and 1)
        ax_idx_to_check = sim_target_axes_indices[i] # Check 1 then 0
        column_suffix_for_unused = "High persistence level" if ax_idx_to_check % 2 == 0 else "Low persistence level"
        if ax_idx_to_check not in plotted_sim_axes and i < len(sim_levels_to_plot):
            # This case implies data was empty for a level that *should* have plotted
            # The title indicating empty data would have been set above.
            pass # Already handled
        elif ax_idx_to_check not in plotted_sim_axes:
            # This case means there were fewer than NUM_LEVELS_TO_PLOT actual levels
            axs_flat[ax_idx_to_check].set_title(f"Simulated Data - (No further levels)\n{column_suffix_for_unused}")
            axs_flat[ax_idx_to_check].axis('off')


    # Plot empirical data (bottom row)
    emp_levels_to_plot = emp_nutrient_levels[:NUM_LEVELS_TO_PLOT]
    # Subplot labels for empirical data
    emp_subplot_labels = ['c)', 'd)']  # c) for bottom-left, d) for bottom-right
    
    for i, nutrient_level in enumerate(emp_levels_to_plot):
        ax_idx = i + NUM_LEVELS_TO_PLOT # Start plotting from axs[1,0] (index 2 in flattened array)
        if ax_idx < 4: 
            subplot_label = emp_subplot_labels[i]
            column_suffix = "High persistence level" if ax_idx % 2 == 0 else "Low persistence level"
            subset_df = emp_plot_df[emp_plot_df['nutrient'] == nutrient_level]
            if not subset_df.empty:
                plot_subplot_content(axs_flat[ax_idx], subset_df, nutrient_level, "Empirical", str(nutrient_level), is_empirical=True, column_specific_title_suffix=column_suffix, subplot_label=subplot_label)
            else:
                base_empty_title = f"Empirical Data - Nutrient Level {nutrient_level}\n(No data or not enough levels)"
                axs_flat[ax_idx].set_title(f"{base_empty_title}\n{column_suffix}")
                axs_flat[ax_idx].axis('off')
        else: # Should not happen
            break

    # Turn off unused empirical plot axes
    for i in range(len(emp_levels_to_plot), NUM_LEVELS_TO_PLOT):
        ax_idx = i + NUM_LEVELS_TO_PLOT
        if ax_idx < 4:
            column_suffix_for_unused = "High persistence level" if ax_idx % 2 == 0 else "Low persistence level"
            axs_flat[ax_idx].set_title(f"Empirical Data - (No further levels)\n{column_suffix_for_unused}")
            axs_flat[ax_idx].axis('off')
    
    fig.suptitle(f'Combined Analysis: Effect of Frequency Threshold on Mutation Categories', fontsize=18, y=1.03)
    
    output_plot_path = os.path.join(script_dir, "S2_figure.png")
    try:
        plt.savefig(output_plot_path, dpi=300)
        print(f"Saved S2 figure plot: {output_plot_path}")
    except Exception as e:
        print(f"Error saving S2 figure plot {output_plot_path}: {e}")
    plt.close(fig)

if __name__ == "__main__":
    main() 