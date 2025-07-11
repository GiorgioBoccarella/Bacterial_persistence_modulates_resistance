import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# -----------------------------
# 0) USER PARAMETERS
# -----------------------------
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
complete_data_file = os.path.join(script_dir, "..", "..", "complete_data", "complete_data.xlsx")
PERSISTERS = [5e-5, 0.8] # The two persister levels to compare
C_VALUE = 12.5
KAPPA_S = 2

# print unique values of initial_pers_level
df = pd.read_excel(complete_data_file, sheet_name='Fig_3_b')
unique_persister_levels = df["initial_pers_level"].unique()
print("Unique initial_pers_level values in the dataset:")
print(unique_persister_levels)
# unique kappaS values
unique_kappaS_values = df["kappaS"].unique()
print("Unique kappaS values in the dataset:")
print(unique_kappaS_values)


SAMPLE_SIZE = 40  # Number of simulations to randomly sample per persister level

# -----------------------------
# 1) Read & filter data
# -----------------------------
df = pd.read_excel(complete_data_file, sheet_name='Fig_3_b')
df_filtered = df[
    (df["c"] == C_VALUE) &
    (df["kappaS"] == KAPPA_S) &
    (df["initial_pers_level"].isin(PERSISTERS))
].copy()

if df_filtered.empty:
    print("No data after filtering. Check your input.")
    exit(0)

# Identify columns for average_mic and most_common_mic
avg_cols = [c for c in df_filtered.columns if c.startswith("average_mic_hour_")]
most_cols = [c for c in df_filtered.columns if c.startswith("most_common_mic_hour_")]

# -----------------------------
# 2) Reshape for average_mic - CREATE TWO DATASETS
# -----------------------------
if "simulation_id" not in df_filtered.columns:
    df_filtered["simulation_id"] = df_filtered.index

# Dataset 1: All data points (for averages)
long_rows_avg_full = []
for _, row in df_filtered.iterrows():
    sim_id = row["simulation_id"]
    pers = row["initial_pers_level"]
    for col in avg_cols:
        match = re.search(r"average_mic_hour_(\d+)", col)
        if match:
            hour_val = float(match.group(1))
            day_val = hour_val / 24.0
            mic_val = row[col]
            long_rows_avg_full.append({
                "simulation_id": sim_id,
                "initial_pers_level": pers,
                "day": day_val,
                "average_mic": mic_val
            })

# Dataset 2: 6-hour interval data points (for individual simulations)
long_rows_avg_filtered = []
for _, row in df_filtered.iterrows():
    sim_id = row["simulation_id"]
    pers = row["initial_pers_level"]
    for col in avg_cols:
        match = re.search(r"average_mic_hour_(\d+)", col)
        if match:
            hour_val = float(match.group(1))
            # Only include data points at 6-hour intervals
            if hour_val % 6 == 0:  
                day_val = hour_val / 24.0
                mic_val = row[col]
                long_rows_avg_filtered.append({
                    "simulation_id": sim_id,
                    "initial_pers_level": pers,
                    "day": day_val,
                    "average_mic": mic_val
                })

df_long_avg_full = pd.DataFrame(long_rows_avg_full)
df_long_avg = pd.DataFrame(long_rows_avg_filtered)  # Using the variable name from the original script

# -----------------------------
# 3) Compute mean from average_mic (using the FULL dataset)
# -----------------------------
grouped_avg_full = df_long_avg_full.groupby(["initial_pers_level", "day"], as_index=False)
summary_avg_full = grouped_avg_full.agg(
    mean_mic=("average_mic", "mean")
)

# -----------------------------
# 4) Plot: Single Subplot with Individual Simulations and Averages
# -----------------------------
fig, ax_top = plt.subplots(figsize=(5, 3.5), dpi=600)

# Define color map and label map
color_map = {
    5e-5: "green",
    0.8: "darkblue"
}
label_map = {
    5e-5: "Low persistence level",
    0.8: "High persistence level"
}

# First, plot a random sample of individual simulation lines with 6-hour interval data
for pers in PERSISTERS:
    # Get data for this persister level
    pers_data = df_long_avg[df_long_avg["initial_pers_level"] == pers]
    pers_data = pers_data[pers_data["day"] <= 24].copy()  # Limit to 24 days as in original
    
    # Get unique simulation IDs for this persister level
    sim_ids = pers_data["simulation_id"].unique()
    
    # Randomly sample simulation IDs (or take all if fewer than SAMPLE_SIZE)
    sample_size = min(SAMPLE_SIZE, len(sim_ids))
    np.random.seed(2147483648)  # Set seed for reproducibility (same as the new script)
    print(np.random.get_state()[1][0])
    sampled_sim_ids = np.random.choice(sim_ids, size=sample_size, replace=False)
    
    # Plot only the sampled simulations
    for sim_id in sampled_sim_ids:
        sim_data = pers_data[pers_data["simulation_id"] == sim_id].copy()
        sim_data = sim_data.sort_values("day")
        
        # Plot individual lines with markers as specified in the new script
        ax_top.plot(
            sim_data["day"], sim_data["average_mic"],
            color=color_map[pers],
            alpha=0.25,  # Reduced alpha to make average lines stand out more
            linewidth=1.1,  # Slightly thinner
            marker='o',    # Add markers at each data point
            markersize=0,  # No visible markers (from the new script)
            zorder=1       # Lower zorder to ensure they're behind the mean lines
        )

# Now plot the means with continuous lines using FULL dataset
summary_top_full = summary_avg_full[summary_avg_full["day"] <= 24].copy()
for pers in PERSISTERS:
    sub = summary_top_full[summary_top_full["initial_pers_level"] == pers].copy()
    sub.sort_values("day", inplace=True)
    ax_top.plot(
        sub["day"], sub["mean_mic"],
        color=color_map[pers],
        label=label_map[pers],
        linewidth=1.8,  # Using the linewidth from the new script
        marker='o',     # Using marker from the new script
        markersize=0,   # Using marker size from the new script
        zorder=3        # Higher zorder to ensure they're on top
    )

# Enhanced focus on 2-4 MIC range
ax_top.set_xlabel("Days", fontsize=12)
ax_top.set_ylabel("Average MIC", fontsize=12)

# Set log scale with more detailed minor ticks in the 2-4 range
ax_top.set_yscale("log")
ax_top.set_ylim(2, 32)
ax_top.set_xlim(0, 12)

# Custom minor tick locator to add more ticks in the 2-4 range
minorLocator = ticker.LogLocator(subs=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5], numticks=12)
ax_top.yaxis.set_minor_locator(minorLocator)
ax_top.yaxis.set_minor_formatter(ticker.NullFormatter())  # Hide labels for minor ticks

# Force specific major ticks and format them
ax_top.yaxis.set_major_locator(ticker.FixedLocator([2, 4, 8, 16, 32]))
ax_top.yaxis.set_major_formatter(ticker.ScalarFormatter())

# Add grid for better readability especially in the 2-4 range
#ax_top.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

# Make the legend more prominent
ax_top.legend(frameon=True, fontsize=8, loc='upper left')

plt.tight_layout()
output_plot_path = os.path.join(script_dir, "Fig_3_b.png")
plt.savefig(output_plot_path)