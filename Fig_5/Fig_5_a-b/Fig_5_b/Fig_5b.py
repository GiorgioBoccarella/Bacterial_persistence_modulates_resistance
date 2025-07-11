import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------------------------------------------
# 1) LOAD PRE-FILTERED DATA
# -----------------------------------------------------------------------------
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
complete_data_file = os.path.join(script_dir, "..", "..", "..", "complete_data", "complete_data.xlsx")
data_filtered_subset = pd.read_excel(complete_data_file, sheet_name='Fig_5_b')

# Convert columns back to category if needed (read_csv might not preserve them)
data_filtered_subset['AB'] = data_filtered_subset['AB'].astype('category')
data_filtered_subset['nutrient'] = data_filtered_subset['nutrient'].astype('category')
data_filtered_subset['pop'] = data_filtered_subset['pop'].astype('category')

# -----------------------------------------------------------------------------
# 5) UNION OF HIGH-FREQUENCY MUTATIONS ACROSS AB=12.5 & 25
#    For each (nutrient, pop), collect all mutation_ids >90% in either AB condition
# -----------------------------------------------------------------------------
high_freq = data_filtered_subset[data_filtered_subset['frequency'] > 80]

# Group by (nutrient, pop, mutation_id) ignoring AB, because we want the union
union_highfreq = high_freq.groupby(['nutrient', 'pop', 'mutation_id'], observed=True).size().reset_index(name='dummy')

# Now, for each nutrient & pop, count how many unique mutation_ids appear
mutation_counts = (
    union_highfreq
    .groupby(['nutrient', 'pop'], observed=True)['mutation_id']
    .nunique()
    .reset_index(name='n_mutation')
)

# -----------------------------------------------------------------------------
# 6) CLAMP n_mutation > 3 TO 3
# -----------------------------------------------------------------------------
mutation_counts['n_mutation_clamped'] = mutation_counts['n_mutation'].apply(lambda x: x if x <= 3 else 3)

# -----------------------------------------------------------------------------
# 7) Ensure we include pops that might have zero high-freq mutations
# -----------------------------------------------------------------------------
all_combos = data_filtered_subset[['nutrient', 'pop']].drop_duplicates()
mutation_counts = pd.merge(
    all_combos,
    mutation_counts,
    on=['nutrient', 'pop'],
    how='left'
)
mutation_counts['n_mutation'] = mutation_counts['n_mutation'].fillna(0)
mutation_counts['n_mutation_clamped'] = mutation_counts['n_mutation_clamped'].fillna(0)

# -----------------------------------------------------------------------------
# 8) WE ONLY WANT TWO DISTINCT NUTRIENTS
#    (Verify we have exactly 2 nutrient levels)
# -----------------------------------------------------------------------------
unique_nutrients = sorted(mutation_counts['nutrient'].unique(), key=float)
if len(unique_nutrients) < 2:
    raise ValueError("Fewer than 2 nutrient levels found in the data.")
elif len(unique_nutrients) > 2:
    print("Warning: More than 2 nutrient levels remain; using only the first two.")
    unique_nutrients = unique_nutrients[:2]

# -----------------------------------------------------------------------------
# 9) COMPUTE FRACTION OF POPS IN EACH MUTATION CATEGORY PER NUTRIENT
# -----------------------------------------------------------------------------
categories = [0, 1, 2, 3]

# For each nutrient, how many replicate pops total?
nut_to_popcount = (
    mutation_counts.groupby('nutrient', observed=True)['pop'].nunique()
    .to_dict()
)

# For each (nutrient, category) => number of pops
counts_dict = {}
for nut in unique_nutrients:
    sub_df = mutation_counts[mutation_counts['nutrient'] == nut]
    # group by n_mutation_clamped, count replicate pops
    cat_counts = sub_df.groupby('n_mutation_clamped')['pop'].nunique()
    # store
    counts_dict[nut] = [cat_counts.get(cat, 0) for cat in categories]

# Convert to fraction
fraction_dict = {}
for nut in unique_nutrients:
    total_reps = nut_to_popcount[nut]
    fraction_dict[nut] = [
        c / total_reps if total_reps else 0 for c in counts_dict[nut]
    ]

# Make arrays that align with the nutrient order
fraction_matrix = np.array([
    fraction_dict[unique_nutrients[0]],
    fraction_dict[unique_nutrients[1]]
])

# -----------------------------------------------------------------------------
# 10) PLOT (2 BARS TOTAL), STACKED BY n_mutation_clamped
# -----------------------------------------------------------------------------

# -- Make it "publication-ready" style
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

# x positions for the bars
x_positions = np.arange(len(unique_nutrients))

# Use the EXACT same color palette from the first script
category_colors = sns.color_palette("Dark2", n_colors=len(categories))
# Replace colors to match first script
category_colors[0] = (0.1, 0.5, 0.6)    # new color for the "0 mutations" segment
category_colors[2] = (0.8, 0.60, 0.1)   # new color for the "2 mutations" segment
category_colors[3] = (0.3, 0.7, 0.34)   # new color for the "3+ mutations" segment
# Match alpha from first script (0.85)
category_colors = [(r, g, b, 0.85) for r, g, b in category_colors]

# Plot stacked bars
bottoms = np.zeros(len(unique_nutrients))

for j, cat in enumerate(categories):
    ax.bar(
        x_positions,
        fraction_matrix[:, j],
        bottom=bottoms,
        color=category_colors[j],
        edgecolor='none',
        width=0.6,
        label=f"Mutations = {cat}"  # Match label format from first script
    )
    bottoms += fraction_matrix[:, j]

# Tidy up
ax.set_xticks(x_positions)
ax.set_xticklabels(["High\nPersistence", "Low\nPersistence"])
ax.set_xlabel("Persistence Level")  # Match x-label from first script
ax.set_ylabel("Fraction of Populations")
ax.set_ylim([0, 1])
ax.set_title("Experimental")  # Keep original title of second plot

# Update legend to match first script
# Rename the last label to "3+" for clarity as in first script
labels = [f"{i}" for i in categories]
labels[-1] = "3+"

# Use the same legend positioning as first script
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Mutations")

fig.tight_layout()
# save in the same directory as the script
output_plot_path = os.path.join(script_dir, "Fig_5b.png")
plt.savefig(output_plot_path, dpi=300)
