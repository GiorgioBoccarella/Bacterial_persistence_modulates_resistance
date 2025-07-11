#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mutation-analysis pipeline
──────────────────────────
*   Filters E. coli mutation data
*   Flags genes of interest (manual + resistance GO)
*   Computes diversity metrics
*   Builds a bar-plot of "fraction of replicate populations" per gene
*   Prints diagnostic replicate counts
───────────────────────────────────────────────────────────────────
"""

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

from goatools import obo_parser
from goatools.associations import read_gaf

# ════════════════════════════════════════════════════════════════
# 1) File paths  (edit if your files live elsewhere)
# ════════════════════════════════════════════════════════════════
GO_OBO_FILE = 'go.obo'          # go-basic.obo
GO_GAF_FILE = 'ecocyc.gaf'      # E. coli annotation GAF
# ────────────────────────────────────────────────────────────────
# Change working directory to this script's location
# ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ════════════════════════════════════════════════════════════════
# 2) Load & pre-filter the mutation table
# ════════════════════════════════════════════════════════════════
complete_data_file = os.path.join(script_dir, "..", "..", "complete_data", "complete_data.xlsx")
df = pd.read_excel(complete_data_file, sheet_name='Fig_5_c')

# numeric conversion
#df['MIC']        = pd.to_numeric(df['MIC'], errors='coerce') # Commented out MIC conversion
df['Antibiotic'] = pd.to_numeric(df['AB'],  errors='coerce')
df['frequency']  = pd.to_numeric(df['frequency'].astype(str).str.rstrip('%'), errors='coerce') # Ensure string type before rstrip

# keep only the conditions analysed downstream
#df = df[df['MIC'] > 0]
df = df[df['Antibiotic'].isin([12.5, 25])]
df = df[df['nutrient'].isin([0.25, 0.8])]

# Store this version of df for true replicate counting
df_for_true_replicate_counts = df.copy()

# ════════════════════════════════════════════════════════════════
# 3) Clean gene name strings  →  df_exploded
# ════════════════════════════════════════════════════════════════
def clean_gene_names(gene):
    gene = re.sub(r'[←→\[\]]', '', str(gene))
    gene = gene.replace('–', '-').replace('—', '-')
    genes_list = re.split(r'[\/,;]| - ', gene)
    final_list = []
    for g in genes_list:
        final_list.extend(g.split('-'))
    return [x.strip() for x in final_list if x.strip()]

df['gene_list'] = df['gene'].apply(clean_gene_names)
df_exploded = df.explode('gene_list')
df_exploded['gene_list_upper'] = df_exploded['gene_list'].str.upper()

# ════════════════════════════════════════════════════════════════
# 4) Manual gene list flag from the paper
# ════════════════════════════════════════════════════════════════
manual_genes = [
    'gltS', 'speG', 'rrlH', 'hdhA', 'sbmA', 'cpsB', 'fusA', 'ubiB',
    'cyoA', 'cpxA', 'atpF', 'atpG', 'arcB', 'wbbK', 'cpxR',
    'yaiW', 'gltA', 'yqhA', 'topA', 'ndh', 'ykgH', 'ppiD',
    'crr', 'fre', 'yiaY', 'rpoC', 'arcA', "arc"
]
manual_upper = [g.upper() for g in manual_genes]

df_exploded['in_manual_list'] = (
    df_exploded['gene_list_upper'].isin(manual_upper) |
    df_exploded['gene_list_upper'].str.startswith('ARC')
)
# explicitly exclude NUO* and AMP*
df_exploded.loc[df_exploded['gene_list_upper'].str.startswith(('AMP')),
                'in_manual_list'] = True

# ════════════════════════════════════════════════════════════════
# 5) GO term mapping & resistance keyword flag
# ════════════════════════════════════════════════════════════════
godag     = obo_parser.GODag(GO_OBO_FILE)
gene2go   = read_gaf(GO_GAF_FILE, godag=godag, namespace='BP')

# quick symbol→GOid dict
symbol2gos = {}
with open(GO_GAF_FILE) as gaf:
    for ln in gaf:
        if ln.startswith('!'):             # header lines
            continue
        cols = ln.strip().split('\t')
        if len(cols) > 5:
            symbol2gos.setdefault(cols[2].upper(), set()).add(cols[4])

df_exploded['go_terms'] = df_exploded['gene_list_upper'].map(symbol2gos)

df_exploded = df_exploded[df_exploded['go_terms'].notna()].explode('go_terms')

goid2name = {goid: term.name for goid, term in godag.items()}
df_exploded['go_name'] = df_exploded['go_terms'].map(goid2name)

def is_resistance(name):
    if pd.isna(name):
        return False
    kws = ('antibiotic', 'resistance')
    return any(k in str(name).lower() for k in kws)

df_exploded['is_resistance_go'] = df_exploded['go_name'].apply(is_resistance)

# ════════════════════════════════════════════════════════════════
# 6) High-frequency filter, then combine criteria
# ════════════════════════════════════════════════════════════════
freq_cutoff = 0
df_exploded['is_high_freq'] = df_exploded['frequency'] > freq_cutoff

print("\n=== Debugging info before creating analysis_df ===")
print(f"Number of rows in df_exploded (post-GO processing): {len(df_exploded)}")
if not df_exploded.empty:
    print(f"Sample of df_exploded (first 5 rows):\\n{df_exploded.head().to_string()}") # use to_string for better console output
    print(f"Columns in df_exploded: {df_exploded.columns.tolist()}")
    print(f"Stats for is_high_freq:\\n{df_exploded['is_high_freq'].value_counts(dropna=False)}")
    print(f"Stats for in_manual_list:\\n{df_exploded['in_manual_list'].value_counts(dropna=False)}")
    if 'is_resistance_go' in df_exploded.columns:
        print(f"Stats for is_resistance_go:\\n{df_exploded['is_resistance_go'].value_counts(dropna=False)}")
        cond_resistance_go = df_exploded['is_resistance_go']
    else:
        print("'is_resistance_go' column not found in df_exploded at this point.")
        cond_resistance_go = pd.Series([False] * len(df_exploded), index=df_exploded.index) # Placeholder if missing

    cond_manual = df_exploded['in_manual_list']
    cond_high_freq = df_exploded['is_high_freq']
    
    combined_criteria = cond_high_freq & (cond_manual | cond_resistance_go)
    print(f"Number of rows satisfying combined criteria (high_freq AND (manual OR resistance_go)): {combined_criteria.sum()}")
else:
    print("df_exploded is empty before mask application.")


mask = (
    df_exploded['is_high_freq'] &
    (df_exploded['in_manual_list'] | df_exploded['is_resistance_go'])
)
analysis_df = df_exploded[mask]

if analysis_df.empty:
    print(f"No genes meet: frequency > {freq_cutoff} "
          "AND (manual list OR resistance GO).")
    exit(0)

# ════════════════════════════════════════════════════════════════
# 7)  ── FIX: replicate accounting ───────────────────────────────
#     Pop-IDs are unique **only within** a nutrient × antibiotic
#     combination.  We therefore:
#        • count replicates in each combo,
#        • sum across the two antibiotics to get the per-nutrient
#          denominators for the bar-plot.
# ════════════════════════════════════════════════════════════════

# Calculate TRUE total number of unique populations per condition from initial filtered df
actual_replicates_per_condition_group = (
    df_for_true_replicate_counts[['nutrient', 'Antibiotic', 'pop']]
    .drop_duplicates()
    .groupby(['nutrient', 'Antibiotic'])['pop']
    .nunique()
)

actual_total_replicates_per_nutrient = (
    actual_replicates_per_condition_group
    .groupby(level='nutrient')
    .sum()
    .to_dict()
)

# Original calculation based on populations with high-frequency mutations
hf = df_exploded[df_exploded['is_high_freq']]
combo_reps_with_hf_mut = (
    hf[['nutrient', 'Antibiotic', 'pop']]
      .drop_duplicates()
      .groupby(['nutrient', 'Antibiotic'])['pop']
      .nunique()                       # e.g. (0.25, 12.5) → 9
)

tot_reps_with_hf_mut = (
    combo_reps_with_hf_mut
      .groupby(level='nutrient')
      .sum()                           # e.g. 0.25 → 9 + 2  = 11
      .to_dict()
)

# diagnostic printout
print("\n=== ACTUAL TOTAL EXPERIMENTAL REPLICATES ===")
for nut in sorted(actual_total_replicates_per_nutrient):
    print(f"  nutrient {nut:>4}: {actual_total_replicates_per_nutrient[nut]} total experimental replicate pops")
print("\nPer nutrient × antibiotic combination (actual experimental replicates):")
for (nut_val, ab_val), n_val in actual_replicates_per_condition_group.items():
    print(f"  nutrient {nut_val:>4} & antibiotic {ab_val:>5}: {n_val}")

print("\nDetailed pop IDs per actual condition group (from mutations_above_4.csv after initial filtering):")
for group_keys, group_df in df_for_true_replicate_counts[['nutrient', 'Antibiotic', 'pop']].drop_duplicates().groupby(['nutrient', 'Antibiotic'], observed=False):
    nut_val, ab_val = group_keys
    unique_pops_in_group = sorted(list(group_df['pop'].unique()))
    print(f"  Nutrient {str(nut_val):<4}, AB {str(ab_val):<5}: Count={len(unique_pops_in_group)} Pop IDs: {unique_pops_in_group}")

print("=======================================================\n")

print("\n=== REPLICATES WITH HIGH-FREQUENCY MUTATIONS OF INTEREST (original diagnostic) ===")
for nut_hf in sorted(tot_reps_with_hf_mut):
    print(f"  nutrient {nut_hf:>4}: {tot_reps_with_hf_mut[nut_hf]} total replicate pops with HF mutations")
print("\nPer nutrient × antibiotic combination (with HF mutations):")
for (nut_hf_combo, ab_hf_combo), n_hf_combo in combo_reps_with_hf_mut.items():
    print(f"  nutrient {nut_hf_combo:>4} & antibiotic {ab_hf_combo:>5}: {n_hf_combo}")
print("=================================================================================\n")

# ════════════════════════════════════════════════════════════════
# 8)  Diversity metrics (unchanged)
# ════════════════════════════════════════════════════════════════
div_res = []
for nut in analysis_df['nutrient'].unique():
    sub_nut = analysis_df[analysis_df['nutrient'] == nut]
    for ab in sub_nut['Antibiotic'].unique():
        sub_ab = sub_nut[sub_nut['Antibiotic'] == ab]
        for pop in sub_ab['pop'].unique():
            sub_pop = sub_ab[sub_ab['pop'] == pop]
            cts     = Counter(sub_pop['go_name'])
            div_res.append({
                'Nutrient'        : nut,
                'Antibiotic'      : ab,
                'Population'      : pop,
                'Richness'        : len(cts),
                'Shannon Entropy' : entropy(np.array(list(cts.values())))
            })

print("Diversity metrics calculated.")

# ════════════════════════════════════════════════════════════════
# 9)  Detailed table (unchanged)
# ════════════════════════════════════════════════════════════════
detailed = analysis_df[[
    'pop', 'nutrient', 'Antibiotic', 'frequency', # Removed 'MIC' and 'pers_frac'
    'gene_list', 'go_name'
]].drop_duplicates().rename(columns={
    'pop'      : 'Population',
    'nutrient' : 'Nutrient',
    'go_name'  : 'Resistance Mechanism',
    'gene_list': 'Gene',
    'frequency': 'Frequency'
})

# mean_mic = (
#     detailed.groupby('Gene')['MIC'].mean()
#             .reset_index().rename(columns={'MIC': 'Mean_MIC'})
# )
# detailed = detailed.merge(mean_mic, on='Gene') # Removed mean_mic calculation and merge
print("Detailed table processed.")

# ════════════════════════════════════════════════════════════════
# 10)  Replicate coverage per gene  (FIXED numerator & denominator)
# ════════════════════════════════════════════════════════════════
rep_df = (
    analysis_df[['gene_list', 'nutrient', 'Antibiotic', 'pop']]
      .drop_duplicates()
      .rename(columns={'gene_list': 'Gene'})
)

# numerator: count pops for gene in each combo, then sum across AB
rep_counts = (
    rep_df.groupby(['Gene', 'nutrient', 'Antibiotic'])['pop']
           .nunique()
           .reset_index(name='Num_Reps_in_combo')
           .groupby(['Gene', 'nutrient'])['Num_Reps_in_combo']
           .sum()
           .reset_index(name='Num_Reps')
           .rename(columns={'nutrient': 'Nutrient'})
)

# fraction within nutrient using the ACTUAL total replicates per nutrient as denominator
rep_counts['Perc_Reps'] = (
    rep_counts['Num_Reps'] /
    rep_counts['Nutrient'].map(actual_total_replicates_per_nutrient) # USE THE NEW DENOMINATOR
)

# find top genes by replicate coverage
coverage = (
    rep_counts.groupby('Gene')['Perc_Reps']
              .sum()
              .reset_index(name='Total_Coverage')
              .sort_values('Total_Coverage', ascending=False)
)
top_genes  = coverage.head(10)['Gene'].tolist()
print(f"Top genes by coverage: {top_genes}")

# ════════════════════════════════════════════════════════════════
# Print population IDs for top 10 genes by treatment
# ════════════════════════════════════════════════════════════════
print("\n=== Population IDs for Top 10 Genes by Treatment ===")
# Use rep_df as it already has 'Gene' column and unique pop per gene/nutrient/antibiotic
top_genes_pop_df = rep_df[rep_df['Gene'].isin(top_genes)]

# Group by Gene, Nutrient, and Antibiotic, then list unique pop IDs
grouped_pops = top_genes_pop_df.groupby(['Gene', 'nutrient', 'Antibiotic'])['pop'].apply(lambda x: sorted(list(set(x)))).reset_index()

for gene_name in top_genes: # Iterate in order of top_genes
    gene_specific_data = grouped_pops[grouped_pops['Gene'] == gene_name]
    if not gene_specific_data.empty:
        print(f"\n── Gene: {gene_name} ──")
        for _, row in gene_specific_data.iterrows():
            print(f"  Nutrient: {row['nutrient']}, Antibiotic: {row['Antibiotic']}")
            print(f"    Population IDs: {row['pop']}")
print("=======================================================\n")

# order genes for plotting
pivot = (
    rep_counts[rep_counts['Gene'].isin(top_genes)]
      .pivot(index='Gene', columns='Nutrient', values='Perc_Reps')
      .fillna(0)
)
pivot.sort_values(by=[0.8, 0.25], ascending=False, inplace=True)
ordered   = pivot.index.tolist()

rep_counts = rep_counts[rep_counts['Gene'].isin(top_genes)]
rep_counts['Gene'] = pd.Categorical(rep_counts['Gene'],
                                    categories=ordered, ordered=True)

# ════════════════════════════════════════════════════════════════
# 11)  Bar-plot
# ════════════════════════════════════════════════════════════════
plt.figure(figsize=(4, 4))
ax = sns.barplot(
    data      = rep_counts,
    x         = 'Perc_Reps',
    y         = 'Gene',
    hue       = 'Nutrient',
    hue_order = [0.8, 0.25],          # green then blue
    palette   = ['green', 'blue'],
    alpha     = 0.75
)
ax.get_legend().remove()
plt.xlabel('Fraction of Replicate Pops')
plt.ylabel('Gene')
plt.xlim(0, 0.4)
plt.tight_layout()
#plt.savefig('top_genes_barplot1.png', dpi=300, bbox_inches='tight')
#print("Bar-plot saved as top_genes_barplot.png")

# here alternative plot in which only higher then 0.1 values are shown
# ── NEW: keep only bars with coverage > 0.1 ─────────────────────
thr = 0.01
rep_counts_filt = rep_counts[rep_counts['Perc_Reps'] > thr].copy()

# keep the original gene ranking but drop those that disappeared
ordered_filt = [g for g in ordered if g in rep_counts_filt['Gene'].unique()]
rep_counts_filt['Gene'] = pd.Categorical(rep_counts_filt['Gene'],
                                         categories=ordered_filt,
                                         ordered=True)

plt.figure(figsize=(4, 4.2))
ax = sns.barplot(
    data      = rep_counts_filt,
    x         = 'Perc_Reps',
    y         = 'Gene',
    hue       = 'Nutrient',
    hue_order = [0.8, 0.25],
    palette   = ['green', 'blue'],
    alpha     = 0.75,
)
legend = ax.get_legend()
if legend:
    legend.set_title('Nutrient')
else:
    print("\nNote: Legend not created for the filtered bar plot (top_genes_barplot_thresh0p1.png) as no data met the threshold.")
plt.xlabel('Fraction of Replicate Pops')
plt.ylabel('Gene')
plt.xlim(thr, 0.35)          # start the axis at the threshold (optional)
plt.tight_layout()
plt.savefig('Fig_5_c.png', dpi=300, bbox_inches='tight')
