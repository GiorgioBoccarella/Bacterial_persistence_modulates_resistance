import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import matplotlib.gridspec as gridspec
import pandas as pd
import os
from matplotlib.ticker import LogLocator, MaxNLocator

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set global parameters for a publication-quality plot
plt.rcParams.update({
    'font.size': 8.5,
    'axes.labelsize': 13,
    'axes.titlesize': 8.6,
    'xtick.labelsize': 12,
    'ytick.labelsize': 9,
    'legend.fontsize': 8
})

# Constants and functions
MIC_wt = 2
k = 2
psi_max = 0.7
tau = 5
Psi_min_sensitive = -6
Psi_min_tolerant = -0.5

def psi(c, MIC, psi_max, k, Psi_min):
    numerator = Psi_min * (1 - (c / MIC)**k)
    denominator = (Psi_min / psi_max) - (c / MIC)**k
    psi_c = numerator / denominator
    return psi_c

def survival(c, tau, MIC, psi_max, k, Psi_min):
    psi_t = psi(c, MIC, psi_max, k, Psi_min)
    S_c_tau = np.exp(psi_t * tau)
    return S_c_tau

AB_concentration = 12.5
c = AB_concentration
persister_levels = [0.00005, 0.8]
MIC_R_range = np.linspace(0.1, AB_concentration * 1.5, 1000)

# Initialize list to store all data for CSV export
csv_data = []


MIC_R_fractions = [0.5, 0.8, 1.1]
MIC_R_interest = [fraction * c for fraction in MIC_R_fractions]
indices_interest = [np.argmin(np.abs(MIC_R_range - MIC)) for MIC in MIC_R_interest]

fig = plt.figure(figsize=(4, 4.6), dpi=300)
gs = gridspec.GridSpec(2, 1)

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[1, 0], sharex=ax1)

colors = ['green', 'darkblue']
markersize = 5
annotation_fontsize = 9

for alpha_persister, color in zip(persister_levels, colors):
    selection_coefficients = []
    survival_mutant = []

    # Wild-type survival
    S_w_sensitive = survival(c, tau, MIC_wt, psi_max, k, Psi_min_sensitive)
    S_w_tolerant = survival(c, tau, MIC_wt, psi_max, k, Psi_min_tolerant)
    S_w = (1 - alpha_persister) * S_w_sensitive + alpha_persister * S_w_tolerant
    S_w = max(S_w, 1e-30)

    for MIC_R in MIC_R_range:
        S_r_sensitive = survival(c, tau, MIC_R, psi_max, k, Psi_min_sensitive)
        S_r_tolerant = survival(c, tau, MIC_R, psi_max, k, Psi_min_tolerant)
        S_m = (1 - alpha_persister) * S_r_sensitive + alpha_persister * S_r_tolerant
        S_m = max(S_m, 1e-30)
        selection_coefficient = S_m / S_w
        selection_coefficients.append(selection_coefficient)
        survival_mutant.append(S_m)
        
        # Store data for CSV export
        csv_data.append({
            'antibiotic_concentration': AB_concentration,
            'antibiotic_concentration_ug_ml': AB_concentration/2,  # Convert to μg/mL
            'persister_level': alpha_persister,
            'MIC_R': MIC_R,
            'MIC_R_ratio': MIC_R / AB_concentration,
            'S_w_sensitive': S_w_sensitive,
            'S_w_tolerant': S_w_tolerant,
            'S_w_combined': S_w,
            'S_r_sensitive': S_r_sensitive,
            'S_r_tolerant': S_r_tolerant,
            'S_m_combined': S_m,
            'selection_coefficient': selection_coefficient,
            'survival_mutant': S_m
        })

    selection_coefficients = np.array(selection_coefficients)
    survival_mutant = np.array(survival_mutant)

    # Plot selection coefficient
    ax1.plot(MIC_R_range / AB_concentration, selection_coefficients,
             label=f'Persister = {alpha_persister}', color=color)
    # Plot raw survival
    ax2.plot(MIC_R_range / AB_concentration, survival_mutant,
             label=f'Persister = {alpha_persister}', color=color)

    for i_mic, MIC_R_value in enumerate(MIC_R_interest):
        x_value = MIC_R_value / AB_concentration
        y_value_ax1 = selection_coefficients[indices_interest[i_mic]]
        y_value_ax2 = survival_mutant[indices_interest[i_mic]]

        # On ax1: label below point
        ax1.plot(x_value, y_value_ax1, marker='o', ms=markersize, linestyle='', color=color, alpha=0.7)
        ax1.annotate(f'{MIC_R_fractions[i_mic]:g}',
                     xy=(x_value, y_value_ax1),
                     xytext=(0, -8),
                     textcoords="offset points",
                     ha='center', va='top',
                     color=color, fontsize=annotation_fontsize)

        # On ax2: label below point
        extra_offset = -18 if alpha_persister == 0.00005 and i_mic != 0 else -8
        ax2.plot(x_value, y_value_ax2, marker='o', ms=markersize, linestyle='', color=color, alpha=0.7)
        ax2.annotate(f'{MIC_R_fractions[i_mic]:g}',
                     xy=(x_value, y_value_ax2),
                     xytext=(0, extra_offset),
                     textcoords="offset points",
                     ha='center', va='top',
                     color=color, fontsize=annotation_fontsize)

# Labels
ax1.set_ylabel(r'Relative survival ($\frac{S_m}{S_w}$)')
ax1.set_yscale('log')
ax1.legend(loc='best', frameon=False)

ax2.set_ylabel(r'Survival ($S_m$)')
ax2.set_xlabel(r'$\frac{\mathrm{MIC}_{\text{m}}}{AB_c}$')
ax2.set_yscale('log')

# Reduce the density of ticks
ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
ax1.xaxis.set_major_locator(MaxNLocator(5))

plt.tight_layout()

# Data is now available in the consolidated Excel file

plt.savefig('Fig_1_a-b.png', dpi=300, bbox_inches='tight')
plt.savefig('Fig_1_a-b.pdf', format='pdf', bbox_inches='tight')
# plt.show()
