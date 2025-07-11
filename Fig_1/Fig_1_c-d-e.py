import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.patches as mpatches  # Add this import for the custom legend

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Constants
t0 = 0  # Starting time
tfin = 15  # End time
tau_max = 0.01  # Maximum time step
kappaS = 2  # Hill coefficient
psiminS = -6  # Minimum net growth rate for sensitive bacteria

# Antibiotic concentration from the first script
AB_concentration = 12.5

# MIC fractions and corresponding MIC values
MIC_R_fractions = [0.5, 0.8, 1.1]
MIC_values = [fraction * AB_concentration for fraction in MIC_R_fractions]  # [6.25, 10, 13.75]

# Persister levels from the first script
pers_levels = [0.00005, 0.8]
pers_level_labels = {0.00005: "Low persistence level", 0.8: "High persistence level"}

num_simulations = 10  # Number of simulations to run for each case

# Antibiotic intervals: 5 hours of treatment, rest is regrowth
intervals = [(0, 5, 12.5), (5, tfin, 0)]  # (start_time, end_time, antibiotic_concentration)

def RUN_tau_leaping(tfin, intervals, pers_level, MIC_S, MIC_R, initial_populations, tau_max=0.05, kappaS=2, psiminS=-6):
    # Constants (redefining inside the function for clarity)
    t0 = 0
    bS = 2
    cS = 1
    K = 1e8
    bP = 0.0
    dP = 0.0

    psimaxS = bS - dS
    psimaxSP = 0.05
    psimaxR = bS - dS
    psimaxPR = 0.05

    psiminSP = -0.5
    psiminPR = -0.5
    dS = 0.5

    np.random.seed(None)  # Ensure different seed for each process

    # Initial populations
    S = 1e7
    PS = 0  # Initialize PS to 0
    S = max(S, 0)
    R = 100
    PR = 0  # Initialize PR to 0
    R = max(R, 0)
    t = t0

    # MIC values
    micS = 2
    micR = MIC_R

    # Prepare time series data
    time_list = []
    S_list = []
    PS_list = []
    R_list = []
    PR_list = []

    treatment_active = False
    current_interval_index = 0
    n_intervals = len(intervals)

    while t <= tfin and (S + PS + R + PR > 0):
        # Determine the current interval and antibiotic concentration
        while current_interval_index < n_intervals and t >= intervals[current_interval_index][1]:
            current_interval_index += 1
        if current_interval_index < n_intervals:
            current_interval = intervals[current_interval_index]
            c_current = current_interval[2]
        else:
            c_current = 0

        # Handle treatment activation and persister dynamics
        if c_current > 0 and not treatment_active:
            treatment_active = True
            PS = int(pers_level * S)
            S -= PS
            S = max(S, 0)

            PR = int(pers_level * R)
            R -= PR
            R = max(R, 0)

        elif c_current == 0 and treatment_active:
            treatment_active = False
            S += PS
            PS = 0
            R += PR
            PR = 0

        if current_interval_index < n_intervals:
            period_end = intervals[current_interval_index][1]
        else:
            period_end = tfin

        tau = min(tau_max, period_end - t, tfin - t)
        if tau <= 0:
            tau = 1e-6

        # Calculate total population
        total_population = S + PS + R + PR

        # Calculate aS and aSP for sensitive cells
        if c_current == 0:
            aS = 0
            aSP = 0
        else:
            aS = (psimaxS - psiminS) * (c_current / micS) ** kappaS / \
                 ((c_current / micS) ** kappaS - psiminS / psimaxS)
            aSP = (psimaxSP - psiminSP) * (c_current / micS) ** kappaS / \
                  ((c_current / micS) ** kappaS - psiminSP / psimaxSP)

        # Net growth rates for S and PS
        birth_S_rate_per_capita = max(0, bS - cS * total_population / K)
        death_S_rate_per_capita = max(0, dS + aS)

        birth_PS_rate_per_capita = max(0, bP - cS * total_population / K)
        death_PS_rate_per_capita = max(0, dP + aSP)

        # Rates for S and PS
        birth_S_rate = max(0, birth_S_rate_per_capita * S)
        death_S_rate = max(0, death_S_rate_per_capita * S)
        birth_PS_rate = max(0, birth_PS_rate_per_capita * PS)
        death_PS_rate = max(0, death_PS_rate_per_capita * PS)

        # Simulate events for S and PS
        births_S = np.random.poisson(birth_S_rate * tau)
        deaths_S = np.random.poisson(death_S_rate * tau)
        births_PS = np.random.poisson(birth_PS_rate * tau)
        deaths_PS = np.random.poisson(death_PS_rate * tau)

        S += births_S - deaths_S
        S = max(S, 0)
        PS += births_PS - deaths_PS
        PS = max(PS, 0)

        # Calculate aR and aPR for resistant cells
        if c_current == 0:
            aR = 0
            aPR = 0
        else:
            aR = (psimaxR - psiminS) * (c_current / micR) ** kappaS / \
                 ((c_current / micR) ** kappaS - psiminS / psimaxR)
            aPR = (psimaxPR - psiminPR) * (c_current / micR) ** kappaS / \
                  ((c_current / micR) ** kappaS - psiminPR / psimaxPR)

        # Net growth rates for R and PR
        birth_R_rate_per_capita = max(0, bS - cS * total_population / K)
        death_R_rate_per_capita = max(0, dS + aR)

        birth_PR_rate_per_capita = max(0, bP - cS * total_population / K)
        death_PR_rate_per_capita = max(0, dP + aPR)

        # Rates for R and PR
        birth_R_rate = max(0, birth_R_rate_per_capita * R)
        death_R_rate = max(0, death_R_rate_per_capita * R)
        birth_PR_rate = max(0, birth_PR_rate_per_capita * PR)
        death_PR_rate = max(0, death_PR_rate_per_capita * PR)

        # Simulate events for R and PR
        births_R = np.random.poisson(birth_R_rate * tau)
        deaths_R = np.random.poisson(death_R_rate * tau)
        births_PR = np.random.poisson(birth_PR_rate * tau)
        deaths_PR = np.random.poisson(death_PR_rate * tau)

        R += births_R - deaths_R
        R = max(R, 0)
        PR += births_PR - deaths_PR
        PR = max(PR, 0)

        # Update time
        t += tau

        # Append current time and populations to lists
        time_list.append(t)
        S_list.append(S)
        PS_list.append(PS)
        R_list.append(R)
        PR_list.append(PR)

    # Return results and time series data
    return (S_list, PS_list, R_list, PR_list, time_list)

# Initialize list to store all simulation data for CSV export
csv_data = []

# Main simulation loop
colors = {0.00005: 'green', 0.8: 'darkblue'}  # Colors matching the first script

# Create a figure with subplots stacked vertically
fig, axs = plt.subplots(nrows=len(MIC_values), ncols=1, figsize=(4, 1.8 * len(MIC_values)), dpi=300, sharex=True)

if len(MIC_values) == 1:
    axs = [axs]  # Ensure axs is iterable when there's only one MIC value

# Create a patch for the antibiotic legend
antibiotic_patch = mpatches.Patch(facecolor='red', alpha=0.1, label='Antibiotic treatment')

# Initialize legend handles and labels
legend_handles = []
legend_labels = []

for idx, MIC in enumerate(MIC_values):
    ax = axs[idx]
    
    # Mark the antibiotic presence from hour 0 to 5
    ax.axvspan(0, 5, facecolor='red', alpha=0.1)
    
    for pers_level in pers_levels:
        # Use flags to add legend entries only once per persister level and type
        resistant_label_added = False
        wildtype_label_added = False
        
        for sim in range(num_simulations):
            initial_populations = {'S': 0, 'PS': 0, 'R': 100, 'PR': int(100 * pers_level)}
            S_list, PS_list, R_list, PR_list, time_list = RUN_tau_leaping(
                tfin, intervals, pers_level, MIC, MIC, initial_populations, tau_max, kappaS, psiminS
            )
            # Total resistant populations
            total_R = np.array(R_list)
            total_PR = np.array(PR_list)
            
            # Store time series data for CSV export
            for t_idx, time_point in enumerate(time_list):
                csv_data.append({
                    'MIC_fraction': MIC_R_fractions[idx],
                    'MIC_value': MIC,
                    'persister_level': pers_level,
                    'persister_level_label': pers_level_labels[pers_level],
                    'simulation_number': sim + 1,
                    'time_hours': time_point,
                    'S_population': S_list[t_idx],
                    'PS_population': PS_list[t_idx],
                    'R_population': R_list[t_idx],
                    'PR_population': PR_list[t_idx],
                    'total_sensitive': S_list[t_idx] + PS_list[t_idx],
                    'total_resistant': R_list[t_idx] + PR_list[t_idx],
                    'total_population': S_list[t_idx] + PS_list[t_idx] + R_list[t_idx] + PR_list[t_idx]
                })
            
            # Plot resistant cells on the current axis
            if not resistant_label_added and idx == 0:
                # Only add to legend for the first MIC value to avoid duplicates
                resistant_line = ax.plot(time_list, total_R + total_PR, color=colors[pers_level], 
                          linestyle='-', alpha=0.4, linewidth=1)[0]
                if pers_level == 0.00005:
                    legend_handles.append(resistant_line)
                    legend_labels.append(f"{pers_level_labels[pers_level]} (MIC$_m$)")
                else:
                    legend_handles.append(resistant_line)
                    legend_labels.append(f"{pers_level_labels[pers_level]} (MIC$_m$)")
                resistant_label_added = True
            else:
                ax.plot(time_list, total_R + total_PR, color=colors[pers_level], linestyle='-', alpha=0.4, linewidth=1)
            
            # For the first simulation only, plot S+PS (background population) as dashed line
            if sim == 0:
                total_S = np.array(S_list)
                total_PS = np.array(PS_list)
                wildtype_line = ax.plot(time_list, total_S + total_PS, color=colors[pers_level], 
                                      linestyle='--', alpha=0.2, linewidth=1.4)[0]
                
                # Add wildtype to legend (only for the first MIC value to avoid duplicates)
                if not wildtype_label_added and idx == 0:
                    legend_handles.append(wildtype_line)
                    legend_labels.append(f"{pers_level_labels[pers_level]} (MIC$_{{wt}}$)")
                    wildtype_label_added = True
    
    ax.set_ylabel('Lineage Size')
    ax.set_yscale('log')  # Logarithmic scale for y-axis
    ax.set_ylim(1, 2e8)  # Limit y-axis to 1 to 2e8
    ax.set_xlim(0, tfin)  # Limit x-axis to 0 to tfin

    # Add MIC_R_fraction annotation to the top-left corner of each subplot
    ax.text(0.05, 0.95, f'{MIC_R_fractions[idx]}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top')

# Set xlabel for the bottom subplot
axs[-1].set_xlabel('Time (hours)')

# Add entry for antibiotic treatment
legend_handles.append(antibiotic_patch)
legend_labels.append('Antibiotic treatment')

# Add a single legend on the right side
plt.tight_layout()
fig.legend(legend_handles, legend_labels, loc='center right', bbox_to_anchor=(1.48, 0.5))

# Adjust the figure to make room for the legend on the right
plt.subplots_adjust(right=0.78)  # Make space at the right for the legend

# Data is now available in the consolidated Excel file

plt.savefig('Fig_1_c-d-e.png', dpi=300, bbox_inches='tight')
plt.savefig('Fig_1_c-d-e.pdf', format='pdf', bbox_inches='tight')
# plt.show()