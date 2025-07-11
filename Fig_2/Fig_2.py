import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.colors import BoundaryNorm
from scipy.stats import lognorm
import matplotlib as mpl
from scipy.optimize import root_scalar
import pandas as pd
import os

# save plot same directory as this script
os.chdir(os.path.dirname(__file__))

# Set up matplotlib for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

# Colormap for probability of establishment
cmap_P_est = 'coolwarm'

# Constants and functions setup
MIC_wt = 2                # Wild type MIC (used as a reference unit)
k = 2                    # Hill coefficient
psi_max = 1               # Maximum growth rate
N_max = 2e8               # Final (carrying capacity) population size
tau = 5 # Fixed time in hours
Psi_min_sensitive = -6    # Minimum growth rate for sensitive bacteria
Psi_min_tolerant_wt = 0   # Minimum growth rate for tolerant bacteria (wild-type MIC)
Psi_min_tolerant_resistant = 0  # Minimum growth rate for tolerant bacteria (resistant MIC)

# Parameters for the log-normal distribution
mu_log_normal = 0           # Mean of the associated normal distribution
sigma_log_normal = 0.7      # Standard deviation of the associated normal distribution
total_mutation_probability = 1e-6  # Total mutation probability for any MIC

# Plotting parameters
alpha_values = np.logspace(-5, 0, 100)
fixed_concentrations = [12.5, 25]         # Antibiotic concentrations as multiples of MIC_wt
MIC_R_ratio_range = np.linspace(0.1, 2, 100)  # Common x-axis: resistance mutant MIC/MIC_wt

# Initialize list to store all data for CSV export
csv_data = []

# Create a figure with proper spacing
fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=600)

# Define log levels for probability plots - starting from 10^-5
log_levels = np.logspace(-5, 0.0, 30)  # Levels from 1e-5 to 1 with fewer divisions
norm_prob = BoundaryNorm(log_levels, ncolors=plt.get_cmap(cmap_P_est).N, clip=True)  # clip=True to enforce bounds


def solve_exponential_equation(A, a, B, b, C, t_bounds=(1, 10)):
    """
    Solves A * exp(a*t) + B * exp(b*t) = C numerically for t.

    Parameters:
        A, a, B, b, C: Coefficients in the equation
        t_bounds: Tuple (t_min, t_max) for bracketing the root

    Returns:
        root: Root result object from scipy.optimize
    """
    def f(t):
        return A * np.exp(a * t) + B * np.exp(b * t) - C

    # Try to find a root in the interval
    result = root_scalar(f, bracket=t_bounds, method='brentq')
    return result.root

# Function definitions
def psi(c, MIC, psi_max, k, Psi_min):
    if c <= 0 or MIC <= 0:
        return psi_max
    denominator = (Psi_min / psi_max) - (c / MIC) ** k
    if denominator == 0:
        return psi_max
    return Psi_min * (1 - (c / MIC) ** k) / denominator

def survival(c, tau, MIC, psi_max, k, Psi_min):
    psi_t = psi(c, MIC, psi_max, k, Psi_min)
    return np.exp(psi_t * tau)

def calculate_P_est_product(c, MIC_R, psi_max, k, N_max, alpha):
    # Compute survival for wild-type (sensitive and tolerant)
    S_t = survival(c, tau, MIC_wt, psi_max, k, Psi_min_tolerant_wt)
    S_t = min(S_t, 1)  # Cap at 1 - added adjustment
    
    
    S_w = survival(c, tau, MIC_wt, psi_max, k, Psi_min_sensitive)
    S_w = min(S_w, 1)  # Cap at 1 - added adjustment
    
    S_tot = (1 - alpha) * S_w + alpha * S_t
    S_tot = min(S_tot, 1)  # Cap at 1 - added adjustment
    S_tot = max(S_tot, 1e-10)  # Avoid zero or negative values

    # Compute survival for resistant mutants
    S_r = survival(c, tau, MIC_R, psi_max, k, Psi_min_sensitive)
    
    Sr_tol = survival(c, tau, MIC_R, psi_max, k, Psi_min_tolerant_resistant)
    Sr_tol = min(Sr_tol, 1)  # Cap at 1 - added adjustment
    
    S_rt = (1 - alpha) * S_r + alpha * Sr_tol
    S_rt = min(S_rt, 1)  # Cap at 1 - added adjustment
    S_rt = max(S_rt, 1e-10)  # Avoid zero or negative values

    # Compute mu_mod using the log-normal distribution
    if MIC_R <= 0:
        mu_mod = 0
    else:
        mu_mod = total_mutation_probability * lognorm.pdf(MIC_R, s=sigma_log_normal, scale=np.exp(mu_log_normal))

    # Population after antibiotic exposure
    N_s = N_max * S_tot

    # Calculate growth parameters
    r_M = psi_max
    d_M = 0.2
    b_M = r_M + d_M

    r_wt = psi_max
    d_wt = 0.2
    b_wt = (r_wt + d_wt)/(1-total_mutation_probability)
    
    T_total = (1/r_wt)*np.log(N_max / N_s)
    
    # Integrate over time to calculate P_est
    def integrand(t):
        P_ext_bd = 1 - r_M / (b_M - d_M* np.exp(-r_M*(T_total - t)))
        return -mu_mod * b_wt * N_s * np.exp(r_wt * t) * S_rt * (1-P_ext_bd)
    
    try:
        log_P_no_est, _ = quad(integrand, 0, T_total)
    except Exception:
        return 0

    P_est = 1 - np.exp(log_P_no_est)
    # Only limit to maximum of 1, don't enforce minimum
    return min(P_est, 1)

# Loop over fixed antibiotic concentrations
for idx, AB_concentration in enumerate(fixed_concentrations):
    MIC_R = MIC_R_ratio_range * AB_concentration
    MIC_R_ratio = MIC_R_ratio_range

    # Initialize surface for P_est
    P_est_product_surface = np.zeros((len(MIC_R_ratio), len(alpha_values)))

    # For each combination of persister fraction and MIC_R value
    for i, alpha in enumerate(alpha_values):
        for j, MIC_R_value in enumerate(MIC_R):
            P_est_value = calculate_P_est_product(AB_concentration, MIC_R_value, psi_max, k, N_max, alpha)
            P_est_product_surface[j, i] = P_est_value
            
            # Store data for CSV export
            csv_data.append({
                'antibiotic_concentration': AB_concentration,
                'antibiotic_concentration_ug_ml': AB_concentration/2,  # Convert to μg/mL
                'MIC_R_ratio': MIC_R_ratio_range[j],
                'MIC_R_value': MIC_R_value,
                'persistence_level_alpha': alpha,
                'probability_of_emergence': P_est_value
            })

    # Plot P_est - values below 10^-5 will naturally appear white
    cax_product = axes[idx].contourf(
        MIC_R_ratio, alpha_values, P_est_product_surface.T,
        levels=log_levels, cmap=cmap_P_est, norm=norm_prob, alpha=0.9
    )
    axes[idx].contour(
        MIC_R_ratio, alpha_values, P_est_product_surface.T,
        levels=log_levels, colors='k', linewidths=0.43
    )
    # Create the main title and units separately with different font sizes
    main_title = f'C = {AB_concentration/2} × $\\mathrm{{MIC}}_{{\\mathrm{{wt}}}}$'
    axes[idx].set_title(main_title, fontsize=12)
    # Add units as a smaller annotation
    axes[idx].text(0.88, 1.03, '(μg/mL)', transform=axes[idx].transAxes, 
             horizontalalignment='center', fontsize=9)
    
    axes[idx].set_ylabel('Persistence level (α)', fontsize=11)
    axes[idx].set_xlabel(r'$\frac{\mathrm{MIC}_{\text{m}}}{C}$', fontsize=14)
    axes[idx].tick_params(axis='both', which='major', labelsize=10)
    axes[idx].set_yscale('log')
    axes[idx].axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[idx].set_xlim(0.2, 2)
    axes[idx].set_ylim(1e-5, 1.01)
    #axes[idx].grid(alpha=0.3, linestyle='--')

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

# Create a separate ScalarMappable with BoundaryNorm to show bins
cmap = plt.get_cmap(cmap_P_est)
# Use BoundaryNorm to get the color bins
sm = mpl.cm.ScalarMappable(cmap=cmap,
                          norm=BoundaryNorm(log_levels, ncolors=plt.get_cmap(cmap_P_est).N, clip=False))
sm.set_array([])  # Empty array - we're manually setting values

# Create a new colorbar using this mappable, without any automatic ticks initially
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[])
cbar.set_label('Probability of emergence  $P_{e}$', fontsize=12)

# Use the exact values that were previously designed for better visual alignment
# These labels use LaTeX for scientific notation.
manual_ticks = [1e-5, 1.065*1e-4, 1.15*1e-3, 1.25*1e-2, 1.35*1e-1, 1]
tick_labels = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$']

# Add the tick marks and labels at the specified positions
cbar.ax.yaxis.set_ticks(manual_ticks)
cbar.ax.set_yticklabels(tick_labels)

# Adjust tick appearance parameters
cbar.ax.tick_params(labelsize=8, length=5.5, width=0.8, direction='out')

# Adjust layout
plt.tight_layout(rect=[0, 0.02, 0.9, 0.97])

# Data is now available in the consolidated Excel file

# Save the figure with high resolution
plt.savefig('Fig_2.png', dpi=600, bbox_inches='tight')
plt.savefig("Fig_2.pdf", format="pdf", bbox_inches="tight")
plt.close()