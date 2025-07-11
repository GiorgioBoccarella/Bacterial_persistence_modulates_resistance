import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.colors import BoundaryNorm
from scipy.stats import lognorm, expon
import matplotlib as mpl
from scipy.optimize import root_scalar
import os

os.chdir(os.path.dirname(__file__))
# save plot same directory as this script
os.makedirs('plots', exist_ok=True)
plot_path = 'plots' 

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
MIC_wt = 1                # Wild type MIC (used as a reference unit)
k = 2                    # Hill coefficient
psi_max = 1               # Maximum growth rate
N_max = 2*1e8               # Final (carrying capacity) population size
tau = 5# Fixed time in hours
Psi_min_sensitive = -6    # Minimum growth rate for sensitive bacteria
Psi_min_tolerant_wt = 0   # Minimum growth rate for tolerant bacteria (wild-type MIC)
Psi_min_tolerant_resistant = 0  # Minimum growth rate for tolerant bacteria (resistant MIC)

# Parameters for the log-normal distribution (original, will be one of the cases)
# mu_log_normal = 0           # Mean of the associated normal distribution
# sigma_log_normal = 0.7      # Standard deviation of the associated normal distribution
total_mutation_probability = 1e-6  # Total mutation probability for any MIC

# Plotting parameters
alpha_values = np.logspace(-5, 0, 100)
fixed_concentrations = [12.5, 25]         # Antibiotic concentrations as multiples of MIC_wt
MIC_R_ratio_range = np.linspace(0.1, 2, 100)  # Common x-axis: resistance mutant MIC/MIC_wt

# Define distribution configurations
distribution_configs = [
    {'type': 'lognormal', 'params': {'mu': 0, 'sigma': 0.5}, 'title_desc': 'Log-Normal (μ=0, σ=0.5)'},
    {'type': 'lognormal', 'params': {'mu': 0, 'sigma': 0.9}, 'title_desc': 'Log-Normal (μ=0, σ=0.9)'},
    {'type': 'exponential', 'params': {'lambda': 0.9}, 'title_desc': 'Exponential (λ=0.9)'},
    {'type': 'exponential', 'params': {'lambda': 0.5}, 'title_desc': 'Exponential (λ=0.5)'}
]

# Swap the first two distribution configurations for row swapping
if len(distribution_configs) >= 2:
    distribution_configs[0], distribution_configs[1] = distribution_configs[1], distribution_configs[0]

# Create a figure with proper spacing for 4x2 plots
# fig, axes = plt.subplots(len(distribution_configs), 2, figsize=(12, 16), dpi=600) # Adjusted for 4 rows # OLD
num_main_rows = len(distribution_configs)
# Increase figure height to accommodate the new distribution plot row (e.g., by 4 inches)
fig = plt.figure(figsize=(12, 16 + 4), dpi=600) 
first_p_est_ax_ref = None # To store reference to the first P_est plot axis (top-left)
last_p_est_ax_ref = None  # To store reference to a P_est plot axis in the last P_est row (bottom-left)

# Define log levels for probability plots - starting from 10^-5
log_levels = np.logspace(-5, 0, 30)  # Levels from 1e-5 to 1 with fewer divisions
norm_prob = BoundaryNorm(log_levels, ncolors=plt.get_cmap(cmap_P_est).N, clip=True)  # clip=True to enforce bounds

def solve_exponential_equation(A, a, B, b, C, t_bounds=(0, 100)):
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

def calculate_P_est_product(c, MIC_R, psi_max, k, N_max, alpha, distribution_config):
    # Compute survival for wild-type (sensitive and tolerant)
    S_t = survival(c, tau, MIC_wt, psi_max, k, Psi_min_tolerant_wt)
    S_t = min(S_t, 1)  # Cap at 1
    
    S_w = survival(c, tau, MIC_wt, psi_max, k, Psi_min_sensitive)
    S_w = min(S_w, 1)  # Cap at 1
    
    S_tot = (1 - alpha) * S_w + alpha * S_t
    S_tot = min(S_tot, 1)  # Cap at 1
    S_tot = max(S_tot, 1e-10)  # Avoid zero or negative values

    # Compute survival for resistant mutants
    S_r = survival(c, tau, MIC_R, psi_max, k, Psi_min_sensitive)
    
    Sr_tol = survival(c, tau, MIC_R, psi_max, k, Psi_min_tolerant_resistant)
    Sr_tol = min(Sr_tol, 1)  # Cap at 1
    
    S_rt = (1 - alpha) * S_r + alpha * Sr_tol
    S_rt = min(S_rt, 1)  # Cap at 1
    S_rt = max(S_rt, 1e-10)  # Avoid zero or negative values

    # Compute mu_mod using the specified distribution
    if MIC_R <= 0:
        mu_mod = 0
    else:
        dist_type = distribution_config['type']
        params = distribution_config['params']
        if dist_type == 'lognormal':
            mu_log_normal_dist = params['mu']
            sigma_log_normal_dist = params['sigma']
            mu_mod = total_mutation_probability * lognorm.pdf(MIC_R, s=sigma_log_normal_dist, scale=np.exp(mu_log_normal_dist))
        elif dist_type == 'exponential':
            lambda_val = params['lambda']
            # For scipy.stats.expon, scale = 1/lambda
            mu_mod = total_mutation_probability * expon.pdf(MIC_R, scale=1.0/lambda_val)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    # Population after antibiotic exposure
    N_s = N_max * S_tot

    # Calculate growth parameters
    r_M = psi_max
    d_M = 0.1
    b_M = r_M + d_M

    r_wt = psi_max
    d_wt = 0.1
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

# Define subplot labels for the 4x2 grid plus 2 distribution plots
subplot_labels = [
    ['a)', 'b)'],  # Row 0
    ['c)', 'd)'],  # Row 1  
    ['e)', 'f)'],  # Row 2
    ['g)', 'h)'],  # Row 3
]
dist_labels = ['i)', 'j)']  # For the distribution plots

# Loop over distribution configurations (rows) and fixed antibiotic concentrations (columns)
for row_idx, dist_config in enumerate(distribution_configs):
    for col_idx, AB_concentration in enumerate(fixed_concentrations):
        # ax = axes[row_idx, col_idx] # OLD
        ax = plt.subplot2grid((num_main_rows + 1, 2), (row_idx, col_idx))
        if col_idx == 0: # Capture references for colorbar positioning
            if row_idx == 0:
                first_p_est_ax_ref = ax
            if row_idx == num_main_rows - 1: # This will be the axis in the first column of the last P_est row
                last_p_est_ax_ref = ax
        
        # Initialize surface for P_est
        P_est_product_surface = np.zeros((len(MIC_R_ratio_range), len(alpha_values)))

        # For each combination of persister fraction and MIC_R_ratio value
        for i, alpha in enumerate(alpha_values):
            for j, mic_r_ratio in enumerate(MIC_R_ratio_range):
                actual_mic_r = mic_r_ratio * AB_concentration # Calculate actual MIC_R for the calculation
                P_est_product_surface[j, i] = calculate_P_est_product(
                    AB_concentration, actual_mic_r, psi_max, k, N_max, alpha, dist_config
                )

        # Plot P_est - values below 10^-5 will naturally appear white
        cax_product = ax.contourf(
            MIC_R_ratio_range, alpha_values, P_est_product_surface.T,
            levels=log_levels, cmap=cmap_P_est, norm=norm_prob, alpha=0.9
        )
        ax.contour(
            MIC_R_ratio_range, alpha_values, P_est_product_surface.T,
            levels=log_levels, colors='k', linewidths=0.4
        )
        
        # Create the main title and units
        concentration_title_part = f'C = {AB_concentration/2} × $\\mathrm{{MIC}}_{{\\mathrm{{wt}}}}$'
        full_title = f'{dist_config["title_desc"]}\n{concentration_title_part}'
        ax.set_title(full_title, fontsize=10) # Adjusted fontsize
        
        # Add subplot label
        subplot_label = subplot_labels[row_idx][col_idx]
        ax.text(0.02, 1.02, subplot_label, transform=ax.transAxes, fontsize=14, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add units as a smaller annotation - position relative to axes
        ax.text(0.95, 1.01, '', transform=ax.transAxes, 
                 horizontalalignment='right', verticalalignment='bottom', fontsize=8) # Adjusted position and alignment
        
        if col_idx == 0:
            ax.set_ylabel('Persistence level (α)', fontsize=11)
        if row_idx == len(distribution_configs) - 1:
            ax.set_xlabel(r'$\frac{\mathrm{MIC}_{\text{m}}}{C}$', fontsize=14)
            
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_yscale('log')
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlim(0.2, 2)
        ax.set_ylim(1e-5, 1)
        #ax.grid(alpha=0.3, linestyle='--') # Optional grid

# --- Add new plot for MIC distributions (integrated into the main figure) ---
# ax_dist = plt.subplot2grid((num_main_rows + 1, 2), (num_main_rows, 0), colspan=2) # OLD - single plot

ax_dist_lognormal = plt.subplot2grid((num_main_rows + 1, 2), (num_main_rows, 0))
ax_dist_exp = plt.subplot2grid((num_main_rows + 1, 2), (num_main_rows, 1))

mic_plot_range = np.linspace(0.1, 32, 500) # Linear scale up to 32, ensure start matches xlim
colors = ['blue', 'green', 'red', 'purple'] # Colors for different distributions

# Plot Log-Normal Distributions
lognormal_color_idx = 0
for i, dist_config_item in enumerate(distribution_configs):
    dist_type = dist_config_item['type']
    if dist_type == 'lognormal':
        params = dist_config_item['params']
        label = dist_config_item['title_desc']
        # Use a consistent color from the 'colors' list for lognormal plots
        color = colors[lognormal_color_idx % len(colors)] 
        lognormal_color_idx +=1

        mu_log_normal_dist = params['mu']
        sigma_log_normal_dist = params['sigma']
        pdf_values = lognorm.pdf(mic_plot_range, s=sigma_log_normal_dist, scale=np.exp(mu_log_normal_dist))
        ax_dist_lognormal.plot(mic_plot_range, pdf_values, label=label, color=color, linewidth=2)

ax_dist_lognormal.set_xlabel(r'$\mathrm{MIC}_{\mathrm{m}}$', fontsize=12)
ax_dist_lognormal.set_ylabel('Probability Density', fontsize=12)
ax_dist_lognormal.set_title('Log-Normal MIC Distributions', fontsize=14)
ax_dist_lognormal.legend(fontsize=10)
ax_dist_lognormal.grid(True, linestyle='--', alpha=0.7)
ax_dist_lognormal.set_xscale('linear')
ax_dist_lognormal.set_xlim(0.1, 32)
ax_dist_lognormal.set_yscale('log')
ax_dist_lognormal.set_ylim(bottom=1e-14)

# Add subplot label for log-normal distribution plot
ax_dist_lognormal.text(0.02, 1.02, dist_labels[0], transform=ax_dist_lognormal.transAxes, 
                      fontsize=14, fontweight='bold', verticalalignment='bottom', horizontalalignment='left',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Plot Exponential Distributions
exp_color_idx = 0 # Separate color index for exponential plots if needed, or continue from lognormal
for i, dist_config_item in enumerate(distribution_configs):
    dist_type = dist_config_item['type']
    if dist_type == 'exponential':
        params = dist_config_item['params']
        label = dist_config_item['title_desc']
        # Use a consistent color from the 'colors' list, possibly offset or a different part of the list
        color = colors[(lognormal_color_idx + exp_color_idx) % len(colors)] 
        exp_color_idx +=1
        
        lambda_val = params['lambda']
        pdf_values = expon.pdf(mic_plot_range, scale=1.0/lambda_val)
        ax_dist_exp.plot(mic_plot_range, pdf_values, label=label, color=color, linewidth=2)

ax_dist_exp.set_xlabel(r'$\mathrm{MIC}_{\mathrm{m}}$', fontsize=12)
ax_dist_exp.set_ylabel('Probability Density', fontsize=12) # Can be omitted if y-axes are shared and aligned
ax_dist_exp.set_title('Exponential MIC Distributions', fontsize=14)
ax_dist_exp.legend(fontsize=10)
ax_dist_exp.grid(True, linestyle='--', alpha=0.7)
ax_dist_exp.set_xscale('linear')
ax_dist_exp.set_xlim(0.1, 32)
ax_dist_exp.set_yscale('log')
ax_dist_exp.set_ylim(bottom=1e-14)

# Add subplot label for exponential distribution plot
ax_dist_exp.text(0.02, 1.02, dist_labels[1], transform=ax_dist_exp.transAxes, 
                fontsize=14, fontweight='bold', verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# If y-axes are truly shared, could do: ax_dist_exp.tick_params(labelleft=False)

# OLD plotting loop - to be replaced by the two specific loops above
# for i, dist_config_item in enumerate(distribution_configs): # Renamed dist_config to avoid clash
#     dist_type = dist_config_item['type']
#     params = dist_config_item['params']
#     label = dist_config_item['title_desc']
#     color = colors[i % len(colors)]
# 
#     if dist_type == 'lognormal':
#         mu_log_normal_dist = params['mu']
#         sigma_log_normal_dist = params['sigma']
#         pdf_values = lognorm.pdf(mic_plot_range, s=sigma_log_normal_dist, scale=np.exp(mu_log_normal_dist))
#     elif dist_type == 'exponential':
#         lambda_val = params['lambda']
#         pdf_values = expon.pdf(mic_plot_range, scale=1.0/lambda_val)
#     else:
#         continue # Skip unknown distribution types
# 
#     ax_dist.plot(mic_plot_range, pdf_values, label=label, color=color, linewidth=2)
# 
# ax_dist.set_xlabel('MIC (relative to MIC_wt)', fontsize=12)
# ax_dist.set_ylabel('Probability Density', fontsize=12)
# ax_dist.set_title('MIC Distribution Functions', fontsize=14)
# ax_dist.legend(fontsize=10)
# ax_dist.grid(True, linestyle='--', alpha=0.7)
# # ax_dist.set_xscale('log') # OLD Log scale for x-axis
# ax_dist.set_xscale('linear') # Linear scale for x-axis
# ax_dist.set_xlim(0.1, 32) # X-axis limit up to 32
# ax_dist.set_yscale('log') # Log scale for y-axis
# # ax_dist.set_ylim(bottom=1e-3) # OLD Set a small positive lower limit for log y-axis
# ax_dist.set_ylim(bottom=1e-14) # Set y-axis lower limit to 1e-14


# Adjust layout of main subplots first
# The rect=[left, bottom, right, top] leaves space for titles and the colorbar on the right.
plt.tight_layout(rect=[0, 0.02, 0.88, 0.97]) # Using original rect, tight_layout should adjust for new content

# Get position of an axis in the first row to align the colorbar.
# This needs to happen after tight_layout has arranged the subplots.
# first_row_ax_bbox = axes[0,0].get_position() # OLD
if first_p_est_ax_ref is not None and last_p_est_ax_ref is not None:
    top_ax_bbox = first_p_est_ax_ref.get_position() # Bbox of the top-left P_est plot
    # bottom_ax_bbox = last_p_est_ax_ref.get_position() # Bbox of the bottom-left P_est plot (in the P_est block) # OLD

    # Define colorbar axes: to the right of plots, aligned vertically with the P_est plots block
    cbar_ax_left = 0.90  # Start colorbar at 90% of figure width
    cbar_ax_width = 0.02 # Width of the colorbar
    # cbar_ax_bottom = bottom_ax_bbox.y0 # Align with bottom of the last P_est plot row # OLD
    # cbar_ax_height = (top_ax_bbox.y0 + top_ax_bbox.height) - bottom_ax_bbox.y0 # Height spans all P_est plot rows # OLD
    
    cbar_ax_bottom = top_ax_bbox.y0 # Align with bottom of the first P_est plot
    cbar_ax_height = top_ax_bbox.height # Height matches the first P_est plot

    cbar_ax = fig.add_axes([cbar_ax_left, cbar_ax_bottom, cbar_ax_width, cbar_ax_height])

    # Create a separate ScalarMappable with BoundaryNorm to show bins
    # import matplotlib as mpl # Already imported at the top
    cmap = plt.get_cmap(cmap_P_est)
    # Use BoundaryNorm to get the color bins
    sm = mpl.cm.ScalarMappable(cmap=cmap, 
                              norm=BoundaryNorm(log_levels, ncolors=plt.get_cmap(cmap_P_est).N, clip=True))
    sm.set_array([])  # Empty array - we're manually setting values

    # Create a new colorbar using this mappable, without any automatic ticks
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[])
    cbar.set_label('Probability of emergence  $P_{e}$', fontsize=12)

    # Use the exact values from log_levels that match our desired tick positions
    # This ensures ticks align perfectly with bin boundaries
    manual_ticks = [1e-5, 1.065*1e-4, 1.15*1e-3, 1.25*1e-2, 1.35*1e-1, 1]
    tick_labels = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$']

    # Add the tick marks and labels at the exact bin boundaries
    cbar.ax.yaxis.set_ticks(manual_ticks)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=8, length=5.5, width=0.8, direction='out')  # Longer, visible tick marks
else:
    print("Warning: Could not determine P_est axes bounds for colorbar positioning.")

# Save the figure with high resolution
plt.savefig(os.path.join(plot_path,'S3_figure.png'), dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(plot_path,"S3_figure.pdf"), format="pdf", bbox_inches="tight")
plt.close()

# --- Add new plot for MIC distributions --- # This whole block will be deleted by context
# fig_dist, ax_dist = plt.subplots(figsize=(8, 6), dpi=300)
# 
# mic_plot_range = np.linspace(1e-2, 5, 500) # Define a suitable range for MIC values to plot
# colors = ['blue', 'green', 'red', 'purple'] # Colors for different distributions
# 
# for i, dist_config in enumerate(distribution_configs):
#     dist_type = dist_config['type']
#     params = dist_config['params']
#     label = dist_config['title_desc']
#     color = colors[i % len(colors)]
# 
#     if dist_type == 'lognormal':
#         mu_log_normal_dist = params['mu']
#         sigma_log_normal_dist = params['sigma']
#         pdf_values = lognorm.pdf(mic_plot_range, s=sigma_log_normal_dist, scale=np.exp(mu_log_normal_dist))
#     elif dist_type == 'exponential':
#         lambda_val = params['lambda']
#         pdf_values = expon.pdf(mic_plot_range, scale=1.0/lambda_val)
#     else:
#         continue # Skip unknown distribution types
# 
#     ax_dist.plot(mic_plot_range, pdf_values, label=label, color=color, linewidth=2)
# 
# ax_dist.set_xlabel('MIC (relative to MIC_wt)', fontsize=12)
# ax_dist.set_ylabel('Probability Density', fontsize=12)
# ax_dist.set_title('MIC Distribution Functions', fontsize=14)
# ax_dist.legend(fontsize=10)
# ax_dist.grid(True, linestyle='--', alpha=0.7)
# ax_dist.set_xlim(left=0) # MIC cannot be negative
# ax_dist.set_ylim(bottom=0) # Probability density cannot be negative
# 
# plt.tight_layout()
# plt.savefig(os.path.join(plot_path, 'mic_distributions_plot.png'), dpi=300, bbox_inches='tight')
# plt.savefig(os.path.join(plot_path, 'mic_distributions_plot.pdf'), format="pdf", bbox_inches='tight')
# plt.close(fig_dist) # Close the new figure
# End of code to be deleted by matching the start and ensuring the rest is replaced or naturally falls off