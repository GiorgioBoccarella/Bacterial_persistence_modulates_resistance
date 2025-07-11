###############################################################################
# FULL SCRIPT: Two Plots Side-by-Side (Simulation vs. Empirical), 
# Aesthetically Matched for a Scientific Manuscript (e.g., Nature style).
#
# NOTE: Adjust file paths as needed for your local setup.
###############################################################################

# -----------------------------------------------------------------------------
# 1) Load Required Libraries
# -----------------------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggthemes)
library(scales)
library(cowplot)  # For arranging plots side-by-side
library(readxl)   # For reading Excel files

# -----------------------------------------------------------------------------
# 2) THEME DEFINITION (COMMON THEME FOR BOTH PLOTS)
#    We create a custom theme to ensure both plots match aesthetically.
# -----------------------------------------------------------------------------
theme_nature_style <- function() {
  theme(
    # Remove grid lines
    panel.grid.major  = element_blank(),
    panel.grid.minor  = element_blank(),
    
    # White background
    panel.background  = element_rect(fill = "white"),
    plot.background   = element_rect(fill = "white"),
    
    # Black border around subplots
    panel.border      = element_rect(color = "black", fill = NA, size = 1),
    
    # Increase text sizes and set color to black
    axis.text.x       = element_text(size = 14, color = "black", angle = 45, hjust = 1),
    axis.text.y       = element_text(size = 14, color = "black"),
    axis.title.x      = element_text(size = 16, color = "black"),
    axis.title.y      = element_text(size = 16, color = "black"),
    legend.title      = element_text(size = 14, color = "black"),
    legend.text       = element_text(size = 14, color = "black"),
    strip.background  = element_blank(),
    strip.text        = element_text(size = 14, color = "black"),
    
    # Legend placement
    legend.position   = "bottom"
  )
}

###############################################################################
# PART A: SIMULATION RESULTS (Density Plot)
###############################################################################

# -----------------------------------------------------------------------------
# 3) Read and preprocess simulation data
# -----------------------------------------------------------------------------
data <- read_excel("../complete_data/complete_data.xlsx", sheet = "Fig_4_a-b")

# Ensure 'initial_pers_level' is treated as a factor
data$initial_pers_level <- as.factor(data$initial_pers_level)

# -----------------------------------------------------------------------------
# 4) Filter data for the density plot
# -----------------------------------------------------------------------------
data_filtered <- data %>%
  filter(
    pers_cost == 0,
    initial_pers_level != "0.05",
    initial_pers_level != "0.005",
    initial_pers_level != "0.3",
    c %in% c(12.5, 25),
    kappaS == 2
  )

# Keep only those that ended up with MIC > 2
data_density <- data_filtered %>%
  filter(most_common_mic_day_12 > 2)

# Define color palette
cb_palette <- c("#008000", "#00008B")

# -----------------------------------------------------------------------------
# 5) Build the density plot
# -----------------------------------------------------------------------------

# First, convert `c` to factor
data_density$c_factor <- factor(data_density$c, levels = c(12.5, 25))

# Create a named vector for facet labels
facet_labels_sim <- c(
  "12.5" = "MIC (ug/ml) 6.25 x WT_mic",
  "25"   = "MIC (ug/ml) 12.5 x WT_mic"
)

p1 <- ggplot(
  data_density,
  aes(
    x = most_common_mic_day_12,
    color = initial_pers_level,
    fill  = initial_pers_level
  )
) +
  geom_density(alpha = 0.6, adjust = 1.5) +
  xlim(0, 40) +
  scale_fill_manual(values = cb_palette) +
  scale_color_manual(values = cb_palette) +
  labs(
    x     = "Most Common Mutant MIC",
    y     = "Density",
    fill  = "Initial Persister Fraction",
    color = "Initial Persister Fraction",
    title = "Simulation Results"
  ) +
  facet_wrap(
    ~ c_factor,
    nrow     = 2,
    labeller = as_labeller(facet_labels_sim)
  ) +
  theme_nature_style() +
  # Remove legend from the first plot
  theme(legend.position = "none")

###############################################################################
# PART B: EMPIRICAL DATA (Bar Chart)
###############################################################################

# -----------------------------------------------------------------------------
# 6) Read and preprocess empirical data
# -----------------------------------------------------------------------------
df <- read_excel("../complete_data/complete_data.xlsx", sheet = "Fig_4_a-b_empirical")

# -----------------------------------------------------------------------------
# 7) Filter to only AB_conc in {12.5,25}
# -----------------------------------------------------------------------------
df_filtered <- df %>%
  filter(
    AB_conc %in% c(12.5, 25)
  )

# -----------------------------------------------------------------------------
# 8) Merge MIC categories: 
#    If MIC < 2 => "2"
#    If MIC >= 32 => "32+"
#    Else keep as character
# -----------------------------------------------------------------------------
df_filtered <- df_filtered %>%
  mutate(
    MIC = case_when(
      MIC < 2    ~ "2",     # <2 merges into "2"
      MIC >= 32  ~ "32+",   # >=32 merges into "32+"
      TRUE       ~ as.character(MIC)
    )
  )

# -----------------------------------------------------------------------------
# 9) Sum pers_frac by (AB_conc, nutrient_conc, MIC)
# -----------------------------------------------------------------------------
df_summed <- df_filtered %>%
  group_by(AB_conc, nutrient_conc, MIC) %>%
  summarise(total_frac = sum(pers_frac), .groups = "drop")

# -----------------------------------------------------------------------------
# 10) Compute relative frequency within (AB_conc, nutrient_conc)
# -----------------------------------------------------------------------------
df_rel <- df_summed %>%
  group_by(AB_conc, nutrient_conc) %>%
  mutate(rel_freq = total_frac / sum(total_frac)) %>%
  ungroup()

# -----------------------------------------------------------------------------
# 11) Recode nutrient_conc => "High" (0.25) or "Low" (0.8)
# -----------------------------------------------------------------------------
df_rel <- df_rel %>%
  mutate(
    pers_label = case_when(
      nutrient_conc == 0.25 ~ "High",
      nutrient_conc == 0.8  ~ "Low",
      TRUE                  ~ NA_character_
    )
  )

# -----------------------------------------------------------------------------
# 12) Define final MIC order: 2, 4, 8, 16, 32+
# -----------------------------------------------------------------------------
mic_levels <- c("2", "4", "8", "16", "32+")
df_rel$MIC <- factor(df_rel$MIC, levels = mic_levels)

# -----------------------------------------------------------------------------
# 13) Complete all combos so missing ones have rel_freq=0
# -----------------------------------------------------------------------------
df_complete <- df_rel %>%
  complete(
    AB_conc,
    pers_label,
    MIC = mic_levels, 
    fill = list(total_frac = 0, rel_freq = 0)
  )

# -----------------------------------------------------------------------------
# 14) Plot the bar chart
# -----------------------------------------------------------------------------

# Convert AB_conc to factor
df_complete$AB_conc_factor <- factor(df_complete$AB_conc, levels = c(12.5, 25))

# Create a named vector for facet labels
facet_labels_emp <- c(
  "12.5" = "MIC (ug/ml) 6.25 x WT_mic",
  "25"   = "MIC (ug/ml) 12.5 x WT_mic"
)

p2 <- ggplot(df_complete, aes(x = MIC, y = rel_freq, fill = pers_label)) +
  geom_bar(
    stat     = "identity",
    position = position_dodge(width = 0.87),
    color    = "black",
    alpha    = 0.65
  ) +
  facet_wrap(
    ~ AB_conc_factor,
    ncol      = 1,
    scales    = "fixed",
    labeller  = as_labeller(facet_labels_emp)
  ) +
  # Remap factor levels to new x-axis labels
  scale_x_discrete(
    drop   = FALSE, 
    limits = mic_levels,
    labels = c("2" = "2-4", "4" = "4-8", "8" = "8-16", "16" = "16-32", "32+" = "32+")
  ) +
  # Match color palette with p1 (the first plot)
  scale_fill_manual(
    values = c("Low" = "#008000", "High" = "#00008B"), 
    guide  = guide_legend(reverse = FALSE)
  ) +
  labs(
    x     = "MIC",
    y     = "Relative Frequency",
    fill  = "Persister Fraction",
    title = "Empirical Data"
  ) +
  theme_nature_style()
# The legend remains for the second plot

###############################################################################
# PART C: COMBINE BOTH PLOTS SIDE-BY-SIDE FOR A SINGLE FIGURE
###############################################################################
combined_plot <- plot_grid(
  p1, p2,
  labels = c("A", "B"),
  ncol   = 2,
  align  = "hv"
)

# If you want to directly display the combined plot:
print(combined_plot)

# Save plot in the same directory as the script
# Get script directory (works in both RStudio and command line R)
if (exists("rstudioapi") && rstudioapi::isAvailable()) {
  # If running in RStudio
  script_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
} else {
  # If running from command line or other R environments
  # Use a more robust method to get script directory
  args <- commandArgs(trailingOnly = FALSE)
  script_path <- sub("--file=", "", args[grep("--file=", args)])
  if (length(script_path) > 0) {
    script_dir <- dirname(script_path)
  } else {
    script_dir <- getwd()  # fallback to current working directory
  }
}

# Save the plot in the script directory
ggsave(file.path(script_dir, "Fig_4_a-b.png"), combined_plot, width = 14, height = 7, dpi = 300)
