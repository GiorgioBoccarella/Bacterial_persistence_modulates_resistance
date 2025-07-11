# Load necessary libraries
library(ggplot2)
library(reshape2)
library(dplyr)
library(scales)
library(RColorBrewer)  # For color palettes

# Attempt to set working directory to script's own directory
script_dir_set <- FALSE
tryCatch({
  if (interactive() && "rstudioapi" %in% installed.packages() && rstudioapi::isAvailable() && rstudioapi::getSourceEditorContext()$path != "") {
    script_path <- rstudioapi::getSourceEditorContext()$path
    script_dir <- dirname(script_path)
    setwd(script_dir)
    print(paste("RStudio context: Set working directory to script directory:", getwd()))
    script_dir_set <- TRUE
  } else if (sys.nframe() > 0 && !is.null(sys.frame(1)$ofile) && sys.frame(1)$ofile != "") {
    script_path <- sys.frame(1)$ofile
    script_dir <- dirname(normalizePath(script_path)) # normalizePath for robustness
    setwd(script_dir)
    print(paste("Sourced script: Set working directory to script directory:", getwd()))
    script_dir_set <- TRUE
  }
}, error = function(e) {
  print(paste("Error setting script directory:", e$message))
})

if (!script_dir_set) {
  print("Could not automatically set working directory to script location. Using current WD. Please ensure it's correct.")
  print(paste("Current working directory is:", getwd()))
}

# Define simulation IDs and persistence types
simulation_ids <- c(5, 7)
persistence_types <- c('low persistence', 'high persistence')
antibiotic_concentrations <- c('12.5')

# Initialize an empty list to store data from all simulations and concentrations
all_data <- list()
mic_levels <- NULL # Initialize mic_levels
mic_labels <- NULL # Initialize mic_labels

# Loop through each simulation ID and persistence type
for (i in seq_along(simulation_ids)) {
  sim_id <- simulation_ids[i]
  persistence_type <- persistence_types[i]
  
  # Determine the results directory based on persistence type
  if (persistence_type == 'low persistence') {
    results_dir_base <- 'data_Fig_3_a_p_'
  } else if (persistence_type == 'high persistence') {
    results_dir_base <- 'data_Fig_3_a_d_'
  }
  
  for (concentration in antibiotic_concentrations) {
    results_dir <- file.path(paste0(results_dir_base, concentration), paste0('simulation_', sim_id))
    
    # Read the time series data
    time_series_filename <- file.path(results_dir, paste0('time_series_', sim_id, '.csv'))
    print(paste("Loop iteration: sim_id=", sim_id, ", type=", persistence_type, ", conc=", concentration))
    print(paste("Working directory:", getwd()))
    print(paste("Checking for time series file:", time_series_filename))
    print(paste("file.exists() check for time series:", file.exists(time_series_filename)))

    if (!file.exists(time_series_filename)) {
      warning(paste("Time series file not found by R:", time_series_filename))
      # Diagnostic: check absolute path
      abs_path_ts <- file.path(getwd(), time_series_filename) # Construct absolute path
      print(paste("Checking absolute path for time series:", abs_path_ts))
      print(paste("file.exists() for absolute time series:", file.exists(abs_path_ts)))
      next
    }
    time_series_data <- read.csv(time_series_filename)
    
    # Read the MIC values and simulation parameters
    mic_values_filename <- file.path(results_dir, paste0('R_MIC_values_', sim_id, '.csv'))
    print(paste("Checking for MIC file:", mic_values_filename))
    print(paste("file.exists() check for MIC file:", file.exists(mic_values_filename)))
    if (!file.exists(mic_values_filename)) {
      warning(paste("MIC file not found by R:", mic_values_filename))
      # Diagnostic: check absolute path
      abs_path_mic <- file.path(getwd(), mic_values_filename) # Construct absolute path
      print(paste("Checking absolute path for MIC file:", abs_path_mic))
      print(paste("file.exists() for absolute MIC file:", file.exists(abs_path_mic)))
      next
    }
    mic_data <- read.csv(mic_values_filename, header=FALSE, stringsAsFactors=FALSE)
    
    # Find the row index where 'R_id' appears
    mic_start_index <- which(mic_data$V1 == 'R_id')
    
    if (length(mic_start_index) == 0) {
      stop(paste("'R_id' not found in", mic_values_filename))
    }
    
    # Extract simulation parameters
    sim_params <- mic_data[1:(mic_start_index - 2), ]  # Exclude the empty row and the header
    simulation_parameters <- setNames(sim_params$V2, sim_params$V1)
    
    # Extract MIC values
    mic_values <- mic_data[(mic_start_index + 1):nrow(mic_data), ]
    colnames(mic_values) <- c('R_id', 'MIC_value')
    mic_values$R_id <- as.character(mic_values$R_id)
    mic_values$MIC_value <- as.numeric(mic_values$MIC_value)
    
    # Define new MIC ranges and labels
    mic_ranges <- c(2, 4, 8, 16, 32, Inf)  # Updated bins
    mic_labels <- c("2-4", "4-8", "8-16", "16-32", "32+")  # Updated labels
    mic_levels <- c('Sensitive_Total', mic_labels)  # Include 'Sensitive_Total' for ordering
    
    # Assign colors to each MIC range using a colorblind-friendly palette
    color_palette <- c('#377eb8', '#4daf4a', '#ff7f00', '#e41a1c', 'darkred')  # Adjusted
    names(color_palette) <- mic_labels  # Assign names to match labels
    
    # Add color for the summed sensitive population
    color_mapping <- c('Sensitive_Total' = 'black', color_palette)  # 'Sensitive_Total' in black
    
    # Categorize MIC values into the defined ranges
    mic_values$mic_range <- cut(
      mic_values$MIC_value, 
      breaks = mic_ranges, 
      labels = mic_labels, 
      right = FALSE  # Use left-inclusive intervals
    )
    
    mic_values$mic_range <- factor(mic_values$mic_range, levels=mic_labels, ordered=TRUE)
    
    # Melt the time series data to long format
    melted_data <- melt(time_series_data, id.vars=c('time'))
    
    # Sum 'S_pop' and 'PS_pop' to create a total sensitive population
    summed_sensitive_data <- melted_data %>%
      filter(variable %in% c('S_pop', 'PS_pop')) %>%
      group_by(time) %>%
      summarise(value = sum(as.numeric(value), na.rm=TRUE)) %>%
      ungroup() %>%
      mutate(variable = 'Sensitive_Total')  # Renaming for clarity
    
    # Assign a single mic_range category for the summed sensitive population
    summed_sensitive_data <- summed_sensitive_data %>%
      mutate(mic_range = 'Sensitive_Total')  # 'Sensitive_Total' will be colored black
    
    # Extract R populations
    R_pop_data <- melted_data %>% filter(grepl('^R\\d+_pop$', variable))
    
    # Extract R IDs from variable names
    R_pop_data$R_id <- sub('R(\\d+)_pop', '\\1', R_pop_data$variable)
    
    # Merge R_pop_data with mic_values to get mic_range
    R_pop_data <- merge(R_pop_data, mic_values[, c('R_id', 'mic_range')], by='R_id', all.x=TRUE)
    
    # Convert population values to numeric
    R_pop_data$value <- as.numeric(R_pop_data$value)
    
    # Define antibiotic intervals based on simulation parameters
    tfin <- as.numeric(simulation_parameters['tfin'])
    if (is.na(tfin)) {
      warning(paste("'tfin' not found or not numeric for simulation ID", sim_id, "in", results_dir))
      tfin <- max(time_series_data$time, na.rm=TRUE)
    }
    antibiotic_intervals <- data.frame(
      start_time = seq(0, tfin, by=24),
      end_time = seq(5, tfin + 5, by=24)
    )
    
    # Add metadata for faceting
    summed_sensitive_data$simulation_id <- sim_id
    summed_sensitive_data$antibiotic_concentration <- concentration
    summed_sensitive_data$persistence_type <- persistence_type
    
    R_pop_data$simulation_id <- sim_id
    R_pop_data$antibiotic_concentration <- concentration
    R_pop_data$persistence_type <- persistence_type
    
    antibiotic_intervals$simulation_id <- sim_id
    antibiotic_intervals$antibiotic_concentration <- concentration
    antibiotic_intervals$persistence_type <- persistence_type
    
    # Store in the list
    all_data[[length(all_data) + 1]] <- list(
      sensitive = summed_sensitive_data,
      resistant = R_pop_data,
      antibiotic_intervals = antibiotic_intervals
    )
  }
}

# Combine all sensitive and resistant data
combined_sensitive <- do.call(rbind, lapply(all_data, `[[`, 'sensitive'))
combined_resistant <- do.call(rbind, lapply(all_data, `[[`, 'resistant'))
combined_intervals <- do.call(rbind, lapply(all_data, `[[`, 'antibiotic_intervals'))

# Convert time to days
combined_sensitive$time_in_days <- combined_sensitive$time / 24
combined_resistant$time_in_days <- combined_resistant$time / 24
combined_intervals$start_time_days <- combined_intervals$start_time / 24
combined_intervals$end_time_days <- combined_intervals$end_time / 24

# Ensure mic_range is a factor with the correct levels

# Handle combined_sensitive
if (!is.data.frame(combined_sensitive) || nrow(combined_sensitive) == 0) {
  warning("combined_sensitive is not a data frame with rows. Skipping factor conversion.")
} else if (is.null(mic_levels)) {
  warning("mic_levels not defined (likely no data files found). Skipping factor conversion for combined_sensitive.")
} else if (!("mic_range" %in% names(combined_sensitive))) {
  warning("mic_range column not found in combined_sensitive. Skipping factor conversion.")
} else {
  combined_sensitive$mic_range <- factor(combined_sensitive$mic_range, levels = mic_levels)
}

# Handle combined_resistant
if (!is.data.frame(combined_resistant) || nrow(combined_resistant) == 0) {
  warning("combined_resistant is not a data frame with rows. Skipping factor conversion.")
} else if (is.null(mic_levels)) {
  warning("mic_levels not defined (likely no data files found). Skipping factor conversion for combined_resistant.")
} else if (!("mic_range" %in% names(combined_resistant))) {
  warning("mic_range column not found in combined_resistant. Skipping factor conversion.")
} else {
  combined_resistant$mic_range <- factor(combined_resistant$mic_range, levels = mic_levels)
}

# Filter out resistant lineages that are never higher than a threshold
threshold_value <- 10 # Adjust as needed
combined_resistant_filtered <- combined_resistant %>%
  group_by(R_id, persistence_type) %>%
  filter(max(value, na.rm = TRUE) > threshold_value) %>%
  ungroup()

# Update color mapping with colorblind-friendly colors
color_mapping <- c(
  'Sensitive_Total' = 'black',
  '2-4' = '#377eb8',   # Blue
  '4-8' = '#4daf4a',   # Green
  '8-16' = '#ff7f00',  # Orange
  '16-32' = '#e41a1c', # Red
  '32+' = 'darkred'    # Dark Red
)

# Plot
library(viridis)  # For colorblind-friendly color scales (optional)

# Create the plot
plot <- ggplot() +
  # Shade antibiotic intervals
  geom_rect(data=combined_intervals, 
            aes(xmin=start_time_days, xmax=end_time_days, ymin=-Inf, ymax=Inf),
            fill='lightpink1', alpha=0.5) +
  # Plot resistant populations (filtered)
  geom_line(data=combined_resistant_filtered, 
            aes(x=time_in_days, y=value, color=mic_range,  
                group=interaction(simulation_id, R_id, antibiotic_concentration)),
            size=0.6, alpha=0.8) +
  # Plot summed sensitive population
  geom_line(data=combined_sensitive, 
            aes(x=time_in_days, y=value, color=mic_range, 
                group=interaction(simulation_id, variable, antibiotic_concentration)),
            size=0.6, alpha=0.7) +
  scale_y_log10(labels=trans_format("log10", math_format(10^.x))) +
  coord_cartesian(ylim = c(10, NA)) +
  scale_color_manual(
    name='Mutant MIC',
    values=color_mapping,
    guide=guide_legend(override.aes=list(size=3))
  ) +
  scale_x_continuous(
    limits = c(0, 12),  # Extended time axis for clarity
    breaks = seq(0, 12, 2),
    expand = expansion(mult = c(0.02, 0), add = c(0, 0)) # Restore left padding, keep right padding removed
  ) +
  facet_wrap(~ persistence_type, ncol=1, labeller = labeller(persistence_type = function(x) tools::toTitleCase(paste(x, "level")))) +  # Custom labels with title case
  theme_classic() +
  theme(
    text = element_text(size=16, color = "black"),
    axis.text = element_text(size=19, color = "black"),
    axis.title = element_text(size=21, color = "black"),
    legend.position = "right",
    legend.title = element_text(size=14, color = "black"),
    legend.text = element_text(size=12, color = "black"),
    strip.background = element_rect(fill="lightgrey"),
    strip.text = element_text(size = 16, face = "plain", color="black"),
    legend.box = "vertical",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill=NA, size=1),
    plot.margin = margin(10, 10, 10, 10, "pt")
  ) +
  # remove the legend 
  theme(legend.position = "none") +
  labs(
    x = "Days",
    y = "Lineage Size "
  )

plot

# Save the plot with high resolution
ggsave("Fig_3_a.pdf", plot=plot, width=8, height=7.2)
ggsave("Fig_3_a.png", plot=plot, width=8, height=7.2, dpi=300)
