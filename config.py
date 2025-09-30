# config.py
"""
Master Configuration File for Comprehensive Moisture and Temperature Analysis
of Lagrangian Particle Trajectories.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
# from matplotlib.cm import get_cmap # get_cmap is deprecated in newer matplotlib
# from matplotlib.colors import ListedColormap, BoundaryNorm # Define these in plotting if complex

# --- 0. Workflow Control ---
# ----------------------------------------------------------------------------
# Specify which stage to start the workflow from.
# 1: Start from the beginning (identifying relevant parcels).
# 2: Start from extracting parcel histories (assumes relevant IDs file exists).
# 3: Start from analyzing parcel histories (assumes filtered hourly data exists).
# 4: Start from loading the master DataFrame (assumes analyzed history files exist).
# 5: Start from tracks analysis (assumes master DataFrame parquet file exists).
# 6: Start from plotting and visualization (assumes master DataFrame parquet file exists).
# 7: Generate the tracks of the parcels which contibute moisture to the target area
START_WORKFLOW_FROM_STAGE = 6


# --- I. Core Simulation & Data Paths ---
# -----------------------------------------------------------------------------
# Base output directory for ALL results from this comprehensive analysis
BASE_OUTPUT_DIR = Path("./post_processing_results")

# Input: Directory containing particle CSVs with AT LEAST:
# id, time_step (or means to derive it), latitude, longitude, pressure, specific_humidity, temperature
# These are assumed to be the files generated with the analysis of q and t from ERA5 data.
#AUGMENTED_PARTICLE_DATA_DIR = Path("./particle_output_with_q_t")
AUGMENTED_PARTICLE_DATA_DIR = Path("./output")
# Optional: Path to the original raw partcel simulation output (before q/t augmentation)
# This might be used by some adapted Tracks21.py logic if needed.
#ORIGINAL_PARTICLE_SIM_DIR = Path("./particle_simulation_output")
#ORIGINAL_PARTICLE_SIM_DIR = Path("D:/Manish/ERA_Data/Data/particle_output_with_q_t")
ORIGINAL_PARTICLE_SIM_DIR = AUGMENTED_PARTICLE_DATA_DIR

# --- II. Event Definition & Initial Particle Selection ---
# -----------------------------------------------------------------------------
# These settings are primarily for identifying the pool of "relevant" particles
# that passed through a target region during a specific period.

# Date and time corresponding to the simulation's time_step = 0
SIMULATION_START_DATETIME = pd.Timestamp('2023-12-15 00:00:00')

# Target region definition for selecting relevant particles
TARGET_LAT_CENTER = 8.5683
TARGET_LON_CENTER = 78.1238
# Half-width of the target box in degrees (e.g., 0.09 means a 0.18x0.18 degree box)
TARGET_BOX_HALF_WIDTH_DEG = 0.18 #0.135

# Time steps (hours from SIMULATION_START_DATETIME) during which particles
# arriving in the target box are considered "relevant" for further analysis.
# Example: range(48, 96) includes hours 48 through 95.
RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS = range(66, 102)

# --- III. Detailed Analysis Configuration ---
# -----------------------------------------------------------------------------
# These settings control the detailed moisture and temperature analysis performed
# on the "relevant" particles identified above.

# **Requirement 3: Analysis Start Time for Particle Histories**
# Hour from simulation start (0-indexed) to begin detailed history analysis
# for each selected particle. For example, if particles are selected based on
# arrival at H48, their history will be traced back to this ANALYSIS_START_HOUR.
ANALYSIS_START_HOUR = 0 # Default: trace from the very beginning (time_step 0)
# Example: To start tracing history from simulation hour 10 onwards: ANALYSIS_START_HOUR = 10

# Core period of the "event" for which specific phenomena are analyzed
# (e.g., moisture release *within* the target *during* these hours).
# This can be the same as RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS or a sub-period.
EVENT_CORE_ANALYSIS_STEPS = range(66, 102) # e.g., analyze release in target from H48-H95

# How many hours *after* the LATEST relevant arrival step (from RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS)
# should the histories of relevant particles be extracted and analyzed?
# This allows looking at what happens to particles *after* passing the target.
TRACK_HISTORY_WINDOW_AFTER_MAX_ARRIVAL_HOURS = 24

# Thresholds for defining significant changes
# Specific humidity change (in kg/kg) over CHANGE_WINDOW_HOURS
DQ_THRESHOLD_KG_PER_KG = 0.0002  # (0.2 g/kg)
# Temperature change (in Kelvin) over CHANGE_WINDOW_HOURS
DT_THRESHOLD_KELVIN = 1.0       # (e.g., 1 K change)
# Time window (in hours) for calculating dq/dt and dT/dt
CHANGE_WINDOW_HOURS = 6

# --- IV. Plotting & Animation Configuration ---
# -----------------------------------------------------------------------------

# **Requirement 1: Fixed Plotting Extent**
# Set to None to use dynamic extent based on data for each plot.
# Otherwise, provide (lon_min, lon_max, lat_min, lat_max) for 2D.
# Example: FIXED_PLOT_EXTENT_2D = (60, 100, 0, 30)
#FIXED_PLOT_EXTENT_2D = None
FIXED_PLOT_EXTENT_2D = (70, 110, -5, 30)
# For 3D plots, the lon/lat component of the extent.
#FIXED_PLOT_LONLAT_EXTENT_3D = None # e.g., (60, 100, 0, 30)
FIXED_PLOT_LONLAT_EXTENT_3D = (60, 105, -5, 30)
# For 3D plots, the pressure component (p_min_hPa, p_max_hPa).
#FIXED_PLOT_PRESSURE_EXTENT_3D = None # e.g., (200, 1000)
FIXED_PLOT_PRESSURE_EXTENT_3D = (200,1000)
DEFAULT_PLOT_VIEW_ANGLE_3D = (25, -70) # (elevation, azimuth) for 3D plots
# Default buffer (in degrees) for dynamic extent calculation if fixed extent is None
DEFAULT_PLOT_BUFFER_DEG = 5.0

# Tracks Analysis
# --- Event Definition & Initial Particle Selection (Tracks21.py like settings) ---
TRACKS_PARTICLE_SOURCE_FOLDER = AUGMENTED_PARTICLE_DATA_DIR # Or Path("./particle_output_with_q_t")
#TRACKS_OUTPUT_DIR = BASE_OUTPUT_DIR / PLOTTING_OUTPUT_DIR/ "tracks_analysis_output" # Dedicated output
# SIMULATION_START_DATETIME = pd.Timestamp('2023-12-01 00:00:00') # Already there
#TRACKS_TARGET_STEPS = range(48, 96) # Steps for this specific analysis
TRACKS_TARGET_STEPS=RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS
TRACKS_PLOT_EXTENT_2D=FIXED_PLOT_EXTENT_2D
# TARGET_LAT_CENTER, TARGET_LON_CENTER, TARGET_BOX_HALF_WIDTH_DEG are used by both


# Animation Settings
# These define the range of simulation hours to generate animation frames for.
# By default, they are set to the core event window.
# To create a longer or different animation, uncomment and edit the two lines below.
ANIMATION_START_HOUR = 48
ANIMATION_END_HOUR = 120
ANIMATION_FRAME_START_HOUR = min(EVENT_CORE_ANALYSIS_STEPS)
ANIMATION_FRAME_END_HOUR = max(EVENT_CORE_ANALYSIS_STEPS) # Inclusive end for frames

# If custom animation window is defined, use it. Otherwise, use the event window.
if 'ANIMATION_START_HOUR' in locals() and 'ANIMATION_END_HOUR' in locals():
    ANIMATION_FRAME_START_HOUR = ANIMATION_START_HOUR
    ANIMATION_FRAME_END_HOUR = ANIMATION_END_HOUR
ANIMATION_FPS = 2
ANIMATION_DPI = 100 # DPI for individual animation frames

# Max number of individual particle trajectories to plot for detailed static plots (None for all)
MAX_INDIVIDUAL_TRAJECTORIES_TO_PLOT = 10

# Time Evolution Plots
# **Requirement 8: Time Evolution Plot Steps**
# Generate focused time-evolution plots for particles arriving at these specific hours.
# These hours should ideally be within RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS.
# Creates plots every 6 hours within the core event.
_event_start = min(EVENT_CORE_ANALYSIS_STEPS)
_event_end = max(EVENT_CORE_ANALYSIS_STEPS)
TIME_EVOLUTION_FOCUS_STEPS = list(range(_event_start, _event_end + 1, 1))
# Example: If EVENT_CORE_ANALYSIS_STEPS is range(48,96), this becomes [48, 54, ..., 90]

# Grid resolution for 2D aggregate heatmap plots (degrees)
AGGREGATE_MAP_GRID_RESOLUTION_DEG = 0.2

# Colormaps
CMAP_SPECIFIC_HUMIDITY = 'YlGnBu'
CMAP_TEMPERATURE = 'OrRd'        # Sequential for temperature values
CMAP_PRESSURE = 'plasma'          # For pressure values
CMAP_NET_MOISTURE_CHANGE = 'coolwarm' # Diverging for net q change
CMAP_NET_TEMP_CHANGE = 'RdBu_r'     # Diverging for net T change (RdBu_r: red=warm, blue=cool)
CMAP_UPTAKE_AGG = 'Blues'         # For aggregate uptake frequency/magnitude
CMAP_RELEASE_AGG = 'Reds'          # For aggregate release frequency/magnitude
CMAP_WARMING_AGG = 'YlOrBr'       # For aggregate warming frequency/magnitude
CMAP_COOLING_AGG = 'PuBu'         # For aggregate cooling frequency/magnitude

# Color for target area visualization in plots
TARGET_AREA_PLOT_COLOR = 'darkgreen' # Or any distinct color

# Percentiles for robust color scaling on aggregate maps (e.g., sum of dq/dt)
AGGREGATE_COLOR_PERCENTILES = (2, 98) # Ignore the most extreme 2% on either side

# Pressure Bins for certain plots (e.g., Tracks21 style, some vertical profiles)
# These are primarily for categorical coloring by pressure level.
PRESSURE_BINS_CATEGORICAL = np.arange(0, 1001, 100) # hPa
# For finer vertical profile analysis (e.g., averaging q or T in target)
VERTICAL_PROFILE_PRESSURE_BINS = np.arange(100, 1001, 50) # hPa

# --- V. Parallel Processing ---
# -----------------------------------------------------------------------------
# Number of worker processes to use for parallelizable tasks.
# Using slightly less than total CPUs can leave resources for system.
NUM_WORKERS = 5 #max(1, cpu_count() - 1 if cpu_count() and cpu_count() > 1 else 1)
#NUM_WORKERS = 20

# --- VI. Output File Naming Conventions (Optional, for consistency) ---
# -----------------------------------------------------------------------------
# Prefixes or suffixes for different types of output files.
# Example: ANIMATION_MP4_SUFFIX = "_animation.mp4"
# This can also be handled within the respective modules.

# --- VII. Derived Configurations (DO NOT EDIT MANUALLY - Populated by main script) ---
# -----------------------------------------------------------------------------
# These will be calculated and set by the main workflow script after this config is loaded.
# Example: `cfg.MAX_HISTORY_TRACKING_HOUR`, `cfg.SIMULATION_START_DATE_PLOT_FORMAT`
# For now, initialize as None or with placeholder values.
MAX_HISTORY_TRACKING_HOUR = None # Calculated based on RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS & TRACK_HISTORY_WINDOW_AFTER_MAX_ARRIVAL_HOURS
DATETIME_PLOT_FORMAT = "%d-%b %H:%M" # For plot titles
#DATETIME_PLOT_FORMAT = "%d-%m-%Y %H:%M" # For plot titles
# Placeholder for colormap objects if they are to be globally defined from names
# cmap_pressure_obj = ListedColormap(get_cmap(CMAP_PRESSURE)(np.linspace(0, 1, len(PRESSURE_BINS_CATEGORICAL) - 1)))
# norm_pressure_obj = BoundaryNorm(PRESSURE_BINS_CATEGORICAL, cmap_pressure_obj.N)


# --- VIII. Module-Specific Output Subdirectories (defined here for overview) ---
# -----------------------------------------------------------------------------
# These paths will be constructed using BASE_OUTPUT_DIR.
# The actual creation (mkdir) will be handled by a setup function in the main script.

# TRACKS_LOGIC_OUTPUT_DIR = BASE_OUTPUT_DIR / "0_initial_particle_selection" (If Tracks21 logic is separate)
DATA_PROCESSING_OUTPUT_DIR = BASE_OUTPUT_DIR / "1_processed_data"

MOISTURE_TEMP_ANALYSIS_OUTPUT_DIR = BASE_OUTPUT_DIR / "1_processed_data"
RELEVANT_IDS_FILE = MOISTURE_TEMP_ANALYSIS_OUTPUT_DIR / "relevant_particle_ids_for_analysis.csv"
FILTERED_HOURLY_DATA_DIR = MOISTURE_TEMP_ANALYSIS_OUTPUT_DIR / "filtered_hourly_data"
ANALYZED_PARTICLE_HISTORIES_DIR = MOISTURE_TEMP_ANALYSIS_OUTPUT_DIR / "analyzed_particle_histories"
MASTER_DF_FILE = MOISTURE_TEMP_ANALYSIS_OUTPUT_DIR / "master_analyzed_df.parquet"


PLOTTING_OUTPUT_DIR = BASE_OUTPUT_DIR / "2_plots_and_animations"
PLOTS_OUTPUT_DIR = BASE_OUTPUT_DIR / "2_plots_and_animations"
TRACKS_OUTPUT_DIR = PLOTTING_OUTPUT_DIR/ "tracks_analysis_output" # Dedicated output
MOISTURE_TRACKS_OUTPUT_DIR = PLOTTING_OUTPUT_DIR / "moisture_releasing_tracks"
# Subdirs for plots:
INDIVIDUAL_TRAJ_PLOTS_SUBDIR = "individual_trajectories"
AGGREGATE_MAPS_MOISTURE_SUBDIR = "aggregate_maps/moisture"
AGGREGATE_MAPS_TEMP_SUBDIR = "aggregate_maps/temperature"
HOURLY_SNAPSHOTS_2D_FRAMES_SUBDIR = "animation_frames/2D"
HOURLY_SNAPSHOTS_3D_FRAMES_SUBDIR = "animation_frames/3D"
ANIMATIONS_SUBDIR = "animations_output"
STATISTICAL_DISTRIBUTIONS_MOISTURE_SUBDIR = "statistical_distributions/moisture"
STATISTICAL_DISTRIBUTIONS_TEMP_SUBDIR = "statistical_distributions/temperature"
TARGET_AREA_ANALYSIS_SUBDIR = "target_area_analysis"
TIME_EVOLUTION_SUBDIR = "time_evolution"
#MOISTURE_TEMP_ANALYSIS_OUTPUT_DIR="moisture_temperature_analysis"

print("Configuration file (config.py) loaded.")