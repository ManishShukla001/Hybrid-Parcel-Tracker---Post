# statistical_analysis.py
"""
Module for performing and plotting statistical analyses of particle trajectories,
moisture changes, and temperature changes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import cartopy.crs as ccrs # For map plots in last_event_points
from matplotlib.colors import Normalize, BoundaryNorm
import cartopy.feature as cfeature
from pathlib import Path
from tqdm import tqdm
import gc
from scipy.stats import binned_statistic
from multiprocessing import Pool, cpu_count
import matplotlib.dates as mdates

# Import global configuration
import config as cfg

# Define a font scaling factor
FONT_SCALE = 2
DEFAULT_FONT_SIZE = 10 # A reasonable base font size

# Import helper from plotting_2d for map features and extent, if needed for map-based stats plots
# Alternatively, define minimal map helpers here if plotting_2d is too heavy a dependency
# For now, let's assume we might need them for plot_last_event_points
try:
    from plotting_2d import _add_map_features_2d, _get_plot_extent, _add_target_area_2d
except ImportError:
    print("Warning: plotting_2d module not found. Some map-based statistical plots might be limited.")
    # Define dummy functions or simplified versions if plotting_2d isn't available yet
    def _add_map_features_2d(ax, include_labels=True): pass
    def _get_plot_extent(df_lon, df_lat, fixed_cfg, buffer_cfg): return (-180, 180, -90, 90)
    def _add_target_area_2d(ax, cfg_obj):pass


def plot_vertical_profile_change(master_df: pd.DataFrame, value_column_base: str,
                                 variable_name: str, config_obj):
    """
    Plots the average change (dq/dt or dT/dt) vs. pressure for uptake/release
    or warming/cooling events across all relevant particles.
    """
    output_dir = config_obj.PLOTS_OUTPUT_DIR / (config_obj.STATISTICAL_DISTRIBUTIONS_MOISTURE_SUBDIR if variable_name == "Moisture" else config_obj.STATISTICAL_DISTRIBUTIONS_TEMP_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Plotting: Vertical Profile of {variable_name} Changes (All Trajectories)")
    if master_df.empty: return

    if variable_name == "Moisture":
        event_type_col = 'moisture_event_type'
        pos_event_label, neg_event_label = 'Uptake', 'Release'
        value_col_plot = 'dq_dt_g_kg' # Assumes this will be created from dq_dt
        master_df[value_col_plot] = master_df['dq_dt'].dropna() * 1000
        cbar_unit_label = f"(g/kg per {config_obj.CHANGE_WINDOW_HOURS}hrs)"
    elif variable_name == "Temperature":
        event_type_col = 'temp_event_type'
        pos_event_label, neg_event_label = 'Warming', 'Cooling'
        value_col_plot = 'dT_dt_K' # Assumes this will be created from dT_dt
        master_df[value_col_plot] = master_df['dT_dt'].dropna() # Already in K
        cbar_unit_label = f"(K per {config_obj.CHANGE_WINDOW_HOURS}hrs)"
    else: return

    event_df = master_df[master_df[event_type_col] != 'Neutral'].copy()
    if event_df.empty or value_col_plot not in event_df.columns:
        print(f"No valid {variable_name} change events found for vertical profile.")
        return
    
    event_df.dropna(subset=['pressure', value_col_plot], inplace=True)
    if event_df.empty:
        print(f"No valid data after NaN drop for {variable_name} vertical profile.")
        return

    pos_events_df = event_df[event_df[event_type_col] == pos_event_label]
    neg_events_df = event_df[event_df[event_type_col] == neg_event_label]

    # Use finer pressure bins for profiles
    profile_bins = config_obj.VERTICAL_PROFILE_PRESSURE_BINS
    
    avg_pos_change, _, _ = binned_statistic(pos_events_df['pressure'], pos_events_df[value_col_plot], statistic='mean', bins=profile_bins)
    avg_neg_change, _, _ = binned_statistic(neg_events_df['pressure'], neg_events_df[value_col_plot], statistic='mean', bins=profile_bins)
    
    bin_centers = (profile_bins[:-1] + profile_bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 10))
    plot_occurred = False
    if not pos_events_df.empty and not np.all(np.isnan(avg_pos_change)):
        ax.plot(avg_pos_change, bin_centers, 'b-^', label=f'Avg. {pos_event_label} {value_col_plot.split("_")[0]}/dt')
        plot_occurred = True
    if not neg_events_df.empty and not np.all(np.isnan(avg_neg_change)):
        ax.plot(avg_neg_change, bin_centers, 'r-v', label=f'Avg. {neg_event_label} {value_col_plot.split("_")[0]}/dt')
        plot_occurred = True
    
    if not plot_occurred:
        print(f"No data to plot for {variable_name} vertical profile.")
        plt.close(fig); gc.collect()
        return

    ax.set_xlabel('Average Change', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_ylabel('Pressure (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_title(f'Vertical Profile of Average {variable_name} Changes {cbar_unit_label}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.invert_yaxis()
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.legend(fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)

    plt.savefig(output_dir / f"vertical_profile_{variable_name.lower()}_change.png", bbox_inches='tight', dpi=150)
    plt.close(fig); gc.collect()

def plot_histogram_event_magnitudes(master_df: pd.DataFrame, value_column_base: str,
                                    variable_name: str, config_obj):
    """Plots histograms of event magnitudes (uptake/release or warming/cooling)."""
    output_dir = config_obj.PLOTS_OUTPUT_DIR / (config_obj.STATISTICAL_DISTRIBUTIONS_MOISTURE_SUBDIR if variable_name == "Moisture" else config_obj.STATISTICAL_DISTRIBUTIONS_TEMP_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting: Histogram of {variable_name} Event Magnitudes")
    if master_df.empty: return

    if variable_name == "Moisture":
        event_type_col = 'moisture_event_type'
        pos_event_label, neg_event_label = 'Uptake', 'Release'
        value_col_plot = 'dq_dt_g_kg'
        master_df[value_col_plot] = master_df['dq_dt'].dropna() * 1000
        unit_label = f"(g/kg per {config_obj.CHANGE_WINDOW_HOURS}hrs)"
        pos_color, neg_color = 'skyblue', 'salmon'
    elif variable_name == "Temperature":
        event_type_col = 'temp_event_type'
        pos_event_label, neg_event_label = 'Warming', 'Cooling'
        value_col_plot = 'dT_dt_K'
        master_df[value_col_plot] = master_df['dT_dt'].dropna()
        unit_label = f"(K per {config_obj.CHANGE_WINDOW_HOURS}hrs)"
        pos_color, neg_color = 'sandybrown', 'lightsteelblue'
    else: return

    event_df = master_df[master_df[event_type_col] != 'Neutral'].copy()
    if event_df.empty or value_col_plot not in event_df.columns:
        print(f"No {variable_name} events for histogram.")
        return
    event_df.dropna(subset=[value_col_plot], inplace=True)

    pos_magnitudes = event_df[event_df[event_type_col] == pos_event_label][value_col_plot]
    # For release/cooling, magnitudes are negative, take absolute for histogram if desired, or plot raw
    neg_magnitudes_raw = event_df[event_df[event_type_col] == neg_event_label][value_col_plot]
    neg_magnitudes_abs = neg_magnitudes_raw.abs()


    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    plot_occurred = False
    if not pos_magnitudes.empty:
        axs[0].hist(pos_magnitudes, bins=25, color=pos_color, edgecolor='black', alpha=0.75)
        axs[0].set_title(f'Distribution of {pos_event_label} Magnitudes', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        axs[0].set_xlabel(f'{variable_name} Gained {unit_label}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        axs[0].set_ylabel('Number of Events', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        axs[0].grid(axis='y', linestyle=':', alpha=0.7)
        axs[0].tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        plot_occurred = True

    if not neg_magnitudes_abs.empty:
        axs[1].hist(neg_magnitudes_abs, bins=25, color=neg_color, edgecolor='black', alpha=0.75)
        axs[1].set_title(f'Distribution of {neg_event_label} Magnitudes (Abs Value)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        axs[1].set_xlabel(f'{variable_name} Lost {unit_label}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        axs[1].grid(axis='y', linestyle=':', alpha=0.7)
        axs[1].tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        plot_occurred = True
    
    if not plot_occurred:
        print(f"No data to plot for {variable_name} magnitude histogram.")
        plt.close(fig); gc.collect()
        return

    plt.tight_layout()
    plt.savefig(output_dir / f"histogram_{variable_name.lower()}_event_magnitudes.png", bbox_inches='tight', dpi=150)
    plt.close(fig); gc.collect()

def plot_last_event_points(master_df: pd.DataFrame, relevant_ids: list,
                           event_to_find: str, variable_name: str, config_obj):
    """
    Plots geographical distribution of the last significant event (Uptake/Warming)
    before particles first arrive in the target area during EVENT_CORE_ANALYSIS_STEPS.
    """
    output_dir = config_obj.PLOTS_OUTPUT_DIR / (config_obj.AGGREGATE_MAPS_MOISTURE_SUBDIR if variable_name == "Moisture" else config_obj.AGGREGATE_MAPS_TEMP_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting: Geographical Distribution of Last {event_to_find} Points for {variable_name}")
    if master_df.empty or not relevant_ids: return

    if variable_name == "Moisture": event_type_col = 'moisture_event_type'
    elif variable_name == "Temperature": event_type_col = 'temp_event_type'
    else: return

    last_event_data = []
    for particle_id in tqdm(relevant_ids, desc=f"Finding last '{event_to_find}' points"):
        df_particle = master_df[master_df['particle_id'] == particle_id].sort_values(by='time_step')
        if df_particle.empty: continue

        first_arrival_step_in_core_event = -1
        for step in config_obj.EVENT_CORE_ANALYSIS_STEPS: # Use core event steps for arrival
            pos_at_step = df_particle[df_particle['time_step'] == step]
            if not pos_at_step.empty:
                lat = pos_at_step['latitude'].iloc[0]
                lon = pos_at_step['longitude'].iloc[0]
                if (config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG <= lat <= config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG and
                    config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG <= lon <= config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG):
                    first_arrival_step_in_core_event = step
                    break
        
        if first_arrival_step_in_core_event == -1: continue

        history_before_arrival = df_particle[df_particle['time_step'] < first_arrival_step_in_core_event]
        qualifying_events = history_before_arrival[history_before_arrival[event_type_col] == event_to_find]

        if not qualifying_events.empty:
            last_event = qualifying_events.iloc[-1]
            last_event_data.append({
                'particle_id': particle_id, 'latitude': last_event['latitude'], 'longitude': last_event['longitude'],
                'pressure': last_event['pressure'], 'time_step': last_event['time_step']
            })

    if not last_event_data:
        print(f"No last '{event_to_find}' events found for particles reaching target during core event.")
        return
    
    df_last_events = pd.DataFrame(last_event_data)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_title(f'Location of Last Sig. {event_to_find} ({variable_name}) Before Target Arrival', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    _add_map_features_2d(ax)
    _add_target_area_2d(ax, config_obj) # Requirement 4
    
    plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max = _get_plot_extent(
        df_last_events['longitude'], df_last_events['latitude'],
        config_obj.FIXED_PLOT_EXTENT_2D, config_obj.DEFAULT_PLOT_BUFFER_DEG
    )
    ax.set_extent([plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max], crs=ccrs.PlateCarree())

    cmap_pressure_obj = get_cmap(config_obj.CMAP_PRESSURE)
    norm_pressure_obj = BoundaryNorm(config_obj.PRESSURE_BINS_CATEGORICAL, cmap_pressure_obj.N)

    sc = ax.scatter(df_last_events['longitude'], df_last_events['latitude'],
                    c=df_last_events['pressure'], cmap=cmap_pressure_obj, norm=norm_pressure_obj,
                    s=35, alpha=0.75, transform=ccrs.Geodetic(), zorder=3, edgecolor='black', linewidth=0.4)
    
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.6)
    cbar.set_label(f'Pressure at Last {event_to_find} (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    cbar.ax.tick_params(labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    cbar.set_ticks(config_obj.PRESSURE_BINS_CATEGORICAL[::2])
    ax.legend(loc='best', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)

    plt.savefig(output_dir / f"last_{event_to_find.lower()}_points_{variable_name.lower()}.png", bbox_inches='tight', dpi=150)
    plt.close(fig); gc.collect()


# --- Requirement 6: Time in Target & Release in Target ---
def analyze_and_plot_time_in_target(master_df: pd.DataFrame, relevant_ids: list, config_obj):
    """Calculates and plots time spent by each relevant particle in the target region."""
    output_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.TARGET_AREA_ANALYSIS_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Analyzing: Time Spent in Target Region")

    if master_df.empty or not relevant_ids: return

    lat_min = config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lat_max = config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_min = config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_max = config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG

    # Filter for particles and time steps when they are in the target region
    # Consider only during the core event for this specific analysis
    core_event_df = master_df[master_df['time_step'].isin(config_obj.EVENT_CORE_ANALYSIS_STEPS)]
    in_target_df = core_event_df[
        (core_event_df['latitude'].between(lat_min, lat_max)) &
        (core_event_df['longitude'].between(lon_min, lon_max)) &
        (core_event_df['particle_id'].isin(relevant_ids)) # Ensure it's one of the overall relevant particles
    ]

    if in_target_df.empty:
        print("No particle positions found within the target region during core event steps.")
        return

    # Time spent is 1 hour per record (assuming hourly data)
    time_in_target_per_particle = in_target_df.groupby('particle_id').size().reset_index(name='hours_in_target')
    time_in_target_per_particle.to_csv(output_dir / "time_spent_in_target.csv", index=False)

    # Plot 1: Histogram of time spent in target
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(time_in_target_per_particle['hours_in_target'], bins=range(1, int(time_in_target_per_particle['hours_in_target'].max()) + 2),
             color='teal', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Hours Spent in Target Region (during core event)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax1.set_ylabel('Number of Particles', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax1.set_title('Distribution of Time Spent by Particles in Target Region', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    plt.savefig(output_dir / "hist_time_in_target.png", bbox_inches='tight', dpi=150)
    plt.close(fig1)

    # Plot 2: Map of initial positions colored by time spent in target (optional, can be complex if too many)
    # This requires merging with initial position data. For now, focusing on histogram.
    print(f"Time in target analysis complete. Data saved, histogram plotted.")
    gc.collect()

def analyze_and_plot_release_in_target(master_df: pd.DataFrame, relevant_ids: list,
                                       variable_name: str, config_obj):
    """Calculates and plots amount of moisture/heat released by particles while in the target region."""
    output_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.TARGET_AREA_ANALYSIS_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing: {variable_name} Release in Target Region")

    if master_df.empty or not relevant_ids: return

    if variable_name == "Moisture":
        event_type_col, value_col, release_event_label = 'moisture_event_type', 'dq_dt', 'Release'
        unit_plot = 'g/kg' # for total release; dq_dt is in kg/kg per X hrs
        multiplier = 1000 # kg/kg to g/kg
    elif variable_name == "Temperature":
        event_type_col, value_col, release_event_label = 'temp_event_type', 'dT_dt', 'Cooling' # "Release" of heat is cooling
        unit_plot = 'K'
        multiplier = 1 # K is K
    else: return

    lat_min = config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lat_max = config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_min = config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_max = config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG

    core_event_df = master_df[master_df['time_step'].isin(config_obj.EVENT_CORE_ANALYSIS_STEPS)]
    in_target_and_releasing = core_event_df[
        (core_event_df['latitude'].between(lat_min, lat_max)) &
        (core_event_df['longitude'].between(lon_min, lon_max)) &
        (core_event_df[event_type_col] == release_event_label) &
        (core_event_df['particle_id'].isin(relevant_ids))
    ].copy() # Make a copy to avoid SettingWithCopyWarning

    if in_target_and_releasing.empty:
        print(f"No significant {variable_name.lower()} release events found within target region during core event.")
        return

    # `value_col` (dq/dt or dT/dt) is change over X hours. To get total, sum these changes.
    # Note: dq_dt/dT_dt for release/cooling is negative. We sum these negative values.
    in_target_and_releasing['release_amount_plot_units'] = in_target_and_releasing[value_col] * multiplier
    total_release_in_target = in_target_and_releasing.groupby('particle_id')['release_amount_plot_units'].sum().reset_index()
    total_release_in_target.rename(columns={'release_amount_plot_units': f'total_release_in_target_{unit_plot.replace("/", "_per_")}'}, inplace=True)
    
    # Filter out particles with zero or positive total release (i.e., net gain or no change)
    total_release_in_target = total_release_in_target[total_release_in_target.iloc[:,1] < 0]


    if total_release_in_target.empty:
        print(f"No net {variable_name.lower()} release by any particle within target region.")
        return

    total_release_in_target.to_csv(output_dir / f"total_{variable_name.lower()}_release_in_target.csv", index=False)

    # Plot 1: Histogram of total release amounts
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # Plot the actual (negative) release values for moisture, or cooling values for temp
    ax1.hist(total_release_in_target.iloc[:,1], bins=25,
             color='maroon' if variable_name=="Moisture" else 'darkblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel(f'Total {variable_name} Released in Target {unit_plot} (sum over {config_obj.CHANGE_WINDOW_HOURS}hr changes)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax1.set_ylabel('Number of Particles', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax1.set_title(f'Distribution of Total {variable_name} Release in Target Region', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    plt.savefig(output_dir / f"hist_total_{variable_name.lower()}_release_in_target.png", bbox_inches='tight', dpi=150)
    plt.close(fig1)
    
    print(f"{variable_name} release in target analysis complete.")
    gc.collect()


# --- Requirement 7: Vertical Profiles in Target ---
def plot_vertical_profile_in_target(master_df: pd.DataFrame, value_col_to_avg: str,
                                    value_label: str, config_obj):
    """
    Calculates and plots the average of a given value_col_to_avg (e.g., q, T, dq/dt)
    against pressure, for particles *while they are inside the target region*
    during EVENT_CORE_ANALYSIS_STEPS.
    """
    output_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.TARGET_AREA_ANALYSIS_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plotting: Vertical Profile of Avg {value_label} in Target Region")

    if master_df.empty or value_col_to_avg not in master_df.columns:
        print(f"Master df empty or column '{value_col_to_avg}' not found. Skipping.")
        return

    lat_min = config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lat_max = config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_min = config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_max = config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG

    in_target_df = master_df[
        (master_df['latitude'].between(lat_min, lat_max)) &
        (master_df['longitude'].between(lon_min, lon_max)) &
        (master_df['time_step'].isin(config_obj.EVENT_CORE_ANALYSIS_STEPS)) # During the core event
    ].copy()

    if in_target_df.empty or value_col_to_avg not in in_target_df.columns:
        print(f"No particle data found within target region for column '{value_col_to_avg}'.")
        return
        
    in_target_df.dropna(subset=['pressure', value_col_to_avg], inplace=True)
    if in_target_df.empty:
        print(f"No valid data after NaN drop for profile of '{value_col_to_avg}' in target.")
        return

    # Convert to g/kg if it's specific humidity or dq_dt
    plot_values = in_target_df[value_col_to_avg]
    unit_str = "(K)" if "temp" in value_label.lower() or "dT_dt" in value_col_to_avg else "(kg/kg)"
    if "humidity" in value_label.lower() or "dq_dt" in value_col_to_avg:
        plot_values = plot_values * 1000
        unit_str = "(g/kg)" if "humidity" in value_label.lower() else f"(g/kg per {config_obj.CHANGE_WINDOW_HOURS}hrs)"
    
    profile_bins = config_obj.VERTICAL_PROFILE_PRESSURE_BINS
    avg_values_in_target, _, _ = binned_statistic(
        in_target_df['pressure'], plot_values,
        statistic='mean', bins=profile_bins
    )
    bin_centers = (profile_bins[:-1] + profile_bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 10))
    if not np.all(np.isnan(avg_values_in_target)):
        ax.plot(avg_values_in_target, bin_centers, 'm-o', label=f'Avg. {value_label}')
    else:
        print(f"Not enough data to plot vertical profile of {value_label} in target.")
        plt.close(fig); gc.collect(); return
        
    ax.set_xlabel(f'Average {value_label} {unit_str}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_ylabel('Pressure (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_title(f'Vertical Profile of Average {value_label}\n(Particles within Target Region during Core Event)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.invert_yaxis()
    if "change" in value_label.lower() or "dt" in value_col_to_avg: # Add vline at 0 for change plots
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.legend(fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)

    filename_suffix = value_col_to_avg.replace('/', '_per_')
    plt.savefig(output_dir / f"vertical_profile_in_target_{filename_suffix}.png", bbox_inches='tight', dpi=150)
    plt.close(fig); gc.collect()


# --- Requirement 8: Time Evolution Plots ---
def _plot_time_evolution_core(
    time_axis_data, # Can be relative hours or actual datetimes
    avg_profile, std_profile,
    event_counts_pos, event_counts_neg,
    variable_name,
    title_focus_label, # More descriptive label for title
    xlabel_str, # Specific x-axis label
    num_particles_in_ensemble: int, # Number of particles this plot is based on
    arrival_datetime_for_vline: pd.Timestamp | None, # Absolute datetime for vline if applicable
    config_obj, output_dir, filename_suffix
):
    """Core plotting logic for time evolution, now more flexible."""
    
    unit_val = "(g/kg)" if variable_name == "Moisture" else "(K)"
    pos_event_label = "Uptake" if variable_name == "Moisture" else "Warming"
    neg_event_label = "Release" if variable_name == "Moisture" else "Cooling"
    val_col_label = 'Specific Humidity' if variable_name == "Moisture" else 'Temperature'

    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True) # Made wider

    # Plot 1: Average Value (q or T)
    axs[0].plot(time_axis_data, avg_profile, label=f'Mean {val_col_label}', color='black')
    axs[0].fill_between(time_axis_data, avg_profile - std_profile, avg_profile + std_profile,
                        color='gray', alpha=0.3, label='Mean ±1 Std Dev')
    axs[0].set_ylabel(f'{val_col_label} {unit_val}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    
    full_title = f'{variable_name} Time Evolution {title_focus_label}\n(N = {num_particles_in_ensemble} particles)'
    axs[0].set_title(full_title, fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)

    if arrival_datetime_for_vline:
        axs[0].axvline(arrival_datetime_for_vline, color='r', linestyle='--', linewidth=1.5, label=f'Arrival at Target:\n{arrival_datetime_for_vline.strftime("%d-%b%H:%M")}')
    
    axs[0].legend(fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].tick_params(axis='y', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)

    # Plot 2: Event Counts
    axs[1].plot(time_axis_data, event_counts_pos, label=f'{pos_event_label} Event Count', color='blue', marker='^', linestyle='-', markersize=5)
    axs[1].plot(time_axis_data, event_counts_neg, label=f'{neg_event_label} Event Count', color='red', marker='v', linestyle='-', markersize=5)
    axs[1].set_xlabel(xlabel_str, fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    axs[1].set_ylabel('No. of Particles with Event', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    if arrival_datetime_for_vline:
        axs[1].axvline(arrival_datetime_for_vline, color='r', linestyle='--', linewidth=1.5)
    
    axs[1].legend(fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    axs[1].grid(True, linestyle=':', alpha=0.7)
    axs[1].tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    
    # Format x-axis if data is datetime
    if pd.api.types.is_datetime64_any_dtype(time_axis_data):
        fig.autofmt_xdate(rotation=0, ha='center') # Auto format and rotate
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to make space for super title potentially
    # fig.suptitle(full_title, fontsize=14) # Alternative title placement
    plt.savefig(output_dir / f"time_evolution_{variable_name.lower()}_{filename_suffix}.png", dpi=150)
    plt.close(fig); gc.collect()

def plot_time_evolution_multi_step(master_df: pd.DataFrame, relevant_ids: list,
                                   variable_name: str, config_obj):
    output_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.TIME_EVOLUTION_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plotting: Time Evolution for {variable_name} (Focused Arrival Steps)")

    if master_df.empty or not relevant_ids: return
    
    val_col = 'specific_humidity' if variable_name == "Moisture" else 'temperature'
    event_type_col = 'moisture_event_type' if variable_name == "Moisture" else 'temp_event_type'
    pos_event, neg_event = ('Uptake', 'Release') if variable_name == "Moisture" else ('Warming', 'Cooling')
    multiplier = 1000 if variable_name == "Moisture" else 1

    for focus_step in config_obj.TIME_EVOLUTION_FOCUS_STEPS:
        arrival_datetime_at_focus = config_obj.SIMULATION_START_DATETIME + pd.Timedelta(hours=focus_step)
        
        df_focus_hour = master_df[ (master_df['time_step'] == focus_step) & (master_df['particle_id'].isin(relevant_ids)) ]
        if df_focus_hour.empty: continue

        lat_min, lat_max = config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
        lon_min, lon_max = config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
        
        mask_arrival = (df_focus_hour['latitude'].between(lat_min, lat_max) & df_focus_hour['longitude'].between(lon_min, lon_max))
        particles_at_focus_step = df_focus_hour.loc[mask_arrival, 'particle_id'].unique().tolist()

        if not particles_at_focus_step: continue
        print(f"  Processing H{focus_step} arrivals: {len(particles_at_focus_step)} particles for {variable_name}.")

        aligned_histories, min_rel_t, max_rel_t = [], 0, 0
        for pid in particles_at_focus_step:
            dfp = master_df[master_df['particle_id'] == pid].copy()
            if dfp.empty: continue
            dfp['time_relative_to_arrival'] = dfp['time_step'] - focus_step
            # Convert relative time to absolute datetime for this specific cohort
            dfp['absolute_datetime'] = arrival_datetime_at_focus + pd.to_timedelta(dfp['time_relative_to_arrival'], unit='h')
            aligned_histories.append(dfp)
            min_rel_t = min(min_rel_t, dfp['time_relative_to_arrival'].min())
            max_rel_t = max(max_rel_t, dfp['time_relative_to_arrival'].max())
        
        if not aligned_histories: continue

        # Use absolute datetime for x-axis
        common_abs_datetime_axis = pd.date_range(
            start=arrival_datetime_at_focus + pd.Timedelta(hours=min_rel_t),
            end=arrival_datetime_at_focus + pd.Timedelta(hours=max_rel_t),
            freq='h' # Assuming hourly steps
        )
        if common_abs_datetime_axis.empty: continue

        interpolated_series_list = []
        pos_counts = pd.Series(0, index=common_abs_datetime_axis, dtype=int)
        neg_counts = pd.Series(0, index=common_abs_datetime_axis, dtype=int)

        for dfp_aligned in aligned_histories:
            # Interpolate using datetime index
            dfp_temp = dfp_aligned.set_index('absolute_datetime')
            # Reindex to common axis and interpolate, then get values
            interp_series = dfp_temp[val_col].dropna().multiply(multiplier).reindex(common_abs_datetime_axis).interpolate(method='time')
            interpolated_series_list.append(interp_series)

            for dt_idx, row in dfp_aligned.iterrows(): # Iterate over original rows
                abs_dt = row['absolute_datetime']
                if abs_dt in common_abs_datetime_axis: # Check if event's datetime is on our axis
                    if row[event_type_col] == pos_event: pos_counts.loc[abs_dt] += 1
                    elif row[event_type_col] == neg_event: neg_counts.loc[abs_dt] += 1
        
        avg_values_df = pd.concat(interpolated_series_list, axis=1)
        mean_profile = avg_values_df.mean(axis=1)
        std_profile = avg_values_df.std(axis=1)

        title_focus_label_str = f"around Arrival at {arrival_datetime_at_focus.strftime('%d-%b')}"
        xlabel_str_val = "Date / Time"
        _plot_time_evolution_core(common_abs_datetime_axis, mean_profile, std_profile, 
                                  pos_counts, neg_counts, variable_name, title_focus_label_str,
                                  xlabel_str_val, len(particles_at_focus_step), arrival_datetime_at_focus,
                                  config_obj, output_dir, f"focus_H{focus_step}")

def plot_time_evolution_full_event(master_df: pd.DataFrame, relevant_ids: list,
                                   variable_name: str, config_obj):
    # ... (This function inherently uses "Relative Hours" so datetime x-axis is not straightforward)
    # We will keep its x-axis as relative hours but add N to title.
    output_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.TIME_EVOLUTION_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plotting: Time Evolution for {variable_name} (Full Event - Aligned by First Arrival)")

    if master_df.empty or not relevant_ids: return
    
    val_col = 'specific_humidity' if variable_name == "Moisture" else 'temperature'
    event_type_col = 'moisture_event_type' if variable_name == "Moisture" else 'temp_event_type'
    pos_event, neg_event = ('Uptake', 'Release') if variable_name == "Moisture" else ('Warming', 'Cooling')
    multiplier = 1000 if variable_name == "Moisture" else 1
    
    lat_min = config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG; lat_max = config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_min = config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG; lon_max = config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG

    aligned_histories_full_event, min_rel_t_full, max_rel_t_full = [], 0, 0
    num_particles_in_full_event_ensemble = 0

    for pid in tqdm(relevant_ids, desc=f"Aligning trajectories for {variable_name} full event"):
        dfp = master_df[master_df['particle_id'] == pid].sort_values(by='time_step').copy()
        if dfp.empty: continue
        first_arrival_in_core = -1
        for step in config_obj.EVENT_CORE_ANALYSIS_STEPS:
            pos_at_step = dfp[dfp['time_step'] == step]
            if not pos_at_step.empty:
                if (pos_at_step['latitude'].between(lat_min, lat_max).iloc[0] and pos_at_step['longitude'].between(lon_min, lon_max).iloc[0]):
                    first_arrival_in_core = step; break
        if first_arrival_in_core != -1:
            dfp['time_relative_to_first_arrival'] = dfp['time_step'] - first_arrival_in_core
            aligned_histories_full_event.append(dfp)
            num_particles_in_full_event_ensemble +=1
            min_rel_t_full = min(min_rel_t_full, dfp['time_relative_to_first_arrival'].min())
            max_rel_t_full = max(max_rel_t_full, dfp['time_relative_to_first_arrival'].max())

    if not aligned_histories_full_event: return
    print(f"  Processing full event: {len(aligned_histories_full_event)} particles for {variable_name}.")

    common_rel_time_full = np.arange(min_rel_t_full, max_rel_t_full + 1)
    if len(common_rel_time_full) == 0: return # Avoid issues if range is empty

    interpolated_values_list = []
    pos_counts_full = pd.Series(0, index=common_rel_time_full, dtype=int)
    neg_counts_full = pd.Series(0, index=common_rel_time_full, dtype=int)

    for dfp_aligned in aligned_histories_full_event:
        # Ensure xp and fp are the same length and aligned
        valid_mask = (~dfp_aligned['time_relative_to_first_arrival'].isna()) & (~dfp_aligned[val_col].isna())
        xp = dfp_aligned.loc[valid_mask, 'time_relative_to_first_arrival']
        fp = dfp_aligned.loc[valid_mask, val_col] * multiplier
        interp_val = np.interp(common_rel_time_full, xp, fp, left=np.nan, right=np.nan)
        interpolated_values_list.append(interp_val)
        for _, row in dfp_aligned.iterrows():
            rel_t = row['time_relative_to_first_arrival']
            if rel_t in common_rel_time_full:
                if row[event_type_col] == pos_event: pos_counts_full.loc[rel_t] += 1
                elif row[event_type_col] == neg_event: neg_counts_full.loc[rel_t] += 1
    
    avg_values_df_full = pd.DataFrame(interpolated_values_list).T
    avg_values_df_full.index = common_rel_time_full
    mean_profile_full = avg_values_df_full.mean(axis=1)
    std_profile_full = avg_values_df_full.std(axis=1)

    title_focus_label_str_full = "Relative to First Target Arrival"
    xlabel_str_val_full = "Hours from First Arrival in Target"
    _plot_time_evolution_core(common_rel_time_full, mean_profile_full, std_profile_full,
                              pos_counts_full, neg_counts_full, variable_name,
                              title_focus_label_str_full, xlabel_str_val_full,
                              num_particles_in_full_event_ensemble, None, # No single vline for full event
                              config_obj, output_dir, "full_event_aligned")
                              
def plot_vertical_profile_change_vs_mean_pressure(master_df: pd.DataFrame,
                                                  variable_name: str, config_obj,
                                                  time_window_hrs: int = 6):
    """
    Plots average 6-hr or 1-hr change (dq/dt or dT/dt) vs. the mean pressure
    over that change window. For all trajectory points where change is calculated.
    """
    subdir_name = f"profile_change_vs_mean_p_{time_window_hrs}hr"
    output_dir = config_obj.PLOTS_OUTPUT_DIR / \
                 (config_obj.STATISTICAL_DISTRIBUTIONS_MOISTURE_SUBDIR if variable_name == "Moisture" else config_obj.STATISTICAL_DISTRIBUTIONS_TEMP_SUBDIR) / \
                 subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Plotting: Vertical Profile of {variable_name} {time_window_hrs}hr-Change vs. {time_window_hrs}hr-Mean Pressure")
    if master_df.empty: return

    # FIX: Handle column name changes based on time_window_hrs
    if time_window_hrs == config_obj.CHANGE_WINDOW_HOURS:
        if variable_name == "Moisture":
            change_col = 'dq_dt'
            plot_value_col = 'dq_dt_g_kg'
        else:
            change_col = 'dT_dt'
            plot_value_col = 'dT_dt_K'
    else:
        if variable_name == "Moisture":
            change_col = f'dq_dt_{time_window_hrs}hr'
            plot_value_col = f'dq_dt_g_kg_{time_window_hrs}hr'
        else:
            change_col = f'dT_dt_{time_window_hrs}hr'
            plot_value_col = f'dT_dt_K_{time_window_hrs}hr'

    mean_p_col = f'mean_pressure_{time_window_hrs}hr_window'

    if variable_name == "Moisture":
        master_df[plot_value_col] = master_df[change_col].dropna() * 1000
        unit_label = f"(g/kg per {time_window_hrs}hrs)"
        event_type_col = 'moisture_event_type' # Still based on 6hr for event classification
        pos_event_label, neg_event_label = 'Uptake', 'Release'
    elif variable_name == "Temperature":
        master_df[plot_value_col] = master_df[change_col].dropna()
        unit_label = f"(K per {time_window_hrs}hrs)"
        event_type_col = 'temp_event_type'
        pos_event_label, neg_event_label = 'Warming', 'Cooling'
    else: return

    # Use only points where the change and mean pressure are valid
    analysis_df = master_df.dropna(subset=[change_col, mean_p_col, plot_value_col, event_type_col])
    if analysis_df.empty:
        print(f"No valid data for {variable_name} {time_window_hrs}hr-change vs mean pressure plot.")
        return

    pos_events_df = analysis_df[analysis_df[event_type_col] == pos_event_label]
    neg_events_df = analysis_df[analysis_df[event_type_col] == neg_event_label]
    
    profile_bins = config_obj.VERTICAL_PROFILE_PRESSURE_BINS
    bin_centers = (profile_bins[:-1] + profile_bins[1:]) / 2
    
    avg_pos_change, _, _ = binned_statistic(pos_events_df[mean_p_col], pos_events_df[plot_value_col], statistic='mean', bins=profile_bins)
    avg_neg_change, _, _ = binned_statistic(neg_events_df[mean_p_col], neg_events_df[plot_value_col], statistic='mean', bins=profile_bins)

    fig, ax = plt.subplots(figsize=(8, 10))
    plot_made = False
    if not pos_events_df.empty and not np.all(np.isnan(avg_pos_change)):
        ax.plot(avg_pos_change, bin_centers, 'b-^', label=f'Avg. {pos_event_label} Events')
        plot_made = True
    if not neg_events_df.empty and not np.all(np.isnan(avg_neg_change)):
        ax.plot(avg_neg_change, bin_centers, 'r-v', label=f'Avg. {neg_event_label} Events')
        plot_made = True

    if not plot_made:
        print(f"No data to plot for {variable_name} {time_window_hrs}hr-change vs mean_pressure.")
        plt.close(fig); gc.collect(); return

    ax.set_xlabel(f'Average {time_window_hrs}hr {variable_name} Change {unit_label}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_ylabel(f'Mean Pressure of {time_window_hrs}hr Window (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_title(f'Vertical Profile of Avg {variable_name} Change\n(vs. Mean Pressure of {time_window_hrs}hr Window)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.invert_yaxis()
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.legend(fontsize=DEFAULT_FONT_SIZE * FONT_SCALE); ax.grid(True, linestyle=':', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    plt.savefig(output_dir / f"vertical_profile_{variable_name.lower()}_{time_window_hrs}hr_change_vs_mean_p.png", bbox_inches='tight', dpi=150)
    plt.close(fig); gc.collect()


# --- 3. Conditional Vertical Profiles (In Target) ---
def plot_conditional_vertical_profiles_in_target(master_df: pd.DataFrame, variable_name: str,
                                                 config_obj, time_window_hrs: int = 6):
    """
    Plots average dq/dt (or dT/dt) vs. current pressure for particles
    IN THE TARGET, conditioned by ascent/descent (using dp_dt_1hr).
    Uses the specified time_window_hrs for dq/dt or dT/dt.
    """
    output_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.TARGET_AREA_ANALYSIS_SUBDIR
    print(f"Plotting: Conditional Vertical Profiles of {variable_name} {time_window_hrs}hr-Change in Target")
    if master_df.empty: return

    # FIX: Handle column name changes based on time_window_hrs
    if time_window_hrs == config_obj.CHANGE_WINDOW_HOURS:
        if variable_name == "Moisture":
            change_col = 'dq_dt'
            plot_value_col = 'dq_dt_g_kg'
        else:
            change_col = 'dT_dt'
            plot_value_col = 'dT_dt_K'
    else:
        if variable_name == "Moisture":
            change_col = f'dq_dt_{time_window_hrs}hr'
            plot_value_col = f'dq_dt_g_kg_{time_window_hrs}hr'
        else:
            change_col = f'dT_dt_{time_window_hrs}hr'
            plot_value_col = f'dT_dt_K_{time_window_hrs}hr'

    if variable_name == "Moisture":
        if plot_value_col not in master_df.columns: master_df[plot_value_col] = master_df[change_col].dropna() * 1000
        unit_label = f"(g/kg per {time_window_hrs}hrs)"
    elif variable_name == "Temperature":
        if plot_value_col not in master_df.columns: master_df[plot_value_col] = master_df[change_col].dropna()
        unit_label = f"(K per {time_window_hrs}hrs)"
    else: return
    
    dp_col = 'dp_dt_1hr'
    if dp_col not in master_df.columns or change_col not in master_df.columns:
        print(f"Required columns ('{dp_col}', '{change_col}') not in master_df. Skipping conditional profiles.")
        return

    lat_min, lat_max = config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_min, lon_max = config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG

    in_target_df = master_df[
        (master_df['latitude'].between(lat_min, lat_max)) &
        (master_df['longitude'].between(lon_min, lon_max)) &
        (master_df['time_step'].isin(config_obj.EVENT_CORE_ANALYSIS_STEPS))
    ].copy()
    in_target_df.dropna(subset=['pressure', plot_value_col, dp_col], inplace=True)
    if in_target_df.empty: print(f"No valid in-target data for conditional profiles of {variable_name}."); return

    # dp_dt_1hr < 0 implies ascent (pressure decreasing)
    # Define a small threshold for 'neutral' vertical motion to avoid noise
    dp_threshold = 0.5 # hPa/hr, particles with |dp/dt| < threshold are 'neutral' vertically
    ascending_df = in_target_df[in_target_df[dp_col] < -dp_threshold]
    descending_df = in_target_df[in_target_df[dp_col] > dp_threshold]
    neutral_vert_df = in_target_df[in_target_df[dp_col].abs() <= dp_threshold]

    profile_bins = config_obj.VERTICAL_PROFILE_PRESSURE_BINS
    bin_centers = (profile_bins[:-1] + profile_bins[1:]) / 2

    avg_ascending, _, _ = binned_statistic(ascending_df['pressure'], ascending_df[plot_value_col], statistic='mean', bins=profile_bins)
    avg_descending, _, _ = binned_statistic(descending_df['pressure'], descending_df[plot_value_col], statistic='mean', bins=profile_bins)
    avg_neutral_vert, _, _ = binned_statistic(neutral_vert_df['pressure'], neutral_vert_df[plot_value_col], statistic='mean', bins=profile_bins)

    fig, ax = plt.subplots(figsize=(11, 10))
    plot_made = False
    if not ascending_df.empty and not np.all(np.isnan(avg_ascending)):
        ax.plot(avg_ascending, bin_centers, 'g-^', label=f'Ascending (dp/dt < {-dp_threshold:.1f} hPa/hr)')
        plot_made = True
    if not descending_df.empty and not np.all(np.isnan(avg_descending)):
        ax.plot(avg_descending, bin_centers, 'm-v', label=f'Descending (dp/dt > {dp_threshold:.1f} hPa/hr)')
        plot_made = True
    if not neutral_vert_df.empty and not np.all(np.isnan(avg_neutral_vert)):
        ax.plot(avg_neutral_vert, bin_centers, 'k-o', label=f'Near-Neutral Vertical Motion', alpha=0.7)
        plot_made = True
    
    if not plot_made: print(f"No data to plot for conditional vertical profile of {variable_name}."); plt.close(fig); gc.collect(); return

    ax.set_xlabel(f'Average {time_window_hrs}hr {variable_name} Change {unit_label}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_ylabel(f'Pressure (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_title(f'Cond. Vert. Profile of Avg {variable_name} {time_window_hrs}hr-Change in Target', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.invert_yaxis(); ax.axvline(0, color='grey', linestyle='--', alpha=0.7)
    ax.legend(fontsize=DEFAULT_FONT_SIZE * FONT_SCALE, loc='center left', bbox_to_anchor=(1, 0.5)); ax.grid(True, linestyle=':', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    plt.savefig(output_dir / f"cond_vert_profile_in_target_{variable_name.lower()}_{time_window_hrs}hr.png", bbox_inches='tight', dpi=150)
    plt.close(fig); gc.collect()


# --- 4. Difference Profiles (Initial vs. Final in Target) ---
def plot_in_target_change_profile(master_df: pd.DataFrame, relevant_ids: list,
                                  variable_name: str, config_obj):
    """
    Calculates change in q/T for particles from entry to exit of target box
    during EVENT_CORE_ANALYSIS_STEPS, and plots this change vs. avg pressure in target.
    """
    output_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.TARGET_AREA_ANALYSIS_SUBDIR
    print(f"Plotting: Profile of Net {variable_name} Change During Target Transit")
    if master_df.empty or not relevant_ids: return

    
    # --- DEFINE value_label and other variable-specifics HERE ---
    if variable_name == "Moisture":
        val_col = 'specific_humidity'
        unit_plot = 'g/kg'
        multiplier = 1000
        value_label = "Specific Humidity" # Define value_label
    elif variable_name == "Temperature":
        val_col = 'temperature'
        unit_plot = 'K'
        multiplier = 1
        value_label = "Temperature" # Define value_label
    else:
        print(f"Unknown variable_name '{variable_name}' for in-target change profile.")
        return
    # --- END DEFINITION ---
    

    lat_min, lat_max = config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_min, lon_max = config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG

    particle_target_transits = []
    for pid in tqdm(relevant_ids, desc=f"Analyzing {variable_name} change in target transit"):
        dfp = master_df[(master_df['particle_id'] == pid) & master_df['time_step'].isin(config_obj.EVENT_CORE_ANALYSIS_STEPS)].sort_values(by='time_step')
        if dfp.empty: continue

        dfp['in_target'] = (dfp['latitude'].between(lat_min, lat_max)) & (dfp['longitude'].between(lon_min, lon_max))
        # Identify blocks of time when particle is in target
        dfp['block'] = (dfp['in_target'].diff().ne(0)).cumsum()
        
        for block_num, block_df in dfp.groupby('block'):
            if block_df['in_target'].all() and len(block_df) > 1: # At least 2 steps in target for entry/exit
                entry_state = block_df.iloc[0]
                exit_state = block_df.iloc[-1]
                
                delta_val = (exit_state[val_col] - entry_state[val_col]) * multiplier
                avg_pressure_in_block = block_df['pressure'].mean()
                duration_in_block = len(block_df) # Hours

                if pd.notna(delta_val) and pd.notna(avg_pressure_in_block):
                    particle_target_transits.append({
                        'particle_id': pid,
                        'entry_time': entry_state['time_step'],
                        'exit_time': exit_state['time_step'],
                        'duration_hours': duration_in_block,
                        f'delta_{val_col}': delta_val,
                        'avg_pressure_in_target': avg_pressure_in_block
                    })
    
    if not particle_target_transits: print(f"No complete target transits found for {variable_name}."); return
    transit_df = pd.DataFrame(particle_target_transits)
    transit_df.to_csv(output_dir / f"particle_net_{variable_name.lower()}_change_in_target.csv", index=False)

    profile_bins = config_obj.VERTICAL_PROFILE_PRESSURE_BINS
    delta_val_col_name = f'delta_{val_col}'
    if delta_val_col_name not in transit_df.columns:
        print(f"Column '{delta_val_col_name}' not found in transit_df. Skipping plot.")
        return
    avg_delta_val, _, _ = binned_statistic(
        transit_df['avg_pressure_in_target'], transit_df[f'delta_{val_col}'],
        statistic='mean', bins=profile_bins
    )
    bin_centers = (profile_bins[:-1] + profile_bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(11, 10))
    plot_made = False # Flag to check if any data was plotted
    if not np.all(np.isnan(avg_delta_val)):
        ax.plot(avg_delta_val, bin_centers, 
                color='purple', marker='s', linestyle='-', # Specify separately
                label=f'Avg. Net {value_label} Change in Target')
        plot_made = True
    else: print(f"No data to plot for profile of {variable_name} change during target transit."); plt.close(fig); gc.collect(); return
        
    ax.set_xlabel(f'Net Change in {value_label}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_ylabel('Pressure (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_title(f'Profile of Net {value_label} Change During Target Transit ({unit_plot})', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.invert_yaxis(); ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.legend(fontsize=DEFAULT_FONT_SIZE * FONT_SCALE, loc='center left', bbox_to_anchor=(1, 0.5)); ax.grid(True, linestyle=':', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    plt.savefig(output_dir / f"profile_net_{variable_name.lower()}_change_in_target.png", bbox_inches='tight', dpi=150)
    plt.close(fig); gc.collect()


# --- 5. q_current vs. q_lagged Scatter Plot (In Target) ---
def plot_q_vs_q_lagged_in_target(master_df: pd.DataFrame, variable_name: str,
                                 config_obj, time_window_hrs_list: list = [2, 4, 6],
                                 analysis_type: str = 'centered'):
    """
    Scatter plot of particle values, supporting two analysis types:
    - 'lagged' (default): Plots current value vs. pre-calculated lagged value.
    - 'centered': Dynamically finds values before and after arrival in the target
                  to analyze moisture/temperature release.
    
    This function now iterates through a list of time windows and saves the plotting data to Excel.
    """
    output_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.TARGET_AREA_ANALYSIS_SUBDIR
    
    if master_df.empty: return

    for time_window_hrs in time_window_hrs_list:
        # --- Common Setup ---
        val_col = 'specific_humidity' if variable_name == "Moisture" else 'temperature'
        unit_plot = 'g/kg' if variable_name == "Moisture" else 'K'
        multiplier = 1000 if variable_name == "Moisture" else 1

        # --- Filter for particles in the target area and time ---
        lat_min, lat_max = config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
        lon_min, lon_max = config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG

        in_target_df = master_df[
            (master_df['latitude'].between(lat_min, lat_max)) &
            (master_df['longitude'].between(lon_min, lon_max)) &
            (master_df['time_step'].isin(config_obj.EVENT_CORE_ANALYSIS_STEPS))
        ].copy()

        if in_target_df.empty:
            print(f"No in-target data found for {variable_name} plot for any time window.")
            return # Exit if no in-target data at all

        # ==============================================================================
        # --- ANALYSIS-TYPE SPECIFIC LOGIC ---
        # ==============================================================================

        plot_df = pd.DataFrame()

        if analysis_type == 'lagged':
            print(f"Plotting (Lagged): {variable_name} Current vs. Lagged ({time_window_hrs}hr) in Target")
            
            if time_window_hrs == config_obj.CHANGE_WINDOW_HOURS:
                lagged_val_col = 'q_lagged' if variable_name == "Moisture" else 'T_lagged'
                change_col = 'dq_dt' if variable_name == "Moisture" else 'dT_dt'
            else:
                lagged_val_col = f'q_lagged_{time_window_hrs}hr' if variable_name == "Moisture" else f'T_lagged_{time_window_hrs}hr'
                change_col = f'dq_dt_{time_window_hrs}hr' if variable_name == "Moisture" else f'dT_dt_{time_window_hrs}hr'

            required_cols = [val_col, lagged_val_col, change_col]
            if not all(col in in_target_df.columns for col in required_cols):
                print(f"Required columns for 'lagged' analysis ({required_cols}) not found for {time_window_hrs}hr window. Skipping.")
                continue
            
            analysis_ready_df = in_target_df.dropna(subset=required_cols).copy()
            if analysis_ready_df.empty:
                print(f"No valid data after dropping NaNs for lagged plot ({time_window_hrs}hr window).")
                continue

            analysis_ready_df['x_vals'] = analysis_ready_df[lagged_val_col] * multiplier
            analysis_ready_df['y_vals'] = analysis_ready_df[val_col] * multiplier
            analysis_ready_df['color_vals'] = analysis_ready_df[change_col] * multiplier
            
            plot_df = analysis_ready_df[['particle_id', 'time_step', 'x_vals', 'y_vals', 'color_vals']]
            
            xlabel = f'Lagged {variable_name} ({time_window_hrs}hr ago)'
            ylabel = f'Current {variable_name}'
            plot_title = f'Current vs. Lagged {variable_name} in Target\n(Color by {time_window_hrs}hr Change)'

        elif analysis_type == 'centered':
            half_window = time_window_hrs / 2
            print(f"Plotting (Centered): {variable_name} Change from t-{half_window:.0f}hr to t+{half_window:.0f}hr in Target")
            
            events_df = in_target_df[['particle_id', 'time_step']].copy()
            events_df.rename(columns={'time_step': 'arrival_time'}, inplace=True)
            
            events_df['before_time'] = events_df['arrival_time'] - half_window
            
            data_before = pd.merge(
                events_df,
                master_df[['particle_id', 'time_step', val_col]],
                left_on=['particle_id', 'before_time'],
                right_on=['particle_id', 'time_step'],
                how='inner'
            )
            data_before.rename(columns={val_col: 'value_before'}, inplace=True)

            events_df['after_time'] = events_df['arrival_time'] + half_window
            
            data_after = pd.merge(
                events_df,
                master_df[['particle_id', 'time_step', val_col]],
                left_on=['particle_id', 'after_time'],
                right_on=['particle_id', 'time_step'],
                how='inner'
            )
            data_after.rename(columns={val_col: 'value_after'}, inplace=True)
            
            final_centered_df = pd.merge(
                data_before[['particle_id', 'arrival_time', 'value_before']],
                data_after[['particle_id', 'arrival_time', 'value_after']],
                on=['particle_id', 'arrival_time'],
                how='inner'
            )

            if final_centered_df.empty:
                print(f"No valid data pairs found for centered analysis ({time_window_hrs}hr window).")
                continue
            
            final_centered_df['x_vals'] = final_centered_df['value_before'] * multiplier
            final_centered_df['y_vals'] = final_centered_df['value_after'] * multiplier
            final_centered_df['color_vals'] = final_centered_df['y_vals'] - final_centered_df['x_vals']
            
            plot_df = final_centered_df.rename(columns={
                'value_before': f'value_before_{unit_plot.replace("/", "_")}',
                'value_after': f'value_after_{unit_plot.replace("/", "_")}'
            })

            xlabel = f'{variable_name} ({half_window:.0f}hr Before Arrival)'
            ylabel = f'{variable_name} ({half_window:.0f}hr After Arrival)'
            plot_title = f'Centered Analysis of {variable_name} in Target\n(Color by {time_window_hrs}hr Change around Arrival)'

        else:
            raise ValueError(f"Invalid analysis_type: '{analysis_type}'. Must be 'lagged' or 'centered'.")

        if plot_df.empty:
            continue # Skip to next time window if no data was generated

        # --- Save data to Excel ---
        excel_filename = output_dir / f"scatter_data_{variable_name.lower()}_{analysis_type}_{time_window_hrs}hr_in_target.xlsx"
        plot_df.to_excel(excel_filename, index=False)
        print(f"Saved plot data to {excel_filename}")

        # ==============================================================================
        # --- COMMON PLOTTING CODE ---
        # ==============================================================================
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(plot_df['x_vals'], plot_df['y_vals'], c=plot_df['color_vals'],
                        cmap=config_obj.CMAP_NET_MOISTURE_CHANGE if variable_name=="Moisture" else config_obj.CMAP_NET_TEMP_CHANGE,
                        s=10, alpha=0.5, edgecolor='grey', linewidth=0.2)
        
        min_val = min(plot_df['x_vals'].min(), plot_df['y_vals'].min())
        max_val = max(plot_df['x_vals'].max(), plot_df['y_vals'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='1:1 Line (No Change)')
        
        ax.set_xlabel(f'{xlabel} ({unit_plot})', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        ax.set_ylabel(f'{ylabel} ({unit_plot})', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        ax.set_title(plot_title, fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        
        ax.legend(fontsize=DEFAULT_FONT_SIZE * FONT_SCALE); ax.grid(True, linestyle=':')
        ax.axis('equal')
        ax.tick_params(axis='both', which='major', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)

        cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label(f'{time_window_hrs}hr Net Change in {variable_name} ({unit_plot})', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        cbar.ax.tick_params(labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        plt.tight_layout()
        
        plt.savefig(output_dir / f"scatter_{variable_name.lower()}_{analysis_type}_{time_window_hrs}hr_in_target.png", dpi=150)
        plt.close(fig); gc.collect()


# Existing functions (plot_vertical_profile_change, plot_histogram_event_magnitudes, plot_last_event_points,
# _plot_time_evolution_core, plot_time_evolution_multi_step, plot_time_evolution_full_event)
# should be kept as they are, as they serve different purposes or operate on the full trajectory.
# The main script will call them as needed.