# plotting_2d.py
"""
Module for generating 2D static plots and 2D animation frames.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable, get_cmap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from tqdm import tqdm
import gc
from scipy.stats import binned_statistic_2d
from multiprocessing import Pool

import config as cfg

FONT_SCALE = 2
DEFAULT_FONT_SIZE = 10

def _add_map_features_2d(ax, include_labels=True):
    """Helper to add standard map features to a 2D Cartopy ax."""
    ax.add_feature(cfeature.LAND.with_scale('110m'), zorder=0, facecolor='none') # No land color
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), zorder=1, linewidth=0.7, edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale('110m'), linestyle=':', zorder=1, linewidth=0.6, edgecolor='dimgray')
    if include_labels:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': DEFAULT_FONT_SIZE * FONT_SCALE}
        gl.ylabel_style = {'size': DEFAULT_FONT_SIZE * FONT_SCALE}
    else:
        ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='gray', alpha=0.5, linestyle='--')


def _get_plot_extent(df_subset_lon, df_subset_lat, fixed_extent_cfg, default_buffer_cfg):
    """Helper to determine plot extent (dynamic or fixed)."""
    if fixed_extent_cfg:
        return fixed_extent_cfg

    if df_subset_lon.empty or df_subset_lat.empty:
        return (-180, 180, -90, 90) # Default global if no data

    lon_min, lon_max = np.nanmin(df_subset_lon), np.nanmax(df_subset_lon)
    lat_min, lat_max = np.nanmin(df_subset_lat), np.nanmax(df_subset_lat)

    # Handle single-point cases or very small spread before buffer
    min_span = 1.0 # degrees
    if lon_max - lon_min < min_span:
        mid = (lon_max + lon_min) / 2
        lon_min, lon_max = mid - min_span / 2, mid + min_span / 2
    if lat_max - lat_min < min_span:
        mid = (lat_max + lat_min) / 2
        lat_min, lat_max = mid - min_span / 2, mid + min_span / 2
        
    plot_lon_min = max(-180, lon_min - default_buffer_cfg)
    plot_lon_max = min(180, lon_max + default_buffer_cfg)
    plot_lat_min = max(-90, lat_min - default_buffer_cfg)
    plot_lat_max = min(90, lat_max + default_buffer_cfg)
    
    if plot_lon_max <= plot_lon_min: plot_lon_max = plot_lon_min + default_buffer_cfg
    if plot_lat_max <= plot_lat_min: plot_lat_max = plot_lat_min + default_buffer_cfg
    return (plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)

def _add_target_area_2d(ax, config_obj):
    """Adds the target area box to a 2D plot."""
    target_lons_box = [config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG,
                       config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG,
                       config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG,
                       config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG,
                       config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG]
    target_lats_box = [config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG,
                       config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG,
                       config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG,
                       config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG,
                       config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG]
    ax.plot(target_lons_box, target_lats_box, color=config_obj.TARGET_AREA_PLOT_COLOR,
            linewidth=1.5, transform=ccrs.Geodetic(), label='Target Area', zorder=5, alpha=0.8)


def plot_aggregate_map_generic(master_df: pd.DataFrame, event_filter: str,
                               value_column_base: str, variable_name: str,
                               config_obj, is_frequency: bool = False, output_suffix_ext: str = ""):
    """
    Generic function for plotting 2D aggregate maps (net change or frequency).

    Args:
        master_df: DataFrame with all analyzed particle histories.
        event_filter: "Uptake", "Release", "Warming", "Cooling", or "AllEvents" (for net change).
        value_column_base: Base name of the column for values (e.g., 'dq_dt', 'dT_dt').
                           If is_frequency, this is ignored for calculation but used for naming.
        variable_name: "Moisture" or "Temperature" (for titles, labels, colormaps).
        config_obj: The configuration module (cfg).
        is_frequency: If True, plots frequency of events. If False, plots sum of value_column.
        output_suffix_ext: Additional string for the output filename.
    """
    output_dir = config_obj.PLOTS_OUTPUT_DIR / (config_obj.AGGREGATE_MAPS_MOISTURE_SUBDIR if variable_name == "Moisture" else config_obj.AGGREGATE_MAPS_TEMP_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    stat_type_label = "Frequency" if is_frequency else f"Sum of {value_column_base}"
    print(f"Plotting: Aggregate 2D Map - {variable_name} {event_filter} ({stat_type_label})")

    if master_df.empty:
        print(f"Master DataFrame is empty. Skipping aggregate plot for {variable_name} {event_filter}.")
        return

    # Determine event type column and specific event types for filtering
    if variable_name == "Moisture":
        event_type_col = 'moisture_event_type'
        actual_value_col = 'dq_dt' # in kg/kg or K
        plot_value_col = 'plot_value_g_kg' # for g/kg or K
        cbar_unit_label = f"(g/kg per {config_obj.CHANGE_WINDOW_HOURS}hrs)" if not is_frequency else "Count"
        cmap_positive = config_obj.CMAP_UPTAKE_AGG
        cmap_negative = config_obj.CMAP_RELEASE_AGG
        cmap_net = config_obj.CMAP_NET_MOISTURE_CHANGE
    elif variable_name == "Temperature":
        event_type_col = 'temp_event_type'
        actual_value_col = 'dT_dt' # in K
        plot_value_col = 'plot_value_K' # for K
        cbar_unit_label = f"(K per {config_obj.CHANGE_WINDOW_HOURS}hrs)" if not is_frequency else "Count"
        cmap_positive = config_obj.CMAP_WARMING_AGG
        cmap_negative = config_obj.CMAP_COOLING_AGG
        cmap_net = config_obj.CMAP_NET_TEMP_CHANGE
    else:
        raise ValueError(f"Unknown variable_name: {variable_name}")

    # Filter for relevant events
    if event_filter == "AllEvents": # For net change
        event_df = master_df[master_df[event_type_col] != 'Neutral'].copy()
        title_prefix = f"Net {variable_name} Change"
        cmap_to_use = cmap_net
    else: # Specific event type (Uptake, Release, Warming, Cooling)
        event_df = master_df[master_df[event_type_col] == event_filter].copy()
        title_prefix = f"Aggregate {variable_name} {event_filter}"
        if event_filter in ["Uptake", "Warming"]: cmap_to_use = cmap_positive
        else: cmap_to_use = cmap_negative
    
    if event_df.empty:
        print(f"No '{event_filter}' events found for {variable_name}. Skipping plot.")
        return

    # Prepare value column for plotting
    if not is_frequency:
        if variable_name == "Moisture":
            event_df[plot_value_col] = event_df[actual_value_col] * 1000 # Convert q to g/kg
        else: # Temperature
            event_df[plot_value_col] = event_df[actual_value_col] # K remains K
    else: # For frequency, we count occurrences
        event_df[plot_value_col] = 1 # Dummy value for counting

    # Drop NaNs from essential columns for binning
    event_df.dropna(subset=['longitude', 'latitude', plot_value_col], inplace=True)
    if event_df.empty:
        print(f"No valid data points after NaN drop for {variable_name} {event_filter}. Skipping plot.")
        return

    lon_min, lon_max, lat_min, lat_max = _get_plot_extent(
        event_df['longitude'], event_df['latitude'],
        config_obj.FIXED_PLOT_EXTENT_2D, config_obj.DEFAULT_PLOT_BUFFER_DEG
    )
    lon_bins = np.arange(lon_min, lon_max + config_obj.AGGREGATE_MAP_GRID_RESOLUTION_DEG, config_obj.AGGREGATE_MAP_GRID_RESOLUTION_DEG)
    lat_bins = np.arange(lat_min, lat_max + config_obj.AGGREGATE_MAP_GRID_RESOLUTION_DEG, config_obj.AGGREGATE_MAP_GRID_RESOLUTION_DEG)

    if len(lon_bins) < 2 or len(lat_bins) < 2:
        print(f"Not enough bins for {variable_name} {event_filter} heatmap. Skipping plot.")
        return

    statistic_binned, x_edge, y_edge, _ = binned_statistic_2d(
        event_df['longitude'], event_df['latitude'], event_df[plot_value_col],
        statistic='sum' if not is_frequency else 'count',
        bins=[lon_bins, lat_bins]
    )
    statistic_masked = np.ma.masked_where(np.isnan(statistic_binned) | (statistic_binned == 0), statistic_binned)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_title(f'{title_prefix} ({stat_type_label})', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    _add_map_features_2d(ax)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    _add_target_area_2d(ax, config_obj) # Requirement 4

    cmap_obj = get_cmap(cmap_to_use)
    norm = None
    valid_stats = np.ma.compressed(statistic_masked)

    if len(valid_stats) > 0 and np.any(valid_stats): # Ensure there are non-zero values
        if event_filter == "AllEvents": # Net change, use symmetric scale
            cmin_p = np.percentile(valid_stats[valid_stats!=0], config_obj.AGGREGATE_COLOR_PERCENTILES[0])
            cmax_p = np.percentile(valid_stats[valid_stats!=0], config_obj.AGGREGATE_COLOR_PERCENTILES[1])
            abs_val = max(abs(cmin_p), abs(cmax_p), 1e-9) # Avoid zero range
            norm = Normalize(vmin=-abs_val, vmax=abs_val)
        elif is_frequency:
            norm = Normalize(vmin=1, vmax=max(1, np.nanmax(valid_stats))) # Freq starts at 1
        else: # Magnitude of uptake/release or warming/cooling
            # For single-sided events (Uptake, Warming), usually min is positive or zero
            # For Release, Cooling, values are negative, so sum is negative
            data_min = np.nanmin(valid_stats[valid_stats!=0]) if np.any(valid_stats[valid_stats!=0]) else 0
            data_max = np.nanmax(valid_stats[valid_stats!=0]) if np.any(valid_stats[valid_stats!=0]) else 1
            if data_min >= data_max: data_max = data_min + 1e-9 # Ensure range
            norm = Normalize(vmin=data_min, vmax=data_max)
    else: # Fallback norm if no valid data or all zeros
        norm = Normalize(vmin=0, vmax=1) if not (event_filter == "AllEvents") else Normalize(vmin=-1, vmax=1)

    mesh = ax.pcolormesh(x_edge, y_edge, statistic_masked.T, cmap=cmap_obj, norm=norm,
                         transform=ccrs.PlateCarree(), zorder=3, alpha=0.85, shading='auto')
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.6, 
                        extend='both' if norm and norm.clip else ('min' if norm.vmin > np.nanmin(valid_stats) else 'max' if norm.vmax < np.nanmax(valid_stats) else 'neither'))
    cbar.set_label(f'{stat_type_label} {cbar_unit_label}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    cbar.ax.tick_params(labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    
    ax.legend(loc='upper right', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)

    filename = f"aggregate_{variable_name.lower()}_{output_suffix_ext}.png"
    plt.savefig(output_dir / filename, bbox_inches='tight', dpi=150)
    plt.close(fig); gc.collect()


def plot_composite_trajectory_density(master_df: pd.DataFrame, config_obj):
    """Plots all trajectories with density, event locations, and target area."""
    output_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.AGGREGATE_MAPS_MOISTURE_SUBDIR # Or a general one
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Plotting: Composite Trajectory Density Map")

    if master_df.empty:
        print("Master DataFrame empty. Skipping composite trajectory plot.")
        return

    fig = plt.figure(figsize=(16, 10)) # Wider for better legend placement
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_title('Composite Trajectories of Target-Reaching Particles & Event Locations', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    _add_map_features_2d(ax)
    
    plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max = _get_plot_extent(
        master_df['longitude'], master_df['latitude'],
        config_obj.FIXED_PLOT_EXTENT_2D, config_obj.DEFAULT_PLOT_BUFFER_DEG
    )
    ax.set_extent([plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max], crs=ccrs.PlateCarree())

    # Plot trajectories
    for particle_id, group in tqdm(master_df.groupby('particle_id'), desc="Plotting composite trajectories", total=master_df['particle_id'].nunique()):
        if len(group) > 1:
            ax.plot(group['longitude'], group['latitude'], '-', color='gray', alpha=0.05,
                    transform=ccrs.Geodetic(), linewidth=0.4, zorder=3)

    # Event locations (Moisture)
    uptake_events = master_df[master_df['moisture_event_type'] == 'Uptake']
    release_events = master_df[master_df['moisture_event_type'] == 'Release']
    if not uptake_events.empty:
        ax.scatter(uptake_events['longitude'], uptake_events['latitude'],
                   s=2, color='blue', marker='o', alpha=0.15,
                   transform=ccrs.Geodetic(), zorder=4, label='_nolegend_')
    if not release_events.empty:
        ax.scatter(release_events['longitude'], release_events['latitude'],
                   s=2, color='red', marker='o', alpha=0.15,
                   transform=ccrs.Geodetic(), zorder=4, label='_nolegend_')

    # Event locations (Temperature) - optional, can make plot too busy
    # warming_events = master_df[master_df['temp_event_type'] == 'Warming']
    # cooling_events = master_df[master_df['temp_event_type'] == 'Cooling']
    # if not warming_events.empty:
    #     ax.scatter(warming_events['longitude'], warming_events['latitude'],
    #                s=2, color='orange', marker='^', alpha=0.1, transform=ccrs.Geodetic(), zorder=4, label='_nolegend_')
    # if not cooling_events.empty:
    #     ax.scatter(cooling_events['longitude'], cooling_events['latitude'],
    #                s=2, color='purple', marker='v', alpha=0.1, transform=ccrs.Geodetic(), zorder=4, label='_nolegend_')

    _add_target_area_2d(ax, config_obj) # Requirement 4

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='Moisture Uptake', alpha=0.7, linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Moisture Release', alpha=0.7, linestyle='None'),
        # Optional temperature event legend items
        # plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=6, label='Warming', alpha=0.7, linestyle='None'),
        # plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='purple', markersize=6, label='Cooling', alpha=0.7, linestyle='None'),
        plt.Line2D([0], [0], color=config_obj.TARGET_AREA_PLOT_COLOR, linewidth=1.5, label='Target Area', linestyle='-')
    ]
    ax.legend(handles=handles, loc='best', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE, facecolor='white', framealpha=0.8)

    plt.savefig(output_dir / "composite_trajectory_density_events.png", bbox_inches='tight', dpi=200)
    plt.close(fig); gc.collect()


def _plot_individual_2d_trajectory_detailed_worker(args):
    """Worker for plotting individual 2D trajectories."""
    # Unpack more arguments
    particle_id, particle_df, variable_name, \
    output_subdir_name, sim_start_dt, dt_plot_format, target_plot_color, \
    fixed_extent_2d, default_buffer, cmap_pressure_name, pressure_bins_cat, \
    target_lon, target_lat, target_half_width = args

    output_dir = cfg.PLOTS_OUTPUT_DIR / output_subdir_name / variable_name.lower() # Use cfg for PLOTS_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)


    if particle_df.empty or len(particle_df) < 2:
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    start_datetime_particle = sim_start_dt + pd.Timedelta(hours=particle_df['time_step'].min())
    end_datetime_particle = sim_start_dt + pd.Timedelta(hours=particle_df['time_step'].max())

    title = (
             f"Particle ID: {int(particle_id)} - {variable_name} Trajectory\n"
             f"{start_datetime_particle.strftime(dt_plot_format)} to "
             f"{end_datetime_particle.strftime(dt_plot_format)}")
    ax.set_title(title, fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    _add_map_features_2d(ax)
    _add_target_area_2d_worker(ax, target_lon, target_lat, target_half_width, target_plot_color)


    plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max = _get_plot_extent(
        particle_df['longitude'], particle_df['latitude'],
        fixed_extent_2d, default_buffer
    )
    ax.set_extent([plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max], crs=ccrs.PlateCarree())

    points = np.array([particle_df['longitude'], particle_df['latitude']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    pressures_for_segments = particle_df['pressure'].iloc[:-1].values
    
    cmap_pressure_obj = get_cmap(cmap_pressure_name)
    # Ensure BoundaryNorm is imported at the top of the file
    # from matplotlib.colors import BoundaryNorm 
    norm_pressure_obj = BoundaryNorm(pressure_bins_cat, cmap_pressure_obj.N)


    lc = LineCollection(segments, cmap=cmap_pressure_obj, norm=norm_pressure_obj,
                        transform=ccrs.Geodetic(), zorder=3, linewidth=1.5)
    lc.set_array(pressures_for_segments)
    line_collection = ax.add_collection(lc)

    # --- CORRECTED SECTION FOR EVENT DEFINITIONS ---
    if variable_name == "Moisture":
        event_col = 'moisture_event_type'
        pos_event_label = 'Uptake'
        neg_event_label = 'Release'
        pos_color = 'blue'
        neg_color = 'red'
        pos_marker = '^'
        neg_marker = 'v'
    elif variable_name == "Temperature": # Use elif for clarity
        event_col = 'temp_event_type'
        pos_event_label = 'Warming'
        neg_event_label = 'Cooling'
        pos_color = 'orange'  # Define colors here
        neg_color = 'purple' # Define colors here
        pos_marker = '^'
        neg_marker = 'v'
    else:
        # Handle unknown variable_name if necessary, or assume it's one of the two
        print(f"Warning: Unknown variable_name '{variable_name}' in _plot_individual_2d_trajectory_detailed_worker. No event markers will be plotted.")
        event_col = None # Prevent further errors
    # --- END OF CORRECTION ---

    if event_col: # Proceed only if event_col is set
        pos_df = particle_df[particle_df[event_col] == pos_event_label]
        neg_df = particle_df[particle_df[event_col] == neg_event_label]

        if not pos_df.empty:
            ax.scatter(pos_df['longitude'], pos_df['latitude'], s=50, color=pos_color, marker=pos_marker,
                       edgecolor='black', label=pos_event_label, transform=ccrs.Geodetic(), zorder=5, alpha=0.8)
        if not neg_df.empty:
            ax.scatter(neg_df['longitude'], neg_df['latitude'], s=50, color=neg_color, marker=neg_marker,
                       edgecolor='black', label=neg_event_label, transform=ccrs.Geodetic(), zorder=5, alpha=0.8)

    ax.plot(particle_df['longitude'].iloc[0], particle_df['latitude'].iloc[0], 'go', markersize=7, transform=ccrs.Geodetic(), label='Start', zorder=6)
    ax.plot(particle_df['longitude'].iloc[-1], particle_df['latitude'].iloc[-1], 'ks', markersize=7, transform=ccrs.Geodetic(), label='End', zorder=6)

    cbar_pressure = plt.colorbar(line_collection, ax=ax, orientation='vertical', pad=0.02, shrink=0.6, aspect=15)
    cbar_pressure.set_label('Pressure (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    cbar_pressure.ax.tick_params(labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    cbar_pressure.set_ticks(pressure_bins_cat[::2])
    
    ax.legend(loc='best', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    plt.savefig(output_dir / f"particle_{int(particle_id)}_{variable_name.lower()}_traj.png", bbox_inches='tight', dpi=150)
    plt.close(fig); gc.collect()


# NEW HELPER that doesn't directly take config_obj
def _add_target_area_2d_worker(ax, target_lon, target_lat, target_half_width, target_plot_color):
    """Adds the target area box to a 2D plot using passed parameters."""
    target_lons_box = [target_lon - target_half_width, target_lon + target_half_width,
                       target_lon + target_half_width, target_lon - target_half_width,
                       target_lon - target_half_width]
    target_lats_box = [target_lat - target_half_width, target_lat - target_half_width,
                       target_lat + target_half_width, target_lat + target_half_width,
                       target_lat - target_half_width]
    ax.plot(target_lons_box, target_lats_box, color=target_plot_color,
            linewidth=1.5, transform=ccrs.Geodetic(), label='Target Area', zorder=5, alpha=0.8)


# MODIFIED CALLING FUNCTION
def plot_selected_individual_2d_trajectories(master_df: pd.DataFrame, relevant_ids: list, variable_name: str, config_obj): # Keep config_obj here
    """Plots detailed 2D trajectories for a selection of particles."""
    print(f"Plotting: Selected Individual 2D Trajectories for {variable_name}")
    if master_df.empty or not relevant_ids:
        print(f"No data or relevant IDs for individual {variable_name} trajectory plots. Skipping.")
        return

    ids_to_plot = relevant_ids
    if config_obj.MAX_INDIVIDUAL_TRAJECTORIES_TO_PLOT is not None and len(relevant_ids) > config_obj.MAX_INDIVIDUAL_TRAJECTORIES_TO_PLOT:
        print(f"Plotting a subset of {config_obj.MAX_INDIVIDUAL_TRAJECTORIES_TO_PLOT} trajectories.")
        ids_to_plot = relevant_ids[:config_obj.MAX_INDIVIDUAL_TRAJECTORIES_TO_PLOT]

    tasks = []
    for pid in ids_to_plot:
        particle_df = master_df[master_df['particle_id'] == pid]
        if not particle_df.empty:
            # Pass specific config values needed by the worker
            task_args = (
                pid, particle_df.copy(), variable_name,
                config_obj.INDIVIDUAL_TRAJ_PLOTS_SUBDIR, # Directory name part
                config_obj.SIMULATION_START_DATETIME,
                config_obj.DATETIME_PLOT_FORMAT,
                config_obj.TARGET_AREA_PLOT_COLOR,
                config_obj.FIXED_PLOT_EXTENT_2D,
                config_obj.DEFAULT_PLOT_BUFFER_DEG,
                config_obj.CMAP_PRESSURE, # Name of colormap
                config_obj.PRESSURE_BINS_CATEGORICAL, # Actual bins
                config_obj.TARGET_LON_CENTER,
                config_obj.TARGET_LAT_CENTER,
                config_obj.TARGET_BOX_HALF_WIDTH_DEG
            )
            tasks.append(task_args)
    
    if tasks:
        if config_obj.NUM_WORKERS > 1 and len(tasks) > 1:
            with Pool(processes=min(config_obj.NUM_WORKERS, len(tasks))) as pool:
                list(tqdm(pool.imap_unordered(_plot_individual_2d_trajectory_detailed_worker, tasks),
                          total=len(tasks), desc=f"Plotting ind. {variable_name} traj"))
        else:
            for task_args in tqdm(tasks, desc=f"Plotting ind. {variable_name} traj (Serial)"):
                _plot_individual_2d_trajectory_detailed_worker(task_args)
                
    print(f"Individual 2D {variable_name} trajectory plots saved.")


def generate_2d_snapshot_frame(
    hour: int,
    df_at_hour: pd.DataFrame,
    output_dir_for_frames: Path,
    fixed_extent_for_anim=None,
    q_norm_for_anim=None,
    t_norm_for_anim=None,
    add_colorbar_to_this_frame=False,
    plot_variable="specific_humidity",
    simulation_start_dt_arg=None,
    datetime_plot_format="%Y-%m-%d %H:%M",
    target_area_color='darkgreen',
    target_lon=0, target_lat=0, target_half_width=0,
    default_plot_buffer=5.0,
    animation_dpi=100
):
    # --- DEBUG PRINT ---
    print(f"Worker for H{hour}, plot_var='{plot_variable}': received simulation_start_dt_arg = {simulation_start_dt_arg} (type: {type(simulation_start_dt_arg)})")
    # --- END DEBUG PRINT ---
    """
    Generates a single 2D snapshot frame for animation. Requirement 5.
    Plots particles colored by the specified plot_variable.
    Now takes specific config values instead of the whole cfg module.
    """
    # output_dir_for_frames is already specific, e.g., .../2D_frames/specific_humidity
    # output_dir_for_frames.mkdir(parents=True, exist_ok=True) # Already done in animations.py

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Ensure simulation_start_dt_arg is a pandas Timestamp
    if isinstance(simulation_start_dt_arg, str):
        current_simulation_start_dt = pd.Timestamp(simulation_start_dt_arg) # This will parse ISO string
    elif isinstance(simulation_start_dt_arg, pd.Timestamp):
        current_simulation_start_dt = simulation_start_dt_arg
    else:
        # Fallback or raise error if type is unexpected
        print(f"Warning: simulation_start_dt_arg is of unexpected type {type(simulation_start_dt_arg)}. Using current time as fallback.")
        current_simulation_start_dt = pd.Timestamp.now() # Fallback, not ideal
        
    current_datetime = current_simulation_start_dt + pd.Timedelta(hours=hour)
    title_var_name = "Specific Humidity" if plot_variable == "specific_humidity" else "Temperature"
    #ax.set_title(f'{title_var_name} at {current_datetime.strftime(datetime_plot_format)} (Hour {hour:03d})', fontsize=10)
    ax.set_title(f'{title_var_name} at {current_datetime.strftime(datetime_plot_format)}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    # _add_map_features_2d might need to be refactored if it used cfg directly.
    # Assuming it's simple or adapted.
    _add_map_features_2d(ax, include_labels=True) 
    _add_target_area_2d_worker(ax, target_lon, target_lat, target_half_width, target_area_color)

    current_extent = _get_plot_extent(df_at_hour['longitude'], df_at_hour['latitude'], 
                                      fixed_extent_for_anim, default_plot_buffer)
    ax.set_extent(current_extent, crs=ccrs.PlateCarree())

    norm_obj_to_use = None
    cmap_name_to_use = None

    if plot_variable == "specific_humidity":
        values_to_plot = df_at_hour['specific_humidity'].dropna() * 1000
        cmap_name_to_use = cfg.CMAP_SPECIFIC_HUMIDITY # Ok to use cfg here if it's just simple value lookup
        norm_obj_to_use = q_norm_for_anim
        cbar_label_str = 'Specific Humidity (g/kg)'
    elif plot_variable == "temperature":
        values_to_plot = df_at_hour['temperature'].dropna()
        cmap_name_to_use = cfg.CMAP_TEMPERATURE
        norm_obj_to_use = t_norm_for_anim
        cbar_label_str = 'Temperature (K)'
    else:
        raise ValueError(f"Unknown plot_variable for 2D snapshot: {plot_variable}")

    lons = df_at_hour.loc[values_to_plot.index, 'longitude']
    lats = df_at_hour.loc[values_to_plot.index, 'latitude']

    if not values_to_plot.empty and cmap_name_to_use or 1:
        cmap_obj = get_cmap(cmap_name_to_use)
        sc = ax.scatter(lons, lats, c=values_to_plot, cmap=cmap_obj, norm=norm_obj_to_use,
                        s=12, alpha=0.7, transform=ccrs.Geodetic(), zorder=3,
                        edgecolor='dimgrey', linewidth=0.2)
        
        if add_colorbar_to_this_frame and norm_obj_to_use or 1:
            sm = ScalarMappable(cmap=cmap_obj, norm=norm_obj_to_use)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.6, aspect=20,
                                extend='both' if norm_obj_to_use.clip else 'neither')
            cbar.set_label(cbar_label_str, fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
            cbar.ax.tick_params(labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)

    filename_suffix = "q" if plot_variable == "specific_humidity" else "temp"
    frame_path = output_dir_for_frames / f"snapshot_2d_{filename_suffix}_hour_{hour:03d}.png" # Use passed output_dir_for_frames
    plt.savefig(frame_path, bbox_inches='tight', dpi=animation_dpi)
    plt.close(fig); gc.collect()
    return frame_path
