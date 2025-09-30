# tracks_analysis.py
"""
Module for performing trajectory analysis:
- Identifies particles from master_df reaching a target at specific times.
- Extracts their initial and final states (including q & T).
- Plots initial positions and full trajectories of these selected particles.
- Plots changes in q and T along these trajectories, extending post-arrival.
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable, get_cmap # Ensure get_cmap is available
from multiprocessing import Pool # For Pool in run_tracks_analysis_for_all_steps
import os
from tqdm import tqdm
import gc
from matplotlib.colors import Normalize, BoundaryNorm # BoundaryNorm for pressure-colored trajectories
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.dates as mdates

# Import global configuration
import config as cfg # Used for some global settings like output paths, datetime formats

# Define a font scaling factor
FONT_SCALE = 2
DEFAULT_FONT_SIZE = 10 # A reasonable base font size

# Colormap and Norm functions (same as before)
def _create_pressure_cmap_norm(cmap_name_str, pressure_bins_array):
    # ... (no changes) ...
    base_cmap_obj = get_cmap(cmap_name_str)
    num_intervals = len(pressure_bins_array) - 1
    if num_intervals <= 0:
        pressure_colors_obj = base_cmap_obj(np.array([0.5]))
        return ListedColormap(pressure_colors_obj), BoundaryNorm(pressure_bins_array, 1)
    pressure_colors_obj = base_cmap_obj(np.linspace(0, 1, num_intervals))
    custom_cmap = ListedColormap(pressure_colors_obj)
    custom_norm = BoundaryNorm(pressure_bins_array, custom_cmap.N)
    return custom_cmap, custom_norm

def _validate_data_tracks(df): return df

def _safe_plotting_boundaries_tracks_worker(longitudes, latitudes, fixed_plot_extent_2d_val, default_plot_buffer_val):
    # ... (no changes) ...
    if fixed_plot_extent_2d_val: return fixed_plot_extent_2d_val
    if not hasattr(longitudes, '__len__') or len(longitudes) == 0 or not hasattr(latitudes, '__len__') or len(latitudes) == 0: return (-180, 180, -90, 90)
    buffer = default_plot_buffer_val
    lon_min, lon_max = np.nanmin(longitudes), np.nanmax(longitudes)
    lat_min, lat_max = np.nanmin(latitudes), np.nanmax(latitudes)
    if lon_min == lon_max: lon_min -= buffer; lon_max += buffer
    if lat_min == lat_max: lat_min -= buffer; lat_max += buffer
    plot_lon_min = max(-180, lon_min - buffer); plot_lon_max = min(180, lon_max + buffer)
    plot_lat_min = max(-90, lat_min - buffer); plot_lat_max = min(90, lat_max + buffer)
    if plot_lon_max <= plot_lon_min: plot_lon_max = plot_lon_min + buffer
    if plot_lat_max <= plot_lat_min: plot_lat_max = plot_lat_min + buffer
    return (plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)

def _add_map_features_tracks(ax):
    ax.add_feature(cfeature.LAND.with_scale('110m'), zorder=0, edgecolor='black', facecolor='#D3D3D3')
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), zorder=1, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale('110m'), linestyle=':', zorder=1, linewidth=0.6)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False, y_inline=False, zorder=2)
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': DEFAULT_FONT_SIZE * FONT_SCALE}; gl.ylabel_style = {'size': DEFAULT_FONT_SIZE * FONT_SCALE}
    ax.set_xlabel('LON', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_ylabel('LAT', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)

def _add_target_area_2d_tracks(ax, target_lon, target_lat, target_half_width, target_plot_color):
    target_lons_box = [target_lon - target_half_width, target_lon + target_half_width,
                       target_lon + target_half_width, target_lon - target_half_width,
                       target_lon - target_half_width]
    target_lats_box = [target_lat - target_half_width, target_lat - target_half_width,
                       target_lat + target_half_width, target_lat + target_half_width,
                       target_lat - target_half_width]
    ax.plot(target_lons_box, target_lats_box, color=target_plot_color,
            linewidth=1.5, transform=ccrs.Geodetic(), label='Target Area', zorder=5, alpha=0.8)

def process_track_step_from_master(
    target_step_hour: int,
    master_df_for_worker: pd.DataFrame,
    tracks_output_dir_str: str,
    simulation_start_datetime_iso: str,
    datetime_plot_format_str: str, # cfg.DATETIME_PLOT_FORMAT
    target_lat_center_val: float,
    target_lon_center_val: float,
    target_box_half_width_deg_val: float,
    target_area_plot_color_val: str,
    fixed_plot_extent_2d_val: tuple | None,
    default_plot_buffer_deg_val: float,
    pressure_bins_categorical_array: np.ndarray,
    cmap_pressure_name_str: str,
    initial_pos_reference_hour: int,
    # New argument for tracking window after event for q/T change plots
    track_after_target_hours_val: int # from cfg.TRACK_HISTORY_WINDOW_AFTER_EVENT_HOURS
):
    try:
        tracks_output_dir = Path(tracks_output_dir_str)
        data_subdir = tracks_output_dir / "data_from_master"
        plots_subdir = tracks_output_dir / "plots_from_master"
        change_plots_subdir = plots_subdir / "change_along_trajectory"
        data_subdir.mkdir(parents=True, exist_ok=True)
        plots_subdir.mkdir(parents=True, exist_ok=True)
        change_plots_subdir.mkdir(parents=True, exist_ok=True)

        simulation_start_datetime_obj = pd.Timestamp(simulation_start_datetime_iso)
        target_datetime_obj = simulation_start_datetime_obj + pd.Timedelta(hours=target_step_hour)
        
        print(f'Tracks Analysis (from master_df): Target {target_datetime_obj.strftime(datetime_plot_format_str)} (Hour {target_step_hour})')

        cmap_tracks_pressure_obj, norm_tracks_pressure_obj = _create_pressure_cmap_norm(
            cmap_pressure_name_str, pressure_bins_categorical_array
        )

        df_at_target_hour = master_df_for_worker[master_df_for_worker['time_step'] == target_step_hour]
        if df_at_target_hour.empty: return f"NoDataAtTargetHour_H{target_step_hour}"

        lat_min_target = target_lat_center_val - target_box_half_width_deg_val; lat_max_target = target_lat_center_val + target_box_half_width_deg_val
        lon_min_target = target_lon_center_val - target_box_half_width_deg_val; lon_max_target = target_lon_center_val + target_box_half_width_deg_val
        
        mask = ( df_at_target_hour['latitude'].between(lat_min_target, lat_max_target) &
                 df_at_target_hour['longitude'].between(lon_min_target, lon_max_target) )
        
        particles_in_target_at_step = df_at_target_hour.loc[mask, 'particle_id'].unique()
        if len(particles_in_target_at_step) == 0:
            print(f"Tracks Analysis: No particles from master_df in target region at hour {target_step_hour}")
            return f"NoParticlesInBox_H{target_step_hour}"

        # ... (Initial and Final states CSV saving logic - no changes here, already includes q/T) ...
        final_states_df = master_df_for_worker[ (master_df_for_worker['particle_id'].isin(particles_in_target_at_step)) & (master_df_for_worker['time_step'] == target_step_hour) ].copy()
        final_states_df.rename(columns={'latitude': 'latitude_final', 'longitude': 'longitude_final', 'pressure': 'pressure_final', 'specific_humidity': 'specific_humidity_final', 'temperature': 'temperature_final' }, inplace=True)
        initial_states_df = master_df_for_worker[ (master_df_for_worker['particle_id'].isin(particles_in_target_at_step)) & (master_df_for_worker['time_step'] == initial_pos_reference_hour) ].copy()
        initial_states_df.rename(columns={'latitude': 'latitude_initial', 'longitude': 'longitude_initial', 'pressure': 'pressure_initial', 'specific_humidity': 'specific_humidity_initial', 'temperature': 'temperature_initial' }, inplace=True)
        cols_to_keep_final = ['particle_id', 'latitude_final', 'longitude_final', 'pressure_final', 'specific_humidity_final', 'temperature_final']
        cols_to_keep_initial = ['particle_id', 'latitude_initial', 'longitude_initial', 'pressure_initial', 'specific_humidity_initial', 'temperature_initial']
        merged_df = pd.merge(initial_states_df[cols_to_keep_initial], final_states_df[cols_to_keep_final], on='particle_id', how='inner')
        if merged_df.empty: return f"NoMergedData_H{target_step_hour}"
        output_parquet_file = data_subdir / f'target_particles_step_{target_step_hour:04d}_analyzed.parquet'
        merged_df.to_parquet(output_parquet_file, index=False)


        # --- Collect Trajectories ---
        # For main trajectory plot: initial_pos_reference_hour up to target_step_hour
        trajectories_for_map = {}
        particle_histories_for_map_plot = master_df_for_worker[
            (master_df_for_worker['particle_id'].isin(merged_df['particle_id'])) &
            (master_df_for_worker['time_step'] >= initial_pos_reference_hour) &
            (master_df_for_worker['time_step'] <= target_step_hour)
        ].sort_values(by=['particle_id', 'time_step'])

        for pid, group in particle_histories_for_map_plot.groupby('particle_id'):
            trajectories_for_map[pid] = {
                'latitudes': group['latitude'].tolist(), 'longitudes': group['longitude'].tolist(),
                'pressures': group['pressure'].tolist(), 'time_steps': group['time_step'].tolist()
                # q/T not strictly needed for this specific map plot, but available in group
            }
        
        # For q/T change plots: initial_pos_reference_hour up to target_step_hour + track_after_target_hours_val
        max_hour_for_change_plot = target_step_hour + track_after_target_hours_val
        
        trajectories_for_change_plots = {}
        particle_histories_for_change_plot = master_df_for_worker[
            (master_df_for_worker['particle_id'].isin(merged_df['particle_id'])) &
            (master_df_for_worker['time_step'] >= initial_pos_reference_hour) &
            (master_df_for_worker['time_step'] <= max_hour_for_change_plot) # Extended window
        ].sort_values(by=['particle_id', 'time_step'])

        for pid, group in particle_histories_for_change_plot.groupby('particle_id'):
             trajectories_for_change_plots[pid] = {
                'time_steps': group['time_step'].tolist(),
                'datetimes': [simulation_start_datetime_obj + pd.Timedelta(hours=ts) for ts in group['time_step'].tolist()], # Actual datetimes
                'specific_humidity': group['specific_humidity'].tolist(),
                'temperature': group['temperature'].tolist()
            }

        # --- Plotting Initial Positions ---
        if not merged_df.empty:
            fig_init = plt.figure(figsize=(12, 8)); ax_init = fig_init.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            initial_datetime_obj = simulation_start_datetime_obj + pd.Timedelta(hours=initial_pos_reference_hour) # Datetime object
            ax_init.set_title(f'Initial Positions ({initial_datetime_obj.strftime(datetime_plot_format_str)}) Reaching Target at {target_datetime_obj.strftime(datetime_plot_format_str)}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
            # ... (rest of initial position plotting - no changes needed here) ...
            lons_init, lats_init = merged_df['longitude_initial'].values, merged_df['latitude_initial'].values
            x0,x1,y0,y1 = _safe_plotting_boundaries_tracks_worker(lons_init, lats_init, fixed_plot_extent_2d_val, default_plot_buffer_deg_val)
            ax_init.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
            _add_map_features_tracks(ax_init)
            _add_target_area_2d_tracks(ax_init, target_lon_center_val, target_lat_center_val, target_box_half_width_deg_val, target_area_plot_color_val)
            sc = ax_init.scatter(lons_init, lats_init, c=merged_df['pressure_initial'], cmap=cmap_tracks_pressure_obj, norm=norm_tracks_pressure_obj,
                                 s=30, transform=ccrs.Geodetic(), edgecolor='black', linewidth=0.5, zorder=3)
            cbar = plt.colorbar(sc, ax=ax_init, pad=0.05, shrink=0.8); cbar.set_label('Initial Pressure (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE); cbar.set_ticks(pressure_bins_categorical_array)
            cbar.ax.tick_params(labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
            plt.savefig(plots_subdir / f'initial_positions_step_{target_step_hour:04d}.png', bbox_inches='tight', dpi=150); plt.close(fig_init)


        # --- Plotting Trajectories (Map) ---
        if trajectories_for_map: # Use trajectories_for_map
            fig_traj = plt.figure(figsize=(12, 8)); ax_traj = fig_traj.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            initial_datetime_obj = simulation_start_datetime_obj + pd.Timedelta(hours=initial_pos_reference_hour) # For title
            ax_traj.set_title(f'Trajectories from {initial_datetime_obj.strftime(datetime_plot_format_str)} to Target ({target_datetime_obj.strftime(datetime_plot_format_str)})', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
            # ... (rest of trajectory map plotting) ...
            all_traj_lons = np.concatenate([t['longitudes'] for t in trajectories_for_map.values() if t['longitudes']])
            all_traj_lats = np.concatenate([t['latitudes'] for t in trajectories_for_map.values() if t['latitudes']])
            if len(all_traj_lons) > 0:
                x0_t,x1_t,y0_t,y1_t = _safe_plotting_boundaries_tracks_worker(all_traj_lons, all_traj_lats, fixed_plot_extent_2d_val, default_plot_buffer_deg_val)
                ax_traj.set_extent([x0_t, x1_t, y0_t, y1_t], crs=ccrs.PlateCarree())
            _add_map_features_tracks(ax_traj)
            _add_target_area_2d_tracks(ax_traj, target_lon_center_val, target_lat_center_val, target_box_half_width_deg_val, target_area_plot_color_val)
            for pid, data in trajectories_for_map.items():
                if len(data['longitudes']) >= 2: ax_traj.plot(data['longitudes'], data['latitudes'], '-', color='gray', alpha=0.6, transform=ccrs.Geodetic(), linewidth=1.0, zorder=3)
                if data['longitudes']:
                    ax_traj.scatter(data['longitudes'][0], data['latitudes'][0], c=[data['pressures'][0]], cmap=cmap_tracks_pressure_obj, norm=norm_tracks_pressure_obj, s=40, edgecolor='black', marker='o', zorder=4, transform=ccrs.Geodetic())
                    ax_traj.scatter(data['longitudes'][-1], data['latitudes'][-1], c=[data['pressures'][-1]], cmap=cmap_tracks_pressure_obj, norm=norm_tracks_pressure_obj, s=40, edgecolor='black', marker='s', zorder=4, transform=ccrs.Geodetic())
            # Corrected Legend (Requirement 5)
            handles = [plt.Line2D([0], [0], marker='o', color='w', mfc='grey', ms=8 * FONT_SCALE, label='Start', ls='None'),
                       plt.Line2D([0], [0], marker='s', color='w', mfc='grey', ms=8 * FONT_SCALE, label='End at Target', ls='None')]
            ax_traj.legend(handles=handles, loc='best', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
            sm = ScalarMappable(cmap=cmap_tracks_pressure_obj, norm=norm_tracks_pressure_obj); sm.set_array([])
            cbar_traj = plt.colorbar(sm, ax=ax_traj, pad=0.05, shrink=0.8); cbar_traj.set_label('Pressure (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE); cbar_traj.set_ticks(pressure_bins_categorical_array)
            cbar_traj.ax.tick_params(labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
            plt.savefig(plots_subdir / f'trajectories_step_{target_step_hour:04d}.png', bbox_inches='tight', dpi=150); plt.close(fig_traj)

               # --- Plotting Change in q and T along Extended Trajectories ---
        for pid, data in trajectories_for_change_plots.items(): # Use extended trajectories
            if not data['time_steps'] or len(data['time_steps']) < 2: continue

            datetimes_for_plot = data['datetimes'] # Already datetime objects
            arrival_datetime_at_target = target_datetime_obj # Datetime of arrival at target

            # Specific Humidity Change Plot
            q_values = np.array(data['specific_humidity']) * 1000 # g/kg
            if len(q_values) == len(datetimes_for_plot):
                fig_q, ax_q = plt.subplots(figsize=(12, 6)) # Wider plot
                ax_q.plot(datetimes_for_plot, q_values, marker='.', linestyle='-', color='dodgerblue')
                ax_q.axvline(arrival_datetime_at_target, color='red', linestyle='--', linewidth=1, label=f'Arrival at Target\n({arrival_datetime_at_target.strftime(datetime_plot_format_str)})')
                
                ax_q.set_xlabel('Date / Time', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                ax_q.set_ylabel('Specific Humidity (g/kg)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                plot_start_dt_str = datetimes_for_plot[0].strftime(datetime_plot_format_str)
                plot_end_dt_str = datetimes_for_plot[-1].strftime(datetime_plot_format_str)
                ax_q.set_title(f'Particle ID {int(pid)}: Specific Humidity Change\n{plot_start_dt_str} to {plot_end_dt_str}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                
                # Format x-axis for datetimes
                #ax_q.xaxis.set_major_formatter(mdates.DateFormatter(datetime_plot_format_str))
                ax_q.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
                ax_q.xaxis.set_major_locator(mdates.HourLocator(interval=24))
                ax_q.tick_params(axis='x', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                ax_q.tick_params(axis='y', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                #plt.xticks(rotation=45, ha='right')
                plt.xticks(rotation=0, ha='center')
                ax_q.legend(fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                ax_q.grid(True, linestyle=':')
                plt.tight_layout()
                plt.savefig(change_plots_subdir / f'particle_{int(pid)}_q_change_targetH{target_step_hour:04d}.png', dpi=150); plt.close(fig_q)

            # Temperature Change Plot
            t_values = np.array(data['temperature']) # Kelvin
            if len(t_values) == len(datetimes_for_plot):
                fig_t, ax_t = plt.subplots(figsize=(12, 6))
                ax_t.plot(datetimes_for_plot, t_values, marker='.', linestyle='-', color='orangered')
                ax_t.axvline(arrival_datetime_at_target, color='red', linestyle='--', linewidth=1, label=f'Arrival at Target\n({arrival_datetime_at_target.strftime(datetime_plot_format_str)})')

                ax_t.set_xlabel('Date / Time', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                ax_t.set_ylabel('Temperature (K)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                plot_start_dt_str = datetimes_for_plot[0].strftime(datetime_plot_format_str)
                plot_end_dt_str = datetimes_for_plot[-1].strftime(datetime_plot_format_str)
                ax_t.set_title(f'Particle ID {int(pid)}: Temperature Change\n{plot_start_dt_str} to {plot_end_dt_str}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)

                #ax_t.xaxis.set_major_formatter(mdates.DateFormatter(datetime_plot_format_str))
                ax_t.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
                ax_t.xaxis.set_major_locator(mdates.HourLocator(interval=24))
                ax_t.tick_params(axis='x', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                ax_t.tick_params(axis='y', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                #plt.xticks(rotation=45, ha='right')
                plt.xticks(rotation=0, ha='center')
                ax_t.legend(fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                ax_t.grid(True, linestyle=':')
                plt.tight_layout()
                plt.savefig(change_plots_subdir / f'particle_{int(pid)}_t_change_targetH{target_step_hour:04d}.png', dpi=150); plt.close(fig_t)
        
        return f"Success_TracksMaster_H{target_step_hour}"
    except Exception as e:
        print(f"Tracks Analysis Error (from master_df) for H{target_step_hour}: {type(e).__name__} - {e}")
        # import traceback # Uncomment for full traceback if needed
        # print(traceback.format_exc())
        return f"Error_TracksMaster_H{target_step_hour}_{e.__class__.__name__}"
    finally:
        gc.collect()
    

def run_tracks_analysis_from_master(master_df_main: pd.DataFrame, config_obj):
    print("\n--- Starting Tracks Analysis (using Master DataFrame) ---")
    
    target_steps_for_tracks = config_obj.TRACKS_TARGET_STEPS 
    if not target_steps_for_tracks:
        print("No target_steps for tracks analysis in cfg.TRACKS_TARGET_STEPS. Skipping.")
        return

    initial_pos_ref_hour_tracks = config_obj.ANALYSIS_START_HOUR
    print(f"Tracks Analysis: Initial positions from H{initial_pos_ref_hour_tracks}. Tracking +{config_obj.TRACK_HISTORY_WINDOW_AFTER_MAX_ARRIVAL_HOURS}hrs post-target for change plots.")

    tasks = []
    for step_hour in target_steps_for_tracks:
        tasks.append((
            step_hour, master_df_main, str(config_obj.TRACKS_OUTPUT_DIR),
            config_obj.SIMULATION_START_DATETIME.isoformat(), config_obj.DATETIME_PLOT_FORMAT,
            config_obj.TARGET_LAT_CENTER, config_obj.TARGET_LON_CENTER, config_obj.TARGET_BOX_HALF_WIDTH_DEG,
            config_obj.TARGET_AREA_PLOT_COLOR, config_obj.TRACKS_PLOT_EXTENT_2D, # Use TRACKS_PLOT_EXTENT_2D
            config_obj.DEFAULT_PLOT_BUFFER_DEG, config_obj.PRESSURE_BINS_CATEGORICAL,
            config_obj.CMAP_PRESSURE, initial_pos_ref_hour_tracks,
            config_obj.TRACK_HISTORY_WINDOW_AFTER_MAX_ARRIVAL_HOURS # Pass new arg
        ))

    if config_obj.NUM_WORKERS > 1 and len(tasks) > 1 and len(master_df_main) < 5000000 :
        print(f"Starting tracks analysis pool with {min(config_obj.NUM_WORKERS, len(tasks))} workers...")
        with Pool(processes=min(config_obj.NUM_WORKERS, len(tasks))) as pool:
            results = list(tqdm(pool.starmap(process_track_step_from_master, tasks), total=len(tasks), desc="Processing track steps (from master)"))
    else:
        # ... (serial execution logic - no changes here) ...
        if config_obj.NUM_WORKERS > 1 and len(master_df_main) >= 500000:
             print("Master_df may be too large for efficient pickling for tracks analysis. Running serially.")
        print("Running tracks analysis (from master_df) in serial mode...")
        results = [process_track_step_from_master(*task_args) for task_args in tqdm(tasks, desc="Processing track steps (from master, serial)")]


    for res in results:
        if "Error" in res: print(f"Worker Error: {res}")
        elif "NoData" in res or "NoParticles" in res: print(f"Worker Info: {res}")
    print("--- Tracks Analysis (from Master DataFrame) Finished ---")