# MoistureTracks.py
"""
Module for analyzing and plotting trajectories of particles that release moisture
over the target area.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable, get_cmap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from tqdm import tqdm
import gc
from multiprocessing import Pool

import config as cfg

def _create_cmap_norm(cmap_name_str, data_values, vmin=None, vmax=None):
    """Creates a colormap and normalization for a given set of data values."""
    cmap = get_cmap(cmap_name_str)
    if vmin is None:
        vmin = np.nanmin(data_values)
    if vmax is None:
        vmax = np.nanmax(data_values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm

def _add_map_features_tracks(ax):
    ax.add_feature(cfeature.LAND.with_scale('110m'), zorder=-1, edgecolor='black', facecolor='#D3D3D3')
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), zorder=1, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale('110m'), linestyle=':', zorder=1, linewidth=0.6)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False, y_inline=False, zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

def _add_target_area_2d_tracks(ax, target_lon, target_lat, target_half_width, target_plot_color):
    target_lons_box = [target_lon - target_half_width, target_lon + target_half_width,
                       target_lon + target_half_width, target_lon - target_half_width,
                       target_lon - target_half_width]
    target_lats_box = [target_lat - target_half_width, target_lat - target_half_width,
                       target_lat + target_half_width, target_lat + target_half_width,
                       target_lat - target_half_width]
    ax.plot(target_lons_box, target_lats_box, color=target_plot_color,
            linewidth=1.5, transform=ccrs.Geodetic(), label='Target Area', zorder=5, alpha=0.8)

def plot_trajectories(df, particle_ids, hour_str, output_dir, 
                      target_lat, target_lon, target_half_width, target_color,
                      color_by, cmap, norm, release_data,
                      sim_start_dt, dt_format, analysis_start_hour, fixed_extent):
    """Plots trajectories for a given set of particles."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if fixed_extent:
        ax.set_extent(fixed_extent, crs=ccrs.PlateCarree())

    # Adjust plot styles for cumulative plots to reduce clutter
    is_cumulative = (hour_str == "cumulative")
    line_width = 0.5 if is_cumulative else 1.0
    marker_size = 10 if is_cumulative else 40
    edge_width = 0.5 if is_cumulative else 1.0

    for pid in particle_ids:
        particle_traj = df[df['particle_id'] == pid].sort_values(by='time_step')
        if particle_traj.empty:
            continue

        ax.plot(particle_traj['longitude'], particle_traj['latitude'], '-', color='gray', alpha=0.6, transform=ccrs.Geodetic(), linewidth=line_width)

        initial_point = particle_traj.iloc[0]
        color_val = None
        if color_by == "pressure":
            color_val = initial_point['pressure']
        elif color_by == "release":
            color_val = release_data.get(pid, 0)

        sc = ax.scatter(initial_point['longitude'], initial_point['latitude'], c=color_val, cmap=cmap, norm=norm, s=marker_size, edgecolor='black', marker='o', transform=ccrs.Geodetic(), zorder=10, linewidths=edge_width)

    _add_map_features_tracks(ax)
    _add_target_area_2d_tracks(ax, target_lon, target_lat, target_half_width, target_color)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8)
    
    title_str = ""
    if hour_str == "cumulative":
        title_str = "Cumulative Trajectories of All Moisture-Releasing Particles"
    else:
        initial_datetime_obj = sim_start_dt + pd.Timedelta(hours=analysis_start_hour)
        target_datetime_obj = sim_start_dt + pd.Timedelta(hours=hour_str)
        title_str = f"Trajectories from {initial_datetime_obj.strftime(dt_format)} to Target ({target_datetime_obj.strftime(dt_format)})"

    if color_by == "pressure":
        ax.set_title(f"{title_str}\nColor-coded by Initial Pressure")
        cbar.set_label("Initial Pressure (hPa)")
    elif color_by == "release":
        ax.set_title(f"{title_str}\nColor-coded by Moisture Release Amount")
        cbar.set_label("Total Moisture Released in Target (g/kg)")

    filename = f"moisture_release_trajectories_hour_{hour_str}_{color_by}.png"
    if hour_str == "cumulative":
        filename = f"cumulative_moisture_release_trajectories_{color_by}.png"

    plt.savefig(output_dir / filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    gc.collect()

def _aggregate_events_on_grid(df_events, value_col, grid_res_deg, lon_extent, lat_extent):
    """
    Aggregates event data onto a regular grid.
    
    Args:
        df_events (pd.DataFrame): DataFrame containing event data with 'longitude', 'latitude', and value_col.
        value_col (str): The name of the column to aggregate (e.g., 'amount').
        grid_res_deg (float): The resolution of the grid in degrees.
        lon_extent (list/tuple): The [lon_min, lon_max] for the grid.
        lat_extent (list/tuple): The [lat_min, lat_max] for the grid.

    Returns:
        tuple: (aggregated_grid, lon_bins, lat_bins)
               - aggregated_grid (np.ndarray): 2D array with summed values.
               - lon_bins (np.ndarray): The longitude bin edges.
               - lat_bins (np.ndarray): The latitude bin edges.
    """
    if df_events.empty:
        return None, None, None

    lon_bins = np.arange(lon_extent[0], lon_extent[1] + grid_res_deg, grid_res_deg)
    lat_bins = np.arange(lat_extent[0], lat_extent[1] + grid_res_deg, grid_res_deg)

    df_events['lon_bin'] = pd.cut(df_events['longitude'], bins=lon_bins, right=False, labels=False)
    df_events['lat_bin'] = pd.cut(df_events['latitude'], bins=lat_bins, right=False, labels=False)

    # Drop events outside the specified grid extent
    df_events.dropna(subset=['lon_bin', 'lat_bin'], inplace=True)
    df_events['lon_bin'] = df_events['lon_bin'].astype(int)
    df_events['lat_bin'] = df_events['lat_bin'].astype(int)

    # Aggregate by summing the value_col in each grid cell
    aggregated = df_events.groupby(['lat_bin', 'lon_bin'])[value_col].sum()
    
    # Create a 2D grid to store the results
    grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
    for (lat_idx, lon_idx), value in aggregated.items():
        grid[lat_idx, lon_idx] = value
        
    return grid, lon_bins, lat_bins

def plot_gridded_aggregate_map(grid, lon_bins, lat_bins, cmap, norm, cbar_label, title, output_path, config_obj, extent=None):
    """
    Generic function to plot a gridded aggregate map.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    elif config_obj.FIXED_PLOT_EXTENT_2D:
        ax.set_extent(config_obj.FIXED_PLOT_EXTENT_2D, crs=ccrs.PlateCarree())

    # Use pcolormesh for gridded data
    # Note: lon_bins and lat_bins are the edges of the grid cells
    mesh = ax.pcolormesh(lon_bins, lat_bins, grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=0)

    _add_map_features_tracks(ax)
    _add_target_area_2d_tracks(ax, config_obj.TARGET_LON_CENTER, config_obj.TARGET_LAT_CENTER, 
                              config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_AREA_PLOT_COLOR)
    
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, extend='both')
    cbar.set_label(cbar_label)
    
    ax.set_title(title)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    gc.collect()

def plot_net_moisture_flux(master_df, particle_ids, output_dir, config_obj):
    """
    Calculates and plots the net moisture flux (Uptake - Release) on a grid.
    """
    print("Generating net moisture flux map...")
    
    event_df = master_df[
        (master_df['particle_id'].isin(particle_ids)) &
        (master_df['moisture_event_type'].isin(['Pickup', 'Release']))
    ].copy()

    if event_df.empty:
        print("No moisture uptake or release events found for the given particles.")
        return

    # IMPORTANT: Use the sign of dq_dt. Uptake is positive, Release is negative.
    # The values are converted to g/kg.
    event_df['amount'] = event_df['dq_dt'] * 1000 

    grid_res = config_obj.AGGREGATE_MAP_GRID_RESOLUTION_DEG
    lon_extent = config_obj.FIXED_PLOT_EXTENT_2D[:2]
    lat_extent = config_obj.FIXED_PLOT_EXTENT_2D[2:]

    net_flux_grid, lon_bins, lat_bins = _aggregate_events_on_grid(event_df, 'amount', grid_res, lon_extent, lat_extent)

    if net_flux_grid is None:
        print("Failed to generate net flux grid.")
        return

    # Create a diverging colormap and normalization centered at 0
    cmap = get_cmap('RdBu_r')
    # Find the maximum absolute value for symmetrical color scaling
    max_abs_val = np.nanmax(np.abs(net_flux_grid))
    if max_abs_val == 0: max_abs_val = 1 # Avoid division by zero
    norm = TwoSlopeNorm(vcenter=0, vmin=-max_abs_val, vmax=max_abs_val)

    plot_gridded_aggregate_map(
        grid=net_flux_grid,
        lon_bins=lon_bins,
        lat_bins=lat_bins,
        cmap=cmap,
        norm=norm,
        cbar_label=f"Net Moisture Exchange (g/kg per {grid_res}°x{grid_res}° cell)",
        title="Net Moisture Exchange (Uptake - Release)\\nFor Particles Releasing Moisture in Target Area",
        output_path=output_dir / "net_moisture_exchange_map.png",
        config_obj=config_obj
    )

def plot_moisture_contribution_grid(master_df, particle_ids, output_dir, config_obj):
    """
    Analyzes and plots the spatial contribution of moisture from different regions.
    This function grids the area, calculates the total moisture released in each grid cell,
    and saves the output as a heatmap plot and a CSV file.
    """
    print("Generating gridded moisture contribution map...")

    # 1. Filter for release events from the specified particles
    release_events_df = master_df[
        (master_df['particle_id'].isin(particle_ids)) &
        (master_df['moisture_event_type'] == 'Release')
    ].copy()

    if release_events_df.empty:
        print("No moisture release events found for the given particles.")
        return

    # 2. Use the absolute value of dq_dt, converted to g/kg, as the contribution amount
    release_events_df['amount'] = release_events_df['dq_dt'].abs() * 1000

    # 3. Define grid parameters
    grid_res = 0.2  # Grid size set to 0.2 as per user request.
    # TODO: The longitude range is hardcoded as per user request for clustering plot.
    # This should be moved to the config file in the future.
    lon_extent = (50, 110)
    lat_extent = config_obj.FIXED_PLOT_EXTENT_2D[2:]
    plot_extent = (lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1])

    # 4. Aggregate the moisture contribution onto the grid
    contribution_grid, lon_bins, lat_bins = _aggregate_events_on_grid(
        release_events_df, 'amount', grid_res, lon_extent, lat_extent
    )

    if contribution_grid is None:
        print("Failed to generate the moisture contribution grid.")
        return

    # 5. Save the aggregated data to a CSV file
    # To do this, we need to convert the grid back to a DataFrame format.
    lon_bin_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_bin_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    
    csv_data = []
    for lat_idx, lat in enumerate(lat_bin_centers):
        for lon_idx, lon in enumerate(lon_bin_centers):
            value = contribution_grid[lat_idx, lon_idx]
            if value > 0:
                csv_data.append({
                    'latitude_center': lat,
                    'longitude_center': lon,
                    'total_moisture_contribution_g_kg': value
                })
    
    if csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv_output_path = output_dir / "gridded_moisture_contribution.csv"
        csv_df.to_csv(csv_output_path, index=False, float_format='%.5f')
        print(f"Saved gridded moisture contribution data to {csv_output_path}")

    # 6. Create colormap and normalization
    cmap = get_cmap(config_obj.CMAP_RELEASE_AGG)
    # We can set a robust max for the color scale to avoid outliers dominating
    vmax = np.percentile(contribution_grid[contribution_grid > 0], 95) if np.any(contribution_grid > 0) else 1
    norm = Normalize(vmin=0, vmax=vmax)

    # 7. Plot the gridded map
    plot_gridded_aggregate_map(
        grid=contribution_grid,
        lon_bins=lon_bins,
        lat_bins=lat_bins,
        cmap=cmap,
        norm=norm,
        cbar_label=f"Total Moisture Contribution (g/kg per {grid_res}°x{grid_res}° cell)",
        title="Spatial Clustering of Moisture Contribution to Target Area",
        output_path=output_dir / "gridded_moisture_contribution_map.png",
        config_obj=config_obj,
        extent=plot_extent
    )
    print("Finished generating gridded moisture contribution map.")

# def plot_moisture_pickup_aggregate(master_df, particle_ids, output_dir, config_obj):
#     """
#     Plots aggregate map of moisture pickup points for moisture-releasing particles.
#     """
#     print("Generating aggregate moisture pickup points map...")
    
#     # Filter for pickup events by moisture-releasing particles
#     pickup_df = master_df[
#         (master_df['particle_id'].isin(particle_ids)) &
#         (master_df['moisture_event_type'] == 'Pickup')
#     ].copy()
    
#     if pickup_df.empty:
#         print("No pickup events found for moisture-releasing particles.")
#         return
    
#     # Calculate pickup amounts (absolute values)
#     pickup_df['pickup_amount'] = pickup_df['dq_dt'].abs() * 1000  # Convert to g/kg
    
#     # Create figure
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
#     if config_obj.FIXED_PLOT_EXTENT_2D:
#         ax.set_extent(config_obj.FIXED_PLOT_EXTENT_2D, crs=ccrs.PlateCarree())
    
#     # Create colormap and normalization for pickup amounts
#     cmap_pickup, norm_pickup = _create_cmap_norm('Blues', pickup_df['pickup_amount'].values)
    
#     # Plot pickup points
#     sc = ax.scatter(pickup_df['longitude'], pickup_df['latitude'], 
#                    c=pickup_df['pickup_amount'], cmap=cmap_pickup, norm=norm_pickup,
#                    s=20, alpha=0.7, edgecolor='none', transform=ccrs.Geodetic(), zorder=10)
    
#     _add_map_features_tracks(ax)
#     _add_target_area_2d_tracks(ax, config_obj.TARGET_LON_CENTER, config_obj.TARGET_LAT_CENTER, 
#                               config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_AREA_PLOT_COLOR)
    
#     # Add colorbar
#     cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
#     cbar.set_label("Moisture Pickup Amount (g/kg)")
    
#     ax.set_title("Aggregate Moisture Pickup Points\n(Particles that Release Moisture in Target Area)")
    
#     plt.savefig(output_dir / "aggregate_moisture_pickup_points.png", bbox_inches='tight', dpi=150)
#     plt.close(fig)
#     gc.collect()

# def plot_moisture_release_aggregate(master_df, particle_ids, output_dir, config_obj):
#     """
#     Plots aggregate map of moisture release points for moisture-releasing particles.
#     """
#     print("Generating aggregate moisture release points map...")
    
#     # Filter for release events by moisture-releasing particles
#     release_df = master_df[
#         (master_df['particle_id'].isin(particle_ids)) &
#         (master_df['moisture_event_type'] == 'Release')
#     ].copy()
    
#     if release_df.empty:
#         print("No release events found for moisture-releasing particles.")
#         return
    
#     # Calculate release amounts (absolute values)
#     release_df['release_amount'] = release_df['dq_dt'].abs() * 1000  # Convert to g/kg
    
#     # Create figure
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
#     if config_obj.FIXED_PLOT_EXTENT_2D:
#         ax.set_extent(config_obj.FIXED_PLOT_EXTENT_2D, crs=ccrs.PlateCarree())
    
#     # Create colormap and normalization for release amounts
#     cmap_release, norm_release = _create_cmap_norm('Reds', release_df['release_amount'].values)
    
#     # Plot release points
#     sc = ax.scatter(release_df['longitude'], release_df['latitude'], 
#                    c=release_df['release_amount'], cmap=cmap_release, norm=norm_release,
#                    s=20, alpha=0.7, edgecolor='none', transform=ccrs.Geodetic(), zorder=10)
    
#     _add_map_features_tracks(ax)
#     _add_target_area_2d_tracks(ax, config_obj.TARGET_LON_CENTER, config_obj.TARGET_LAT_CENTER, 
#                               config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_AREA_PLOT_COLOR)
    
#     # Add colorbar
#     cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
#     cbar.set_label("Moisture Release Amount (g/kg)")
    
#     ax.set_title("Aggregate Moisture Release Points\n(Particles that Release Moisture in Target Area)")
    
#     plt.savefig(output_dir / "aggregate_moisture_release_points.png", bbox_inches='tight', dpi=150)
#     plt.close(fig)
#     gc.collect()

# def plot_moisture_combined_aggregate(master_df, particle_ids, output_dir, config_obj):
#     """
#     Plots combined aggregate map showing both moisture pickup and release points.
#     """
#     print("Generating combined aggregate moisture map...")
    
#     # Filter for pickup and release events by moisture-releasing particles
#     pickup_df = master_df[
#         (master_df['particle_id'].isin(particle_ids)) &
#         (master_df['moisture_event_type'] == 'Pickup')
#     ].copy()
    
#     release_df = master_df[
#         (master_df['particle_id'].isin(particle_ids)) &
#         (master_df['moisture_event_type'] == 'Release')
#     ].copy()
    
#     if pickup_df.empty and release_df.empty:
#         print("No pickup or release events found for moisture-releasing particles.")
#         return
    
#     # Calculate amounts (absolute values)
#     if not pickup_df.empty:
#         pickup_df['pickup_amount'] = pickup_df['dq_dt'].abs() * 1000  # Convert to g/kg
#     if not release_df.empty:
#         release_df['release_amount'] = release_df['dq_dt'].abs() * 1000  # Convert to g/kg
    
#     # Create figure
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
#     if config_obj.FIXED_PLOT_EXTENT_2D:
#         ax.set_extent(config_obj.FIXED_PLOT_EXTENT_2D, crs=ccrs.PlateCarree())
    
#     # Plot pickup points in blue
#     if not pickup_df.empty:
#         cmap_pickup, norm_pickup = _create_cmap_norm('Blues', pickup_df['pickup_amount'].values)
#         sc1 = ax.scatter(pickup_df['longitude'], pickup_df['latitude'], 
#                         c=pickup_df['pickup_amount'], cmap=cmap_pickup, norm=norm_pickup,
#                         s=15, alpha=0.6, edgecolor='none', transform=ccrs.Geodetic(), 
#                         zorder=10, label='Moisture Pickup')
    
#     # Plot release points in red
#     if not release_df.empty:
#         cmap_release, norm_release = _create_cmap_norm('Reds', release_df['release_amount'].values)
#         sc2 = ax.scatter(release_df['longitude'], release_df['latitude'], 
#                         c=release_df['release_amount'], cmap=cmap_release, norm=norm_release,
#                         s=15, alpha=0.6, edgecolor='none', transform=ccrs.Geodetic(), 
#                         zorder=11, label='Moisture Release')
    
#     _add_map_features_tracks(ax)
#     _add_target_area_2d_tracks(ax, config_obj.TARGET_LON_CENTER, config_obj.TARGET_LAT_CENTER, 
#                               config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_AREA_PLOT_COLOR)
    
#     # Add colorbars
#     if not pickup_df.empty and not release_df.empty:
#         # Create two colorbars
#         cbar1 = fig.colorbar(sc1, ax=ax, shrink=0.4, pad=0.01, aspect=30)
#         cbar1.set_label("Pickup (g/kg)", fontsize=10)
        
#         cbar2 = fig.colorbar(sc2, ax=ax, shrink=0.4, pad=0.12, aspect=30)
#         cbar2.set_label("Release (g/kg)", fontsize=10)
#     elif not pickup_df.empty:
#         cbar = fig.colorbar(sc1, ax=ax, shrink=0.8)
#         cbar.set_label("Moisture Pickup Amount (g/kg)")
#     elif not release_df.empty:
#         cbar = fig.colorbar(sc2, ax=ax, shrink=0.8)
#         cbar.set_label("Moisture Release Amount (g/kg)")
    
#     ax.set_title("Combined Aggregate Moisture Map\n(Pickup in Blue, Release in Red - Particles Releasing in Target)")
    
#     # Add legend
#     if not pickup_df.empty and not release_df.empty:
#         ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    
#     plt.savefig(output_dir / "aggregate_moisture_combined_map.png", bbox_inches='tight', dpi=150)
#     plt.close(fig)
#     gc.collect()

def _plot_hourly_trajectories_worker(args):
    (target_step_hour, master_df, moisture_releasing_particle_ids, output_dir, 
    target_lat, target_lon, target_half_width, target_color, 
    cmap_pressure, norm_pressure, cmap_release, norm_release, total_release, 
    tracks_data_dir, sim_start_dt, dt_format, analysis_start_hour, fixed_extent) = args


    # Read the list of particles arriving at this hour from tracks analysis output
    arriving_particles_file = tracks_data_dir / f'target_particles_step_{target_step_hour:04d}_analyzed.parquet'
    if not arriving_particles_file.exists():
        return

    arriving_df = pd.read_parquet(arriving_particles_file)
    arriving_particle_ids = arriving_df['particle_id'].unique()

    # Find the intersection of arriving particles and moisture-releasing particles
    particle_subset_ids = np.intersect1d(arriving_particle_ids, moisture_releasing_particle_ids)

    if len(particle_subset_ids) == 0:
        return

    trajectories = master_df[
        (master_df['particle_id'].isin(particle_subset_ids)) &
        (master_df['time_step'] <= target_step_hour)
    ]

    # Plot 1: Color by pressure
    plot_trajectories(trajectories, particle_subset_ids, target_step_hour, output_dir, 
                      target_lat, target_lon, target_half_width, target_color,
                      "pressure", cmap_pressure, norm_pressure, total_release,
                      sim_start_dt, dt_format, analysis_start_hour, fixed_extent)
    # Plot 2: Color by moisture release
    plot_trajectories(trajectories, particle_subset_ids, target_step_hour, output_dir, 
                      target_lat, target_lon, target_half_width, target_color,
                      "release", cmap_release, norm_release, total_release,
                      sim_start_dt, dt_format, analysis_start_hour, fixed_extent)

def plot_moisture_release_time_evolution(master_df, particle_ids, output_dir, config_obj):
    """
    Plots the time evolution of specific humidity and its change for a given set of particles.
    """
    print("Plotting time evolution for moisture-releasing particles...")

    df = master_df[master_df['particle_id'].isin(particle_ids)].copy()
    
    # Convert time_step to datetime for plotting
    df['datetime'] = config_obj.SIMULATION_START_DATETIME + pd.to_timedelta(df['time_step'], unit='h')

    # --- Plot 1: Specific Humidity ---
    fig, ax1 = plt.subplots(figsize=(12, 6)) 
    
    # Calculate mean and std dev of specific humidity at each time step
    q_mean = df.groupby('datetime')['specific_humidity'].mean() * 1000 # to g/kg
    q_std = df.groupby('datetime')['specific_humidity'].std() * 1000

    ax1.plot(q_mean.index, q_mean, label='Mean Specific Humidity', color='blue')
    ax1.fill_between(q_mean.index, q_mean - q_std, q_mean + q_std, color='blue', alpha=0.2, label='Â±1 Std Dev')
    
    ax1.set_xlabel("Date / Time")
    ax1.set_ylabel("Specific Humidity (g/kg)")
    ax1.set_title("Time Evolution of Specific Humidity for Moisture-Releasing Particles")
    ax1.legend()
    ax1.grid(True)
    
    plt.savefig(output_dir / "time_evolution_specific_humidity.png", bbox_inches='tight', dpi=150)
    plt.close(fig)

    # --- Plot 2: Change in Specific Humidity ---
    fig, ax2 = plt.subplots(figsize=(12, 6))

    # Calculate mean of dq_dt at each time step
    dq_dt_mean = df.groupby('datetime')['dq_dt'].mean() * 1000 # to g/kg per X hrs

    ax2.plot(dq_dt_mean.index, dq_dt_mean, label='Mean Change in Specific Humidity (dq/dt)', color='red')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)

    ax2.set_xlabel("Date / Time")
    ax2.set_ylabel(f"Change in Specific Humidity (g/kg per {config_obj.CHANGE_WINDOW_HOURS}hrs)")
    ax2.set_title("Time Evolution of Moisture Change for Moisture-Releasing Particles")
    ax2.legend()
    ax2.grid(True)

    plt.savefig(output_dir / "time_evolution_dq_dt.png", bbox_inches='tight', dpi=150)
    plt.close(fig)

def plot_total_q_in_target_time_evolution(master_df, particle_ids, output_dir, config_obj):
    """
    Plots the time evolution of total specific humidity for particles within the target region.
    """
    print("Plotting time evolution of total specific humidity in target...")

    lat_min = config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lat_max = config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_min = config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_max = config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG

    in_target_df = master_df[
        (master_df['particle_id'].isin(particle_ids)) &
        (master_df['latitude'].between(lat_min, lat_max)) &
        (master_df['longitude'].between(lon_min, lon_max))
    ].copy()

    in_target_df['datetime'] = config_obj.SIMULATION_START_DATETIME + pd.to_timedelta(in_target_df['time_step'], unit='h')

    total_q = in_target_df.groupby('datetime')['specific_humidity'].sum() * 1000 # to g/kg

    total_q.to_excel(output_dir / "total_specific_humidity_in_target_over_time.xlsx")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(total_q.index, total_q, label='Total Specific Humidity in Target', color='green')
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Total Specific Humidity (g/kg)")
    ax.set_title("Time Evolution of Total Specific Humidity of Moisture-Releasing Particles in Target Area")
    ax.legend()
    ax.grid(True)
    plt.savefig(output_dir / "time_evolution_total_specific_humidity_in_target.png", bbox_inches='tight', dpi=150)
    plt.close(fig)

def generate_hourly_q_details_excel(master_df, particle_ids, output_dir, config_obj):
    """
    Generates hourly Excel files with specific humidity details for particles in the target region.
    """
    print("Generating hourly specific humidity detail Excel files...")
    lat_min = config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lat_max = config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_min = config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_max = config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG

    for hour in tqdm(config_obj.EVENT_CORE_ANALYSIS_STEPS, desc="Generating hourly q details"):
        in_target_at_hour = master_df[
            (master_df['particle_id'].isin(particle_ids)) &
            (master_df['time_step'] == hour) &
            (master_df['latitude'].between(lat_min, lat_max)) &
            (master_df['longitude'].between(lon_min, lon_max))
        ]
        
        if in_target_at_hour.empty:
            continue

        pids_at_hour = in_target_at_hour['particle_id'].unique()
        
        q_details = []
        for pid in pids_at_hour:
            q_h = in_target_at_hour[in_target_at_hour['particle_id'] == pid]['specific_humidity'].iloc[0]
            
            q_h_minus_1_series = master_df[(master_df['particle_id'] == pid) & (master_df['time_step'] == hour - 1)]['specific_humidity']
            q_h_minus_1 = q_h_minus_1_series.iloc[0] if not q_h_minus_1_series.empty else np.nan
            
            q_h_plus_1_series = master_df[(master_df['particle_id'] == pid) & (master_df['time_step'] == hour + 1)]['specific_humidity']
            q_h_plus_1 = q_h_plus_1_series.iloc[0] if not q_h_plus_1_series.empty else np.nan
            
            q_details.append({
                'particle_id': pid,
                f'q_{hour-1}': q_h_minus_1 * 1000,
                f'q_{hour}': q_h * 1000,
                f'q_{hour+1}': q_h_plus_1 * 1000
            })
        
        if q_details:
            df_hour = pd.DataFrame(q_details)
            df_hour.to_excel(output_dir / f"hourly_q_details_hour_{hour}.xlsx", index=False)

def run_moisture_tracks_analysis(master_df: pd.DataFrame, config_obj):
    """
    Main function to identify moisture-releasing particles and generate plots.
    """
    print("--- Starting Moisture Tracks Analysis ---")
    output_dir = config_obj.MOISTURE_TRACKS_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Identify particles releasing moisture in the target area
    lat_min = config_obj.TARGET_LAT_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lat_max = config_obj.TARGET_LAT_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_min = config_obj.TARGET_LON_CENTER - config_obj.TARGET_BOX_HALF_WIDTH_DEG
    lon_max = config_obj.TARGET_LON_CENTER + config_obj.TARGET_BOX_HALF_WIDTH_DEG

    in_target_df = master_df[
        (master_df['latitude'].between(lat_min, lat_max)) &
        (master_df['longitude'].between(lon_min, lon_max))
    ].copy()

    releasing_in_target_df = in_target_df[in_target_df['moisture_event_type'] == 'Release']
    if releasing_in_target_df.empty:
        print("No moisture-releasing particles found in the target area.")
        return

    moisture_releasing_particle_ids = releasing_in_target_df['particle_id'].unique()
    print(f"Found {len(moisture_releasing_particle_ids)} particles releasing moisture in the target area.")

    # Calculate total moisture release for each particle and save to CSV
    total_release = releasing_in_target_df.groupby('particle_id')['dq_dt'].sum().abs() * 1000 # as g/kg
    total_release_df = total_release.reset_index()
    total_release_df.columns = ['particle_id', 'total_moisture_release_g_kg']
    output_csv_path = output_dir / "moisture_releasing_particles_summary.csv"
    total_release_df.to_csv(output_csv_path, index=False, float_format='%.5f')
    print(f"Saved moisture release summary to {output_csv_path}")

    # 2. Generate plots
    # Create colormaps
    cmap_pressure, norm_pressure = _create_cmap_norm(cfg.CMAP_PRESSURE, cfg.PRESSURE_BINS_CATEGORICAL)
    cmap_release, norm_release = _create_cmap_norm(cfg.CMAP_RELEASE_AGG, total_release.values)

    # Hourly plots (parallelized)
    tracks_data_dir = config_obj.TRACKS_OUTPUT_DIR / "data_from_master"
    tasks = []
    for target_step_hour in config_obj.TRACKS_TARGET_STEPS:
        tasks.append((target_step_hour, master_df, moisture_releasing_particle_ids, output_dir, 
                      config_obj.TARGET_LAT_CENTER, config_obj.TARGET_LON_CENTER, 
                      config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_AREA_PLOT_COLOR,
                      cmap_pressure, norm_pressure, cmap_release, norm_release, total_release,
                      tracks_data_dir, config_obj.SIMULATION_START_DATETIME, 
                      config_obj.DATETIME_PLOT_FORMAT, config_obj.ANALYSIS_START_HOUR, config_obj.FIXED_PLOT_EXTENT_2D))

    with Pool(processes=config_obj.NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(_plot_hourly_trajectories_worker, tasks), total=len(tasks), desc="Generating hourly moisture release trajectory plots"))

    # Cumulative plots
    print("Generating cumulative trajectory plots for all moisture-releasing particles...")
    
    last_arrival_times = in_target_df[
        in_target_df['particle_id'].isin(moisture_releasing_particle_ids)
    ].groupby('particle_id')['time_step'].max()

    all_releaser_trajectories_list = []
    for pid, last_arrival_time in last_arrival_times.items():
        particle_traj = master_df[
            (master_df['particle_id'] == pid) &
            (master_df['time_step'] <= last_arrival_time)
        ]
        all_releaser_trajectories_list.append(particle_traj)
    
    all_releaser_trajectories = pd.concat(all_releaser_trajectories_list)

    # Cumulative plot colored by pressure
    plot_trajectories(all_releaser_trajectories, moisture_releasing_particle_ids, "cumulative", output_dir, 
                      config_obj.TARGET_LAT_CENTER, config_obj.TARGET_LON_CENTER, 
                      config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_AREA_PLOT_COLOR,
                      "pressure", cmap_pressure, norm_pressure, total_release,
                      config_obj.SIMULATION_START_DATETIME, config_obj.DATETIME_PLOT_FORMAT, config_obj.ANALYSIS_START_HOUR, config_obj.FIXED_PLOT_EXTENT_2D)

    # Cumulative plot colored by moisture release amount
    plot_trajectories(all_releaser_trajectories, moisture_releasing_particle_ids, "cumulative", output_dir, 
                      config_obj.TARGET_LAT_CENTER, config_obj.TARGET_LON_CENTER, 
                      config_obj.TARGET_BOX_HALF_WIDTH_DEG, config_obj.TARGET_AREA_PLOT_COLOR,
                      "release", cmap_release, norm_release, total_release,
                      config_obj.SIMULATION_START_DATETIME, config_obj.DATETIME_PLOT_FORMAT, config_obj.ANALYSIS_START_HOUR, config_obj.FIXED_PLOT_EXTENT_2D)

    # Generate aggregate moisture maps
    plot_net_moisture_flux(master_df, moisture_releasing_particle_ids, output_dir, config_obj)
    plot_moisture_contribution_grid(master_df, moisture_releasing_particle_ids, output_dir, config_obj)

    # Time evolution plots
    plot_moisture_release_time_evolution(master_df, moisture_releasing_particle_ids, output_dir, config_obj)
    plot_total_q_in_target_time_evolution(master_df, moisture_releasing_particle_ids, output_dir, config_obj)
    generate_hourly_q_details_excel(master_df, moisture_releasing_particle_ids, output_dir, config_obj)

    print("--- Moisture Tracks Analysis Finished ---")
