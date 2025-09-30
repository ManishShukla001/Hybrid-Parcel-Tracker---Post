# plotting_3d.py
"""
Module for generating 3D static plots and 3D animation frames.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection # For 3D target volume & trajectories
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from tqdm import tqdm
import gc
from matplotlib.colors import Normalize, BoundaryNorm # BoundaryNorm for pressure-colored trajectories
from matplotlib.cm import ScalarMappable, get_cmap

# Import global configuration
import config as cfg

# Define a font scaling factor
FONT_SCALE = 2
DEFAULT_FONT_SIZE = 10 # A reasonable base font size

# --- Helper Functions (Mostly from your provided version) ---

def _get_3d_plot_lims(df_subset_lon, df_subset_lat, df_subset_pres, 
                       fixed_lonlat_extent_cfg, fixed_pressure_extent_cfg, 
                       default_buffer_cfg):
    """
    Helper to determine 3D plot limits (dynamic or fixed).
    Takes specific config values as arguments.
    """
    if fixed_lonlat_extent_cfg:
        lon_lim = (fixed_lonlat_extent_cfg[0], fixed_lonlat_extent_cfg[1])
        lat_lim = (fixed_lonlat_extent_cfg[2], fixed_lonlat_extent_cfg[3])
    elif not df_subset_lon.empty and not df_subset_lat.empty:
        lon_min_data, lon_max_data = np.nanmin(df_subset_lon), np.nanmax(df_subset_lon)
        lat_min_data, lat_max_data = np.nanmin(df_subset_lat), np.nanmax(df_subset_lat)
        min_span = 1.0
        if lon_max_data - lon_min_data < min_span:
            mid = (lon_max_data + lon_min_data) / 2; lon_min_data, lon_max_data = mid - min_span / 2, mid + min_span / 2
        if lat_max_data - lat_min_data < min_span:
            mid = (lat_max_data + lat_min_data) / 2; lat_min_data, lat_max_data = mid - min_span / 2, mid + min_span / 2
        
        lon_lim = (max(-180, lon_min_data - default_buffer_cfg), min(180, lon_max_data + default_buffer_cfg))
        lat_lim = (max(-90, lat_min_data - default_buffer_cfg), min(90, lat_max_data + default_buffer_cfg))
        if lon_lim[1] <= lon_lim[0]: lon_lim = (lon_lim[0], lon_lim[0] + default_buffer_cfg)
        if lat_lim[1] <= lat_lim[0]: lat_lim = (lat_lim[0], lat_lim[0] + default_buffer_cfg)
    else:
        lon_lim = (60, 120); lat_lim = (-10, 40)

    if fixed_pressure_extent_cfg:
        pres_lim = (min(fixed_pressure_extent_cfg), max(fixed_pressure_extent_cfg))
    elif not df_subset_pres.empty and not df_subset_pres.isnull().all():
        p_min_data, p_max_data = np.nanmin(df_subset_pres), np.nanmax(df_subset_pres)
        pres_lim = (max(0, p_min_data - 50), p_max_data + 50)
        if pres_lim[0] >= pres_lim[1]: pres_lim = (pres_lim[0], pres_lim[0] + 100)
    else:
        pres_lim = (100, 1000)
        
    return {'lon': lon_lim, 'lat': lat_lim, 'pres': pres_lim}

def _add_target_volume_3d_worker(ax, current_3d_plot_limits_pres, # Pass only pressure limits
                                 target_lon_center, target_lat_center, target_box_half_width,
                                 target_area_plot_color_val):
    lon_min = target_lon_center - target_box_half_width
    lon_max = target_lon_center + target_box_half_width
    lat_min = target_lat_center - target_box_half_width
    lat_max = target_lat_center + target_box_half_width
    
    p_bottom = current_3d_plot_limits_pres[1] # Max pressure (visual bottom of current plot)
    p_top = current_3d_plot_limits_pres[0]    # Min pressure (visual top of current plot)

    vertices = [
        (lon_min, lat_min, p_bottom), (lon_max, lat_min, p_bottom), (lon_max, lat_max, p_bottom), (lon_min, lat_max, p_bottom),
        (lon_min, lat_min, p_top),    (lon_max, lat_min, p_top),    (lon_max, lat_max, p_top),    (lon_min, lat_max, p_top)
    ]
    faces = [ # Define faces for a hollow box (sides, top, bottom)
        [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[2], vertices[3], vertices[7], vertices[6]], [vertices[3], vertices[0], vertices[4], vertices[7]],
        [vertices[4], vertices[5], vertices[6], vertices[7]], [vertices[0], vertices[1], vertices[2], vertices[3]],
    ]
    target_box = Poly3DCollection(faces, facecolors=target_area_plot_color_val,
                                  linewidths=0.7, edgecolors=(0.1,0.1,0.1,0.4), alpha=0.10, zorder=1)
    ax.add_collection3d(target_box)

# --- Frame Generation for Animation (Worker Function) ---
def generate_3d_snapshot_frame(
    hour: int,
    df_at_hour: pd.DataFrame,
    output_dir_for_frames: Path,
    fixed_lims_3d_for_anim, # This is the pre-calculated dict {'lon':(...), 'lat':(...), 'pres':(...)}
    q_norm_for_anim,
    t_norm_for_anim,
    view_angle_for_anim,
    plot_variable,
    simulation_start_dt_iso_str, # ISO string
    datetime_plot_format,
    target_area_color_val,
    target_lon_val,
    target_lat_val,
    target_half_width_val,
    default_plot_buffer_val, # Not needed if fixed_lims_3d_for_anim is always used
    animation_dpi_val,
    cmap_q_name, # from cfg.CMAP_SPECIFIC_HUMIDITY
    cmap_t_name  # from cfg.CMAP_TEMPERATURE
):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    current_simulation_start_dt = pd.Timestamp(simulation_start_dt_iso_str) # Parse ISO string
    current_datetime = current_simulation_start_dt + pd.Timedelta(hours=hour)
    variable_label = "Specific Humidity" if plot_variable == "specific_humidity" else "Temperature"
    #ax.set_title(f'{variable_label} at {current_datetime.strftime(datetime_plot_format)} (Hour {hour:03d})', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.set_title(f'{variable_label} at {current_datetime.strftime(datetime_plot_format)}', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)


    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

    norm_obj_to_use = None
    cmap_obj_to_use = None
    cbar_label_str_val = ""

    if plot_variable == "specific_humidity":
        values_for_color = df_at_hour['specific_humidity'].dropna() * 1000
        cmap_obj_to_use = get_cmap(cmap_q_name)
        norm_obj_to_use = q_norm_for_anim
        cbar_label_str_val = 'Specific Humidity (g/kg)'
    elif plot_variable == "temperature":
        values_for_color = df_at_hour['temperature'].dropna()
        cmap_obj_to_use = get_cmap(cmap_t_name)
        norm_obj_to_use = t_norm_for_anim
        cbar_label_str_val = 'Temperature (K)'
    else:
        plt.close(fig); gc.collect()
        raise ValueError(f"Unknown plot_variable for 3D snapshot: {plot_variable}")

    if values_for_color.empty:
        plt.close(fig); gc.collect(); return None

    valid_indices = values_for_color.index
    lons_scatter = df_at_hour.loc[valid_indices, 'longitude']
    lats_scatter = df_at_hour.loc[valid_indices, 'latitude']
    pressures_scatter = df_at_hour.loc[valid_indices, 'pressure']

    # Use the globally calculated fixed_lims_3d_for_anim for consistent animation frames
    ax.set_xlim(fixed_lims_3d_for_anim['lon'])
    ax.set_ylim(fixed_lims_3d_for_anim['lat'])
    ax.set_zlim(fixed_lims_3d_for_anim['pres'][1], fixed_lims_3d_for_anim['pres'][0]) # (max_p, min_p)
    
    # Set custom ticks
    ax.set_xticks([60, 70, 80, 90, 100])
    ax.set_yticks([-5, 5, 15, 25])
    ax.set_zticks([200, 400, 600, 800, 1000])

    #floor_pressure = fixed_lims_3d_for_anim['pres'][1] # Max pressure value is the floor
    floor_pressure = 1000  # Always use 1000 hPa as floor
    ceiling_pressure = 200  # Always use 200 hPa as ceiling
    
    # --- Floor Rectangle for the plot domain ---
    x_rect = [fixed_lims_3d_for_anim['lon'][0], fixed_lims_3d_for_anim['lon'][1], 
             fixed_lims_3d_for_anim['lon'][1], fixed_lims_3d_for_anim['lon'][0]]
    y_rect = [fixed_lims_3d_for_anim['lat'][0], fixed_lims_3d_for_anim['lat'][0],
             fixed_lims_3d_for_anim['lat'][1], fixed_lims_3d_for_anim['lat'][1]]
    verts_rect = [list(zip(x_rect, y_rect, [floor_pressure]*4))]
    floor_poly = Poly3DCollection(verts_rect, facecolors=(0.92,0.92,0.92,0.6), 
                                 edgecolors='none', zorder=-100)
    ax.add_collection3d(floor_poly)
    
    # Add domain boundary lines to clearly mark the box edges
    # Bottom corners of the box (at floor pressure)
    corners_x = [fixed_lims_3d_for_anim['lon'][0], fixed_lims_3d_for_anim['lon'][1], 
                fixed_lims_3d_for_anim['lon'][1], fixed_lims_3d_for_anim['lon'][0], 
                fixed_lims_3d_for_anim['lon'][0]]
    corners_y = [fixed_lims_3d_for_anim['lat'][0], fixed_lims_3d_for_anim['lat'][0], 
                fixed_lims_3d_for_anim['lat'][1], fixed_lims_3d_for_anim['lat'][1], 
                fixed_lims_3d_for_anim['lat'][0]]
    
    # Draw box edges along the floor
    ax.plot(corners_x, corners_y, floor_pressure, color='black', linestyle='-', 
           linewidth=1.0, alpha=0.7, zorder=-90)

    # Plotting coastlines with proper filtering
    coast = cfeature.NaturalEarthFeature('physical', 'coastline', '110m', 
                                         edgecolor='black', facecolor='none')
    
    for geom in coast.geometries():
        geom_b = geom.bounds
        # Quick check if geometry might intersect our domain (optimization)
        if not (geom_b[2] < fixed_lims_3d_for_anim['lon'][0] or 
                geom_b[0] > fixed_lims_3d_for_anim['lon'][1] or 
                geom_b[3] < fixed_lims_3d_for_anim['lat'][0] or 
                geom_b[1] > fixed_lims_3d_for_anim['lat'][1]):
            
            xs, ys = _filter_coastline_coordinates(geom, 
                                                  fixed_lims_3d_for_anim['lon'], 
                                                  fixed_lims_3d_for_anim['lat'])
            if len(xs) > 0:
                ax.plot(xs, ys, zs=floor_pressure, zdir='z', 
                       color='black', linewidth=0.8, zorder=-50, alpha=0.8)
    
    
    #_add_target_volume_3d_worker(ax, fixed_lims_3d_for_anim['pres'], 
    #                             target_lon_val, target_lat_val, target_half_width_val,
    #                             target_area_color_val)

    target_pressure_limits = (ceiling_pressure, floor_pressure)  # (200, 1000)
    _add_target_volume_3d_worker(ax, target_pressure_limits,
                             target_lon_val, target_lat_val, target_half_width_val,
                             target_area_color_val)
    
    # Particles
    sc = ax.scatter(lons_scatter, lats_scatter, pressures_scatter,
                    c=values_for_color, cmap=cmap_obj_to_use, norm=norm_obj_to_use, s=10, alpha=0.6, 
                    edgecolor='black', linewidth=0.1, depthshade=True, zorder=20)

    #ax.set_xlabel('Longitude (°E)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE); ax.tick_params(axis='x', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE, pad=-25)
    #ax.set_ylabel('Latitude (°N)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE); ax.tick_params(axis='y', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE, pad=-25)
    #ax.set_zlabel('Pressure (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE); ax.tick_params(axis='z', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE, pad=25)
    ax.set_xlabel('Longitude (°E)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE, labelpad=15)
    ax.set_ylabel('Latitude (°N)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE, labelpad=15) 
    ax.set_zlabel('Pressure (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE, labelpad=15)
    ax.tick_params(axis='x', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE, pad=5)
    # ax.tick_params(axis='y', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE, pad=5)
    ax.tick_params(axis='y', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE, pad=-2) # Move labels closer to the axis
    ax.tick_params(axis='z', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE, pad=8)    

    #ax.invert_zaxis()
    ax.view_init(elev=view_angle_for_anim[0], azim=view_angle_for_anim[1])

    # Colorbar - added by animations.py as a static image, or to first frame if needed for 2D
    # If individual static 3D frames need a colorbar:
    if norm_obj_to_use and (hour == cfg.ANIMATION_FRAME_START_HOUR or fixed_lims_3d_for_anim is None or 1): # Example: cbar on first frame or if not part of global animation
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        sm = ScalarMappable(cmap=cmap_obj_to_use, norm=norm_obj_to_use)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical',
                            extend='both' if norm_obj_to_use.clip else 'neither')
        cbar.set_label(cbar_label_str_val, fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
        cbar.ax.tick_params(labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)

    filename_suffix = "q" if plot_variable == "specific_humidity" else "temp"
    frame_path = output_dir_for_frames / f"snapshot_3d_{filename_suffix}_hour_{hour:03d}.png"
    plt.savefig(frame_path, bbox_inches='tight', dpi=animation_dpi_val)
    plt.close(fig); gc.collect()
    return frame_path


# --- Static Plot of Individual 3D Trajectories ---
def _plot_one_static_3d_traj(particle_id: int, particle_df: pd.DataFrame,
                             variable_name: str, config_obj, view_angle=(25, -70)):
    """Helper to plot a single static 3D trajectory."""
    output_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.INDIVIDUAL_TRAJ_PLOTS_SUBDIR / "3D" / variable_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    start_dt = config_obj.SIMULATION_START_DATETIME + pd.Timedelta(hours=particle_df['time_step'].min())
    end_dt = config_obj.SIMULATION_START_DATETIME + pd.Timedelta(hours=particle_df['time_step'].max())
    #title = (f"Particle ID: {int(particle_id)} - 3D Trajectory ({variable_name})\n" # Use int(particle_id)
    #         f"{start_dt.strftime(config_obj.DATETIME_PLOT_FORMAT)} to {end_dt.strftime(config_obj.DATETIME_PLOT_FORMAT)}")
    title = (f"Particle ID: {int(particle_id)} - 3D Trajectory ({variable_name})\n"
         f"{start_dt.strftime(config_obj.DATETIME_PLOT_FORMAT)} to {end_dt.strftime(config_obj.DATETIME_PLOT_FORMAT)}")
    #fig.suptitle(f"{main_title_str}\n{time_range_str}", fontsize=12, y=0.98) # Adjust y for spacing         
    ax.set_title(title, fontsize=DEFAULT_FONT_SIZE * FONT_SCALE,y=0.98)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"].update(linewidth=0.2, linestyle='--', color='lightgray', alpha=0.5)
    ax.yaxis._axinfo["grid"].update(linewidth=0.2, linestyle='--', color='lightgray', alpha=0.5)
    ax.zaxis._axinfo["grid"].update(linewidth=0.2, linestyle='--', color='lightgray', alpha=0.5)

    # Calculate limits specific to this particle's trajectory for static plot
    lims = _get_3d_plot_lims(particle_df['longitude'], particle_df['latitude'], particle_df['pressure'],
                             config_obj.FIXED_PLOT_LONLAT_EXTENT_3D, # Use global fixed if set
                             None,#config_obj.FIXED_PLOT_PRESSURE_EXTENT_3D, # Use global fixed if set
                             config_obj.DEFAULT_PLOT_BUFFER_DEG)
    ax.set_xlim(lims['lon'])
    ax.set_ylim(lims['lat'])
    #ax.set_zlim(lims['pres'][1], lims['pres'][0]) # max_p, min_p for inverted axis
    floor_pressure = 1000  # Always use 1000 hPa as floor
    ceiling_pressure = 200  # Always use 200 hPa as ceiling

    ax.set_zlim(floor_pressure, ceiling_pressure)  # (max_p, min_p) for inverted axis
    
    # Set custom ticks
    ax.set_xticks([60, 70, 80, 90, 100])
    ax.set_yticks([-5, 5, 15, 25])
    #pressure_ticks = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    pressure_ticks = [200, 400, 600, 800, 1000]
    ax.set_zticks(pressure_ticks)
    ax.set_zticklabels(pressure_ticks)
    #floor_pressure = lims['pres'][1]

    # --- Floor Rectangle ---
    x_rect = [lims['lon'][0], lims['lon'][1], lims['lon'][1], lims['lon'][0]]
    y_rect = [lims['lat'][0], lims['lat'][0], lims['lat'][1], lims['lat'][1]]
    verts_rect = [list(zip(x_rect, y_rect, [floor_pressure]*4))]
    floor_poly = Poly3DCollection(verts_rect, facecolors=(0.92,0.92,0.92,0.6), 
                                 edgecolors='none', zorder=-100) # Draw first
    ax.add_collection3d(floor_poly)
    
    # Add domain boundary lines to clearly mark the box edges
    # Bottom corners of the box (at floor pressure)
    corners_x = [lims['lon'][0], lims['lon'][1], lims['lon'][1], lims['lon'][0], lims['lon'][0]]
    corners_y = [lims['lat'][0], lims['lat'][0], lims['lat'][1], lims['lat'][1], lims['lat'][0]]
    
    # Draw box edges along the floor
    ax.plot(corners_x, corners_y, floor_pressure, color='black', linestyle='-', 
           linewidth=1.0, alpha=0.7, zorder=-90)

    # --- Coastlines (with proper filtering) ---
    coast = cfeature.NaturalEarthFeature('physical', 'coastline', '110m', 
                                        edgecolor=(0.1,0.1,0.1), facecolor='none')
    
    for geom in coast.geometries():
        geom_b = geom.bounds
        # Quick check if geometry might intersect our domain (optimization)
        if not (geom_b[2] < lims['lon'][0] or geom_b[0] > lims['lon'][1] or 
                geom_b[3] < lims['lat'][0] or geom_b[1] > lims['lat'][1]):
            
            xs, ys = _filter_coastline_coordinates(geom, lims['lon'], lims['lat'])
            if len(xs) > 0:
                ax.plot(xs, ys, zs=floor_pressure, zdir='z', 
                       color=coast.kwargs['edgecolor'], linewidth=0.8, 
                       zorder=-50, alpha=0.7)
    
    # --- CORRECTED CALL to _add_target_volume_3d_worker ---
    #_add_target_volume_3d_worker(ax, lims['pres'], # Use local lims for pressure extent
    #                             config_obj.TARGET_LON_CENTER, 
    #                             config_obj.TARGET_LAT_CENTER,
    #                             config_obj.TARGET_BOX_HALF_WIDTH_DEG, 
    #                             config_obj.TARGET_AREA_PLOT_COLOR)
    
    target_pressure_limits = (ceiling_pressure, floor_pressure)  # (200, 1000)
    _add_target_volume_3d_worker(ax, target_pressure_limits,
                                config_obj.TARGET_LON_CENTER, 
                                config_obj.TARGET_LAT_CENTER,
                                config_obj.TARGET_BOX_HALF_WIDTH_DEG, 
                                config_obj.TARGET_AREA_PLOT_COLOR)
    
    # Trajectory Line
    if variable_name == "Moisture":
        values_for_color = particle_df['specific_humidity'].dropna() * 1000
        cmap_traj_name = config_obj.CMAP_SPECIFIC_HUMIDITY # Get name from cfg
        cbar_label_traj = "Specific Humidity (g/kg) along Trajectory"
    else: # Temperature
        values_for_color = particle_df['temperature'].dropna()
        cmap_traj_name = config_obj.CMAP_TEMPERATURE # Get name from cfg
        cbar_label_traj = "Temperature (K) along Trajectory"
    
    cmap_traj = get_cmap(cmap_traj_name) # Create cmap object

    if not values_for_color.empty and len(particle_df) > 1:
        valid_idx = values_for_color.index # Ensure we only use rows where color value is valid
        # Align all plottable series to these valid indices
        lons_p = particle_df.loc[valid_idx,'longitude']
        lats_p = particle_df.loc[valid_idx,'latitude']
        pres_p = particle_df.loc[valid_idx,'pressure']
        
        # Create segments only from valid, aligned points
        points = np.array([lons_p, lats_p, pres_p]).T.reshape(-1,1,3)
        if len(points) > 1 : # Need at least 2 points for a segment
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Color segments by value at start of segment
            colors_for_segments = values_for_color.loc[lons_p.index[:-1]] # Align colors with start of segments

            if not colors_for_segments.empty:
                norm_traj = Normalize(vmin=np.percentile(values_for_color, 2), 
                                      vmax=np.percentile(values_for_color, 98), clip=True)
                lc = Line3DCollection(segments, cmap=cmap_traj, norm=norm_traj, zorder=5, 
                                      linewidth=2.2, alpha=0.95)
                lc.set_array(colors_for_segments)
                ax.add_collection(lc)

                sm_traj = ScalarMappable(cmap=cmap_traj, norm=norm_traj)
                sm_traj.set_array([])
                cbar_t = fig.colorbar(sm_traj, ax=ax, shrink=0.5, aspect=10, pad=0.08, 
                                      extend='both' if norm_traj.clip else 'neither')
                cbar_t.set_label(cbar_label_traj, fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
                cbar_t.ax.tick_params(labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)

    # Event markers
    if variable_name == "Moisture": event_col, pos_ev, neg_ev, pos_c, neg_c = 'moisture_event_type', 'Uptake', 'Release', 'blue', 'red'
    else: event_col, pos_ev, neg_ev, pos_c, neg_c = 'temp_event_type', 'Warming', 'Cooling', 'orange', 'purple'
    
    pos_df = particle_df[particle_df[event_col] == pos_ev]
    neg_df = particle_df[particle_df[event_col] == neg_ev]
    if not pos_df.empty: ax.scatter(pos_df['longitude'], pos_df['latitude'], pos_df['pressure'], s=35, color=pos_c, marker='^', depthshade=True, edgecolor='k', linewidth=0.3, label=pos_ev, zorder=15)
    if not neg_df.empty: ax.scatter(neg_df['longitude'], neg_df['latitude'], neg_df['pressure'], s=35, color=neg_c, marker='v', depthshade=True, edgecolor='k', linewidth=0.3, label=neg_ev, zorder=15)

    ax.scatter(particle_df['longitude'].iloc[0], particle_df['latitude'].iloc[0], particle_df['pressure'].iloc[0], s=60, color='lime', marker='o', depthshade=False, edgecolor='k', label='Start', zorder=20)
    ax.scatter(particle_df['longitude'].iloc[-1], particle_df['latitude'].iloc[-1], particle_df['pressure'].iloc[-1], s=60, color='black', marker='X', depthshade=False, edgecolor='w', label='End', zorder=20)

    # ax.set_xlabel('Longitude (°E)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE); ax.set_ylabel('Latitude (°N)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE); ax.set_zlabel('Pressure (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    # ax.tick_params(axis='x', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    # ax.tick_params(axis='y', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    # ax.tick_params(axis='z', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    # Set tick parameters with more padding
    ax.tick_params(axis='x', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE, pad=8)
    # ax.tick_params(axis='y', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE, pad=8)
    ax.tick_params(axis='y', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE, pad=-2)
    ax.tick_params(axis='z', labelsize=DEFAULT_FONT_SIZE * FONT_SCALE, pad=10)

    # Set axis labels
    ax.xaxis.set_label_text('Longitude (°E)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.yaxis.set_label_text('Latitude (°N)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)
    ax.zaxis.set_label_text('Pressure (hPa)', fontsize=DEFAULT_FONT_SIZE * FONT_SCALE)

    # Manually position the labels further away
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20  
    ax.zaxis.labelpad = 20
    # ax.invert_zaxis()
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.95])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=4, fontsize=DEFAULT_FONT_SIZE * FONT_SCALE, frameon=True, facecolor=(1,1,1,0.7))

    plt.savefig(output_dir / f"particle_{int(particle_id)}_{variable_name.lower()}_3D_trajectory_static.png", bbox_inches='tight', dpi=150)
    plt.close(fig); gc.collect()


def plot_selected_individual_3d_trajectories(master_df: pd.DataFrame, relevant_ids: list,
                                              variable_name: str, config_obj): # config_obj is passed here
    print(f"Plotting: Selected Individual 3D Static Trajectories for {variable_name}")
    if master_df.empty or not relevant_ids: return

    ids_to_plot = relevant_ids
    if config_obj.MAX_INDIVIDUAL_TRAJECTORIES_TO_PLOT is not None and len(relevant_ids) > config_obj.MAX_INDIVIDUAL_TRAJECTORIES_TO_PLOT:
        ids_to_plot = relevant_ids[:config_obj.MAX_INDIVIDUAL_TRAJECTORIES_TO_PLOT]

    # For static plots, multiprocessing might be overkill if MAX_INDIVIDUAL_TRAJECTORIES_TO_PLOT is small.
    # If it's large, you could adapt it to use Pool, passing specific config values to the worker.
    # For now, let's keep it serial for simplicity for this static plotting function.
    for pid in tqdm(ids_to_plot, desc=f"Plotting ind. 3D {variable_name} traj (static)"):
        particle_df = master_df[master_df['particle_id'] == pid]
        if not particle_df.empty and len(particle_df) > 1:
             _plot_one_static_3d_traj(int(pid), particle_df.copy(), variable_name, config_obj) # Pass int(pid) and config_obj
    print(f"Individual static 3D {variable_name} trajectory plots saved.")
    
    
def _filter_coastline_coordinates(geom, lon_lim, lat_lim):
    """
    Filter coastline coordinates to only those within the plotting domain.
    Splits line segments at domain boundaries to ensure proper clipping.
    """
    def clip_line_to_bounds(xs, ys, lon_min, lon_max, lat_min, lat_max):
        """Clip a line to domain boundaries and return segments within bounds"""
        if len(xs) < 2:
            return np.array([]), np.array([])
        
        # Initialize arrays to store clipped segments
        xs_clipped, ys_clipped = [], []
        
        # Process each segment of the line
        for i in range(len(xs)-1):
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[i+1], ys[i+1]
            
            # Check if segment is completely outside bounds
            if ((x1 < lon_min and x2 < lon_min) or 
                (x1 > lon_max and x2 > lon_max) or
                (y1 < lat_min and y2 < lat_min) or
                (y1 > lat_max and y2 > lat_max)):
                continue
                
            # If segment crosses boundary, clip it
            if x1 < lon_min or x1 > lon_max or y1 < lat_min or y1 > lat_max or \
               x2 < lon_min or x2 > lon_max or y2 < lat_min or y2 > lat_max:
                
                # Clip x-coordinates
                if x1 < lon_min:
                    if x2 == x1:  # Vertical line
                        continue
                    t = (lon_min - x1) / (x2 - x1)
                    y1 = y1 + t * (y2 - y1)
                    x1 = lon_min
                elif x1 > lon_max:
                    if x2 == x1:  # Vertical line
                        continue
                    t = (lon_max - x1) / (x2 - x1)
                    y1 = y1 + t * (y2 - y1)
                    x1 = lon_max
                
                if x2 < lon_min:
                    if x2 == x1:  # Vertical line
                        continue
                    t = (lon_min - x1) / (x2 - x1)
                    y2 = y1 + t * (y2 - y1)
                    x2 = lon_min
                elif x2 > lon_max:
                    if x2 == x1:  # Vertical line
                        continue
                    t = (lon_max - x1) / (x2 - x1)
                    y2 = y1 + t * (y2 - y1)
                    x2 = lon_max
                
                # Clip y-coordinates
                if y1 < lat_min:
                    if y2 == y1:  # Horizontal line
                        continue
                    t = (lat_min - y1) / (y2 - y1)
                    x1 = x1 + t * (x2 - x1)
                    y1 = lat_min
                elif y1 > lat_max:
                    if y2 == y1:  # Horizontal line
                        continue
                    t = (lat_max - y1) / (y2 - y1)
                    x1 = x1 + t * (x2 - x1)
                    y1 = lat_max
                
                if y2 < lat_min:
                    if y2 == y1:  # Horizontal line
                        continue
                    t = (lat_min - y1) / (y2 - y1)
                    x2 = x1 + t * (x2 - x1)
                    y2 = lat_min
                elif y2 > lat_max:
                    if y2 == y1:  # Horizontal line
                        continue
                    t = (lat_max - y1) / (y2 - y1)
                    x2 = x1 + t * (x2 - x1)
                    y2 = lat_max
            
            # Final check if clipped points are within bounds
            if (lon_min <= x1 <= lon_max and lat_min <= y1 <= lat_max and
                lon_min <= x2 <= lon_max and lat_min <= y2 <= lat_max):
                xs_clipped.extend([x1, x2])
                ys_clipped.extend([y1, y2])
        
        return np.array(xs_clipped), np.array(ys_clipped)
    
    all_xs, all_ys = [], []
    
    if geom.geom_type == 'LineString':
        xs, ys = geom.xy
        xs, ys = np.array(xs), np.array(ys)
        xs_clip, ys_clip = clip_line_to_bounds(xs, ys, lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1])
        if len(xs_clip) > 0:
            all_xs.extend(xs_clip)
            all_ys.extend(ys_clip)
            
    elif geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            xs, ys = line.xy
            xs, ys = np.array(xs), np.array(ys)
            xs_clip, ys_clip = clip_line_to_bounds(xs, ys, lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1])
            if len(xs_clip) > 0:
                all_xs.extend(xs_clip)
                all_ys.extend(ys_clip)
    
    return np.array(all_xs), np.array(all_ys)