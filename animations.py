# animations.py
"""
Module for creating 2D and 3D animations of particle snapshots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from pathlib import Path
from tqdm import tqdm
import gc
import imageio
from multiprocessing import Pool

import config as cfg

try:
    import plotting_2d
    from plotting_2d import generate_2d_snapshot_frame
    import plotting_3d
    from plotting_3d import generate_3d_snapshot_frame
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import frame generation functions from plotting modules: {e}")
    print("Ensure plotting_2d.py and plotting_3d.py exist and contain the necessary functions.")
    def generate_2d_snapshot_frame(*args, **kwargs):
        print("Dummy generate_2d_snapshot_frame called. Plotting module likely missing.")
        return None
    def generate_3d_snapshot_frame(*args, **kwargs):
        print("Dummy generate_3d_snapshot_frame called. Plotting module likely missing.")
        return None

def _compile_animation_files(frame_paths: list, output_animation_path_base: Path, fps: int, is_gif: bool = True, is_mp4: bool = True):
    """
    Compiles a list of image frame paths into GIF and/or MP4 animation.
    """
    valid_frames = [fp for fp in frame_paths if fp and Path(fp).exists()]
    if not valid_frames:
        print(f"No valid frames found to compile animation for {output_animation_path_base.stem}.")
        return

    if is_gif:
        gif_path = output_animation_path_base.with_suffix(".gif")
        print(f"Creating GIF animation at {gif_path}...")
        try:
            with imageio.get_writer(gif_path, mode='I', fps=fps, loop=0) as writer: # loop=0 for infinite loop
                for filename in tqdm(valid_frames, desc=f"Writing {gif_path.name}"):
                    image = imageio.imread(filename)
                    writer.append_data(image)
            print(f"GIF Animation created: {gif_path}")
        except Exception as e:
            print(f"Error creating GIF {gif_path}: {e}")
            print("  Ensure imageio and imageio-ffmpeg are installed if making MP4s, or that image files are valid.")

    if is_mp4:
        mp4_path = output_animation_path_base.with_suffix(".mp4")
        print(f"Creating MP4 animation at {mp4_path}...")
        try:
            with imageio.get_writer(mp4_path, mode='I', fps=fps, format='FFMPEG', codec='h264', quality=7) as writer: # quality 1-10
                for filename in tqdm(valid_frames, desc=f"Writing {mp4_path.name}"):
                    image = imageio.imread(filename)
                    writer.append_data(image)
            print(f"MP4 Animation created: {mp4_path}")
        except Exception as e:
            print(f"Error creating MP4 {mp4_path}: {e}")
            print("  Ensure ffmpeg is installed and accessible in your system PATH for MP4 output.")
            print("  Consider installing with: pip install imageio-ffmpeg")


def _generate_static_colorbar(norm_obj, cmap_name, label_str, output_path: Path, config_obj):
    """Generates and saves a static colorbar image."""
    fig_cbar, ax_cbar = plt.subplots(figsize=(1.5, 5)) # Adjusted size for better proportions
    plt.subplots_adjust(left=0.1, right=0.4, top=0.95, bottom=0.05) # Fine-tune margins
    
    cmap_obj = get_cmap(cmap_name)
    sm = ScalarMappable(cmap=cmap_obj, norm=norm_obj)
    sm.set_array([]) # Important!

    cbar = plt.colorbar(sm, cax=ax_cbar, orientation='vertical',
                        extend='both' if norm_obj.clip else 'neither') # Use extend based on norm clipping
    cbar.set_label(label_str, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig_cbar); gc.collect()
    print(f"Static colorbar saved: {output_path}")


def create_all_animations(master_df: pd.DataFrame, config_obj):
    """
    Main function to generate all 2D and 3D animations.
    """
    print("\n--- Creating Animations for Hourly Snapshots ---")
    if master_df.empty:
        print("Master DataFrame is empty. Skipping animation generation.")
        return

    animation_output_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.ANIMATIONS_SUBDIR
    animation_output_dir.mkdir(parents=True, exist_ok=True)

    frames_2d_q_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.HOURLY_SNAPSHOTS_2D_FRAMES_SUBDIR / "specific_humidity"
    frames_2d_t_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.HOURLY_SNAPSHOTS_2D_FRAMES_SUBDIR / "temperature"
    frames_3d_q_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.HOURLY_SNAPSHOTS_3D_FRAMES_SUBDIR / "specific_humidity"
    frames_3d_t_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.HOURLY_SNAPSHOTS_3D_FRAMES_SUBDIR / "temperature"
    frames_2d_q_dir.mkdir(parents=True, exist_ok=True)
    frames_2d_t_dir.mkdir(parents=True, exist_ok=True)
    frames_3d_q_dir.mkdir(parents=True, exist_ok=True)
    frames_3d_t_dir.mkdir(parents=True, exist_ok=True)


    # --- Determine Global Normalizations and Extents for Animations ---
    print("Calculating global extents and normalization for animations...")
    # For 2D animations
    lon_min_2d, lon_max_2d, lat_min_2d, lat_max_2d = plotting_2d._get_plot_extent(
        master_df['longitude'], master_df['latitude'],
        config_obj.FIXED_PLOT_EXTENT_2D, config_obj.DEFAULT_PLOT_BUFFER_DEG
    )
    fixed_extent_2d_anim = (lon_min_2d, lon_max_2d, lat_min_2d, lat_max_2d)
    
    # For 3D animations, use FIXED_PLOT_LONLAT_EXTENT_3D if available
    if config_obj.FIXED_PLOT_LONLAT_EXTENT_3D:
        lon_lim_3d_anim = (config_obj.FIXED_PLOT_LONLAT_EXTENT_3D[0], config_obj.FIXED_PLOT_LONLAT_EXTENT_3D[1])
        lat_lim_3d_anim = (config_obj.FIXED_PLOT_LONLAT_EXTENT_3D[2], config_obj.FIXED_PLOT_LONLAT_EXTENT_3D[3])
    else: # Fallback to 2D extent if 3D specific is not set
        lon_lim_3d_anim = (lon_min_2d, lon_max_2d)
        lat_lim_3d_anim = (lat_min_2d, lat_max_2d)

    pres_data_anim = master_df['pressure'].dropna()
    pres_min_anim = pres_data_anim.min() if not pres_data_anim.empty else (config_obj.FIXED_PLOT_PRESSURE_EXTENT_3D[0] if config_obj.FIXED_PLOT_PRESSURE_EXTENT_3D else 100)
    pres_max_anim = pres_data_anim.max() if not pres_data_anim.empty else (config_obj.FIXED_PLOT_PRESSURE_EXTENT_3D[1] if config_obj.FIXED_PLOT_PRESSURE_EXTENT_3D else 1000)
    if pres_min_anim >= pres_max_anim: pres_max_anim = pres_min_anim + 100

    fixed_lims_3d_anim = {
        'lon': lon_lim_3d_anim,
        'lat': lat_lim_3d_anim,
        'pres': (pres_min_anim, pres_max_anim)
    }

    # Global normalization for specific humidity
    q_g_kg_all_anim = master_df['specific_humidity'].dropna() * 1000
    if not q_g_kg_all_anim.empty:
        q_min_plot_anim = np.percentile(q_g_kg_all_anim, 0.5)
        q_max_plot_anim = np.percentile(q_g_kg_all_anim, 99.5)
        if q_min_plot_anim >= q_max_plot_anim: q_max_plot_anim = q_min_plot_anim + 1
        q_norm_global_anim = Normalize(vmin=q_min_plot_anim, vmax=q_max_plot_anim, clip=True)
    else:
        q_norm_global_anim = Normalize(vmin=0, vmax=20, clip=True) # Default

    # Global normalization for temperature
    temp_all_anim = master_df['temperature'].dropna()
    if not temp_all_anim.empty:
        t_min_plot_anim = np.percentile(temp_all_anim, 0.5)
        t_max_plot_anim = np.percentile(temp_all_anim, 99.5)
        if t_min_plot_anim >= t_max_plot_anim: t_max_plot_anim = t_min_plot_anim + 1
        t_norm_global_anim = Normalize(vmin=t_min_plot_anim, vmax=t_max_plot_anim, clip=True)
    else:
        t_norm_global_anim = Normalize(vmin=273, vmax=303, clip=True) # Default K

    # --- Generate Static Colorbars ---
    _generate_static_colorbar(q_norm_global_anim, config_obj.CMAP_SPECIFIC_HUMIDITY,
                              'Specific Humidity (g/kg)',
                              animation_output_dir / "colorbar_specific_humidity.png", config_obj)
    _generate_static_colorbar(t_norm_global_anim, config_obj.CMAP_TEMPERATURE,
                              'Temperature (K)',
                              animation_output_dir / "colorbar_temperature.png", config_obj)

    # --- Prepare Tasks for Frame Generation ---
    hours_for_frames = range(config_obj.ANIMATION_FRAME_START_HOUR, config_obj.ANIMATION_FRAME_END_HOUR + 1)
    
    tasks_2d_q, tasks_2d_t, tasks_3d_q, tasks_3d_t = [], [], [], []

    # Define base output directories for frames here, from config_obj
    base_frames_2d_q_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.HOURLY_SNAPSHOTS_2D_FRAMES_SUBDIR / "specific_humidity"
    base_frames_2d_t_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.HOURLY_SNAPSHOTS_2D_FRAMES_SUBDIR / "temperature"
    base_frames_3d_q_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.HOURLY_SNAPSHOTS_3D_FRAMES_SUBDIR / "specific_humidity"
    base_frames_3d_t_dir = config_obj.PLOTS_OUTPUT_DIR / config_obj.HOURLY_SNAPSHOTS_3D_FRAMES_SUBDIR / "temperature"

    # Ensure these directories exist (although generate_..._snapshot_frame might also do it)
    base_frames_2d_q_dir.mkdir(parents=True, exist_ok=True)
    base_frames_2d_t_dir.mkdir(parents=True, exist_ok=True)
    base_frames_3d_q_dir.mkdir(parents=True, exist_ok=True)
    base_frames_3d_t_dir.mkdir(parents=True, exist_ok=True)
    
    default_view_angle_3d = config_obj.DEFAULT_PLOT_VIEW_ANGLE_3D if hasattr(config_obj, 'DEFAULT_PLOT_VIEW_ANGLE_3D') else (25,-70)

    for hour in hours_for_frames:
        df_at_hour = master_df[master_df['time_step'] == hour]
        if not df_at_hour.empty:
            add_cbar_to_first_2d_frame = (hour == config_obj.ANIMATION_FRAME_START_HOUR)
            
            # 2D Specific Humidity - Pass the constructed output_dir
            tasks_2d_q.append((
                hour,                                  # 1. hour
                df_at_hour.copy(),                     # 2. df_at_hour
                base_frames_2d_q_dir,                  # 3. output_dir_for_frames
                fixed_extent_2d_anim,                  # 4. fixed_extent_for_anim
                q_norm_global_anim,                    # 5. q_norm_for_anim
                t_norm_global_anim,                    # 6. t_norm_for_anim (pass even if not used by this plot_variable)
                add_cbar_to_first_2d_frame,            # 7. add_colorbar_to_this_frame
                "specific_humidity",                   # 8. plot_variable
                config_obj.SIMULATION_START_DATETIME.isoformat(), # 9. simulation_start_dt_arg (as ISO string)
                config_obj.DATETIME_PLOT_FORMAT,       # 10. datetime_plot_format
                config_obj.TARGET_AREA_PLOT_COLOR,     # 11. target_area_color
                config_obj.TARGET_LON_CENTER,          # 12. target_lon
                config_obj.TARGET_LAT_CENTER,          # 13. target_lat
                config_obj.TARGET_BOX_HALF_WIDTH_DEG,  # 14. target_half_width
                config_obj.DEFAULT_PLOT_BUFFER_DEG,    # 15. default_plot_buffer
                config_obj.ANIMATION_DPI               # 16. animation_dpi
            ))
            # 2D Temperature - Pass the constructed output_dir
            tasks_2d_t.append((
                hour,
                df_at_hour.copy(),
                base_frames_2d_t_dir,
                fixed_extent_2d_anim,
                q_norm_global_anim,                    # Pass q_norm even if plot_variable is temp
                t_norm_global_anim,                    # Pass t_norm
                add_cbar_to_first_2d_frame,
                "temperature",                         # plot_variable
                config_obj.SIMULATION_START_DATETIME.isoformat(),
                config_obj.DATETIME_PLOT_FORMAT,
                config_obj.TARGET_AREA_PLOT_COLOR,
                config_obj.TARGET_LON_CENTER,
                config_obj.TARGET_LAT_CENTER,
                config_obj.TARGET_BOX_HALF_WIDTH_DEG,
                config_obj.DEFAULT_PLOT_BUFFER_DEG,
                config_obj.ANIMATION_DPI
            ))
            
            # 3D Specific Humidity - Pass the constructed output_dir (and other cfg values)
            # Assuming generate_3d_snapshot_frame will also be modified similarly
            tasks_3d_q.append((
                hour,                                   # 1
                df_at_hour.copy(),                      # 2
                base_frames_3d_q_dir,                   # 3
                fixed_lims_3d_anim,                     # 4
                q_norm_global_anim,                     # 5
                t_norm_global_anim,                     # 6
                default_view_angle_3d,                  # 7
                "specific_humidity",                    # 8
                config_obj.SIMULATION_START_DATETIME.isoformat(), # 9
                config_obj.DATETIME_PLOT_FORMAT,        # 10
                config_obj.TARGET_AREA_PLOT_COLOR,      # 11
                config_obj.TARGET_LON_CENTER,           # 12
                config_obj.TARGET_LAT_CENTER,           # 13
                config_obj.TARGET_BOX_HALF_WIDTH_DEG,   # 14
                config_obj.DEFAULT_PLOT_BUFFER_DEG,     # 15 (or a 3D specific buffer from cfg)
                config_obj.ANIMATION_DPI,               # 16
                config_obj.CMAP_SPECIFIC_HUMIDITY,      # 17
                config_obj.CMAP_TEMPERATURE             # 18
            ))
            # 3D Temperature - Pass the constructed output_dir
            tasks_3d_t.append((
                hour,
                df_at_hour.copy(),
                base_frames_3d_t_dir,
                fixed_lims_3d_anim,
                q_norm_global_anim,
                t_norm_global_anim,
                default_view_angle_3d,
                "temperature",
                config_obj.SIMULATION_START_DATETIME.isoformat(),
                config_obj.DATETIME_PLOT_FORMAT,
                config_obj.TARGET_AREA_PLOT_COLOR,
                config_obj.TARGET_LON_CENTER,
                config_obj.TARGET_LAT_CENTER,
                config_obj.TARGET_BOX_HALF_WIDTH_DEG,
                config_obj.DEFAULT_PLOT_BUFFER_DEG,
                config_obj.ANIMATION_DPI,
                config_obj.CMAP_SPECIFIC_HUMIDITY,
                config_obj.CMAP_TEMPERATURE
            ))
    # --- Generate Frames in Parallel (or serially if NUM_WORKERS=1) ---
    all_task_sets = [
        (tasks_2d_q, generate_2d_snapshot_frame, "2D Specific Humidity Frames"),
        (tasks_2d_t, generate_2d_snapshot_frame, "2D Temperature Frames"),
        (tasks_3d_q, generate_3d_snapshot_frame, "3D Specific Humidity Frames"),
        (tasks_3d_t, generate_3d_snapshot_frame, "3D Temperature Frames"),
    ]

    frame_paths_dict = {} # To store paths for each animation type

    for tasks, frame_func, desc in all_task_sets:
        if not tasks:
            print(f"No tasks for {desc}. Skipping frame generation.")
            frame_paths_dict[desc.replace(" Frames", "")] = []
            continue
        
        print(f"Generating {desc}...")
        current_frame_paths = []
        if config_obj.NUM_WORKERS > 1 and len(tasks) > 1:
            with Pool(processes=min(config_obj.NUM_WORKERS, len(tasks))) as pool:
                # starmap expects a list of tuples, where each tuple contains args for one call
                current_frame_paths = list(tqdm(pool.starmap(frame_func, tasks), total=len(tasks), desc=desc))
        else: # Serial execution
            for task_args in tqdm(tasks, desc=f"{desc} (Serial)"):
                try:
                    current_frame_paths.append(frame_func(*task_args))
                except Exception as e:
                    print(f"Error generating frame for {desc} with args {task_args[:2]}: {e}") # Log error and particle ID/hour
                    current_frame_paths.append(None) # Add None so list length matches tasks
        frame_paths_dict[desc.replace(" Frames", "")] = current_frame_paths


    # --- Compile Animations ---
    print("\n--- Compiling Animations ---")
    _compile_animation_files(frame_paths_dict.get("2D Specific Humidity", []), animation_output_dir / "animation_2D_specific_humidity", config_obj.ANIMATION_FPS)
    _compile_animation_files(frame_paths_dict.get("2D Temperature", []), animation_output_dir / "animation_2D_temperature", config_obj.ANIMATION_FPS)
    _compile_animation_files(frame_paths_dict.get("3D Specific Humidity", []), animation_output_dir / "animation_3D_specific_humidity", config_obj.ANIMATION_FPS)
    _compile_animation_files(frame_paths_dict.get("3D Temperature", []), animation_output_dir / "animation_3D_temperature", config_obj.ANIMATION_FPS)

    print("Animation generation process complete.")
