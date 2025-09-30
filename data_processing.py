"""
Module for processing particle trajectory data:
1. Identifying relevant particles.
2. Extracting full histories for these particles.
3. Calculating moisture (dq/dt) and temperature (dT/dt) changes and classifying events.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
from multiprocessing import Pool
from datetime import datetime, timedelta

import config as cfg
import xarray as xr

_first_nc_read_logged = False
_first_csv_read_logged = False


def get_particle_filename_by_date(base_dir: Path, start_datetime: datetime, hour: int) -> Path:
    """
    Returns the correct NetCDF file path for a given simulation hour.
    Works with two naming conventions:
    1. DateTime format: particles_output_YYYYMMDD_HHMMSS.nc
    2. Sequential format: particles_output_NNNN.nc (counting hours from 2023-12-15 00:00:00)
    
    Tries datetime format first, falls back to sequential format if not found.
    """
    # Try datetime format first
    dt = start_datetime + timedelta(hours=hour)
    date_str = dt.strftime('%Y%m%d')
    hhmmss = dt.strftime('%H') + '0000'
    datetime_file = base_dir / f"particles_output_{date_str}_{hhmmss}.nc"
    
    if datetime_file.exists():
        return datetime_file
    
    # Fall back to sequential format
    # Calculate hours from reference date (2023-12-15 00:00:00)
    reference_datetime = datetime(2023, 12, 15, 0, 0, 0)
    current_datetime = start_datetime + timedelta(hours=hour)
    hours_from_reference = int((current_datetime - reference_datetime).total_seconds() / 3600)
    
    sequential_file = base_dir / f"particles_output_{hours_from_reference:04d}.nc"
    return sequential_file


def identify_target_particles_from_augmented_data(augmented_data_dir: Path,
                                                  event_steps: range,
                                                  target_lat_center: float,
                                                  target_lon_center: float,
                                                  target_box_half_width_deg: float,
                                                  output_ids_file: Path) -> list | None:
    """
    Identifies unique particle IDs that pass through the target area during specified event steps,
    reading from augmented particle data files. It ensures particle IDs are handled as integers
    to prevent floating-point comparison issues.
    """
    print("--- Identifying Target-Reaching Particles (from Augmented Data) ---")
    all_relevant_ids = set()

    global _first_nc_read_logged, _first_csv_read_logged

    lat_min = target_lat_center - target_box_half_width_deg
    lat_max = target_lat_center + target_box_half_width_deg
    lon_min = target_lon_center - target_box_half_width_deg
    lon_max = target_lon_center + target_box_half_width_deg

    for target_step in tqdm(event_steps, desc="Scanning event steps for target particles"):
        df_target_hour = None
        nc_file_path = get_particle_filename_by_date(augmented_data_dir, cfg.SIMULATION_START_DATETIME, target_step)
        csv_file_path = augmented_data_dir / f"particles_output_{target_step:04d}.csv"

        try:
            if nc_file_path.exists():
                if not _first_nc_read_logged:
                    print(f"INFO: Reading NetCDF (.nc) input files. First instance: {nc_file_path}")
                    _first_nc_read_logged = True
                ds = xr.open_dataset(nc_file_path)
                df_target_hour = ds.to_dataframe().reset_index()
                ds.close()
            elif csv_file_path.exists():
                if not _first_csv_read_logged:
                    print(f"INFO: Reading CSV (.csv) input files. First instance: {csv_file_path}")
                    _first_csv_read_logged = True
                df_target_hour = pd.read_csv(csv_file_path)

            if df_target_hour is None or df_target_hour.empty:
                continue

            if 'id' in df_target_hour.columns and 'particle_id' not in df_target_hour.columns:
                df_target_hour.rename(columns={'id': 'particle_id'}, inplace=True)

            if 'particle_id' not in df_target_hour.columns:
                print(f"Warning: 'particle_id' column missing in data for target_step {target_step}. Skipping.")
                continue

            # --- FIX: Ensure particle_id is a consistent integer type ---
            df_target_hour['particle_id'] = pd.to_numeric(df_target_hour['particle_id'], errors='coerce')
            df_target_hour.dropna(subset=['particle_id'], inplace=True)
            df_target_hour['particle_id'] = df_target_hour['particle_id'].astype(np.int64)
            # --- END FIX ---

            mask = (
                df_target_hour['latitude'].between(lat_min, lat_max, inclusive='both') &
                df_target_hour['longitude'].between(lon_min, lon_max, inclusive='both')
            )
            ids_in_target = df_target_hour.loc[mask, 'particle_id'].unique()
            all_relevant_ids.update(ids_in_target)
        except Exception as e:
            error_path = nc_file_path if nc_file_path.exists() else csv_file_path
            print(f"Error processing data for target_step {target_step} (Path: {error_path}): {e}")

    if not all_relevant_ids:
        print("No particles found reaching the target area in the specified event steps.")
        return None

    relevant_ids_list = sorted([int(pid) for pid in all_relevant_ids])
    relevant_ids_df = pd.DataFrame(relevant_ids_list, columns=['particle_id'])
    output_ids_file.parent.mkdir(parents=True, exist_ok=True)
    relevant_ids_df.to_csv(output_ids_file, index=False)
    print(f"Found {len(relevant_ids_list)} unique particles. IDs saved to {output_ids_file}")
    return relevant_ids_list

def extract_particle_histories(relevant_ids: list,
                               augmented_data_dir: Path,
                               filtered_output_dir: Path,
                               start_analysis_hour: int,
                               max_history_hour: int):
    """
    Extracts data for relevant_ids from augmented_data_dir for the specified time range
    and saves them into new hourly files in filtered_output_dir. Ensures particle IDs are
    handled as integers.
    """
    print(f"--- Extracting Full Histories for {len(relevant_ids)} Relevant Particles ---")
    if not relevant_ids:
        print("No relevant IDs provided. Skipping history extraction.")
        return

    filtered_output_dir.mkdir(parents=True, exist_ok=True)
    relevant_ids_set = set(relevant_ids) # Set of ints for faster lookups

    global _first_nc_read_logged, _first_csv_read_logged

    print(f"Extracting histories from hour {start_analysis_hour} to {max_history_hour}.")

    for hour in tqdm(range(start_analysis_hour, max_history_hour + 1), desc="Filtering hourly files for histories"):
        df_hour = None
        original_nc_file_path = get_particle_filename_by_date(augmented_data_dir, cfg.SIMULATION_START_DATETIME, hour)
        original_csv_file_path = augmented_data_dir / f"particles_output_{hour:04d}.csv" # Fallback path

        try:
            if original_nc_file_path.exists():
                if not _first_nc_read_logged:
                    print(f"INFO: Reading NetCDF (.nc) input files for history extraction. First instance: {original_nc_file_path}")
                    _first_nc_read_logged = True
                ds = xr.open_dataset(original_nc_file_path)
                df_hour = ds.to_dataframe().reset_index()
                ds.close()
            elif original_csv_file_path.exists():
                if not _first_csv_read_logged:
                    print(f"INFO: Reading CSV (.csv) input files for history extraction. First instance: {original_csv_file_path}")
                    _first_csv_read_logged = True
                df_hour = pd.read_csv(original_csv_file_path)

            if df_hour is None or df_hour.empty:
                continue

            if 'id' in df_hour.columns and 'particle_id' not in df_hour.columns:
                df_hour.rename(columns={'id': 'particle_id'}, inplace=True)
            
            if 'particle_id' not in df_hour.columns:
                print(f"Warning: 'particle_id' column missing in data for hour {hour}. Skipping.")
                continue

            # --- FIX: Ensure particle_id is a consistent integer type ---
            df_hour['particle_id'] = pd.to_numeric(df_hour['particle_id'], errors='coerce')
            df_hour.dropna(subset=['particle_id'], inplace=True)
            df_hour['particle_id'] = df_hour['particle_id'].astype(np.int64)
            # --- END FIX ---

            df_filtered = df_hour[df_hour['particle_id'].isin(relevant_ids_set)]
            if not df_filtered.empty:
                df_filtered_copy = df_filtered.copy()
                df_filtered_copy['time_step'] = hour
                filtered_file_path = filtered_output_dir / f"filtered_particles_hour_{hour:04d}.parquet"
                df_filtered_copy.to_parquet(filtered_file_path, index=False)
            
        except Exception as e:
            error_path = original_nc_file_path if original_nc_file_path.exists() else original_csv_file_path
            print(f"Error processing or filtering file for hour {hour} (path tried: {error_path}): {e}")

    print("Filtered hourly history files created.")


def _analyze_single_particle_history_worker(args_tuple):
    """
    Worker function for multiprocessing. Analyzes a single particle's history,
    ensuring its ID is treated as an integer.
    """
    particle_id, filtered_input_dir, analyzed_output_dir, max_history_hour, \
    dq_threshold, dt_threshold, change_window = args_tuple

    particle_history_frames = []
    try:
        for hour in range(cfg.ANALYSIS_START_HOUR, max_history_hour + 1):
            filtered_file_path = filtered_input_dir / f"filtered_particles_hour_{hour:04d}.parquet"
            if filtered_file_path.exists():
                df_hour_all_particles = pd.read_parquet(filtered_file_path)
                df_particle_at_hour = df_hour_all_particles[df_hour_all_particles['particle_id'] == particle_id]
                if not df_particle_at_hour.empty:
                    particle_history_frames.append(df_particle_at_hour)
    except Exception as e:
        print(f"Error reading filtered files for particle {particle_id}: {e}")
        return f"Error_Read_{particle_id}"

    if not particle_history_frames:
        return f"NoData_{particle_id}"

    df_particle = pd.concat(particle_history_frames).sort_values(by='time_step').reset_index(drop=True)
    df_particle = df_particle.drop_duplicates(subset=['particle_id', 'time_step'], keep='first')

    # --- Moisture Change Analysis ---
    if 'specific_humidity' in df_particle.columns:
        df_particle['specific_humidity'] = pd.to_numeric(df_particle['specific_humidity'], errors='coerce')
        
        # Use a clear, single name for the change calculated over the configured window.
        df_particle['q_lagged'] = df_particle['specific_humidity'].shift(change_window)
        df_particle['dq_dt'] = df_particle['specific_humidity'] - df_particle['q_lagged']

        df_particle['moisture_event_type'] = 'Neutral'
        df_particle.loc[df_particle['dq_dt'] >= dq_threshold, 'moisture_event_type'] = 'Uptake'
        df_particle.loc[df_particle['dq_dt'] <= -dq_threshold, 'moisture_event_type'] = 'Release'
        
        # 1-hourly change
        df_particle['q_lagged_1hr'] = df_particle['specific_humidity'].shift(1)
        df_particle['dq_dt_1hr'] = df_particle['specific_humidity'] - df_particle['q_lagged_1hr']
    else:
        # Ensure all columns exist even if specific_humidity is missing
        df_particle['q_lagged'] = np.nan
        df_particle['dq_dt'] = np.nan
        df_particle['moisture_event_type'] = 'Unknown'
        df_particle['q_lagged_1hr'] = np.nan
        df_particle['dq_dt_1hr'] = np.nan

    # --- Temperature Change Analysis ---
    if 'temperature' in df_particle.columns:
        df_particle['temperature'] = pd.to_numeric(df_particle['temperature'], errors='coerce')

        df_particle['T_lagged'] = df_particle['temperature'].shift(change_window)
        df_particle['dT_dt'] = df_particle['temperature'] - df_particle['T_lagged']

        df_particle['temp_event_type'] = 'Neutral'
        df_particle.loc[df_particle['dT_dt'] >= dt_threshold, 'temp_event_type'] = 'Warming'
        df_particle.loc[df_particle['dT_dt'] <= -dt_threshold, 'temp_event_type'] = 'Cooling'

        # 1-hourly change
        df_particle['T_lagged_1hr'] = df_particle['temperature'].shift(1)
        df_particle['dT_dt_1hr'] = df_particle['temperature'] - df_particle['T_lagged_1hr']
    else:
        # Ensure all columns exist even if temperature is missing
        df_particle['T_lagged'] = np.nan
        df_particle['dT_dt'] = np.nan
        df_particle['temp_event_type'] = 'Unknown'
        df_particle['T_lagged_1hr'] = np.nan
        df_particle['dT_dt_1hr'] = np.nan

    # --- Pressure calculations for mean pressure and vertical motion proxy ---
    if 'pressure' in df_particle.columns:
        df_particle['pressure'] = pd.to_numeric(df_particle['pressure'], errors='coerce')
        
        # For 6-hour window mean pressure
        df_particle['p_lagged_6hr'] = df_particle['pressure'].shift(change_window)
        df_particle['mean_pressure_6hr_window'] = (df_particle['pressure'] + df_particle['p_lagged_6hr']) / 2
        
        # For 1-hour window mean pressure
        df_particle['p_lagged_1hr'] = df_particle['pressure'].shift(1)
        df_particle['mean_pressure_1hr_window'] = (df_particle['pressure'] + df_particle['p_lagged_1hr']) / 2
        
        # For dp_dt_1hr (proxy for vertical motion: negative means ascent)
        df_particle['dp_dt_1hr'] = df_particle['pressure'] - df_particle['p_lagged_1hr']
    else:
        df_particle['p_lagged_6hr'] = np.nan
        df_particle['mean_pressure_6hr_window'] = np.nan
        df_particle['p_lagged_1hr'] = np.nan
        df_particle['mean_pressure_1hr_window'] = np.nan
        df_particle['dp_dt_1hr'] = np.nan

    output_path = analyzed_output_dir / f"particle_{int(particle_id)}_analyzed_history.parquet"
    df_particle.to_parquet(output_path, index=False)
    
    del df_particle
    gc.collect()
    return f"Success_{particle_id}"


def analyze_all_particle_histories_detailed(relevant_ids: list,
                                            filtered_input_dir: Path,
                                            analyzed_output_dir: Path,
                                            max_history_hour: int,
                                            dq_threshold: float,
                                            dt_threshold: float,
                                            change_window: int,
                                            num_workers: int):
    """
    Orchestrates the analysis of individual particle histories using multiprocessing.
    Calculates dq/dt and dT/dt and classifies events.
    """
    print(f"--- Analyzing {len(relevant_ids)} Particle Histories for Moisture & Temperature Changes ---")
    if not relevant_ids:
        print("No relevant IDs provided. Skipping detailed analysis.")
        return

    analyzed_output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for particle_id in relevant_ids:
        tasks.append((
            particle_id, filtered_input_dir, analyzed_output_dir, max_history_hour,
            dq_threshold, dt_threshold, change_window
        ))

    if tasks:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_analyze_single_particle_history_worker, tasks),
                                total=len(tasks), desc="Analyzing particle histories"))

        success_count = sum(1 for r in results if r and r.startswith("Success"))
        print(f"Successfully analyzed {success_count}/{len(relevant_ids)} particle histories.")
        if success_count < len(relevant_ids):
            print("Some particles may have failed analysis or had no data. Check logs/warnings.")
            # for r in results: # For debugging
            #     if not (r and r.startswith("Success")):
            #         print(f"  Issue: {r}")

    print(f"Analyzed particle history Parquet files saved to {analyzed_output_dir}")


def load_master_analyzed_df(analyzed_histories_dir: Path, relevant_ids_list: list = None) -> pd.DataFrame:
    """
    Loads all analyzed particle history Parquet files into a single master DataFrame, ensuring
    particle IDs are handled as integers.
    """
    print(f"--- Loading All Analyzed Histories into Master DataFrame from {analyzed_histories_dir} ---")
    all_dfs = []
    
    files_to_load = []
    if relevant_ids_list:
        # Ensure IDs are integers for filename matching
        for particle_id in relevant_ids_list:
            file_path = analyzed_histories_dir / f"particle_{int(particle_id)}_analyzed_history.parquet"
            if file_path.exists():
                files_to_load.append(file_path)
    else:
        files_to_load = list(analyzed_histories_dir.glob("particle_*_analyzed_history.parquet"))

    if not files_to_load:
        print("No analyzed history files found to load.")
        return pd.DataFrame()

    for file_path in tqdm(files_to_load, desc="Loading analyzed Parquet files"):
        try:
            df = pd.read_parquet(file_path)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load analyzed data file {file_path}: {e}")
    
    if not all_dfs:
        print("No data loaded into master DataFrame.")
        return pd.DataFrame()

    master_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Master DataFrame loaded with {len(master_df)} rows, from {master_df['particle_id'].nunique()} unique particles.")

    # --- FIX: Final cleanup and type enforcement for particle_id ---
    master_df.dropna(subset=['particle_id'], inplace=True)
    master_df['particle_id'] = master_df['particle_id'].astype(np.int64)
    # --- END FIX ---
    
    master_df['time_step'] = pd.to_numeric(master_df['time_step'], errors='coerce')
    for col in ['latitude', 'longitude', 'pressure', 'specific_humidity', 'temperature', 'dq_dt', 'dT_dt']:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
            
    master_df.sort_values(by=['particle_id', 'time_step'], inplace=True)
    master_df.reset_index(drop=True, inplace=True)
    
    return master_df


# Example of how this module might be called (for testing or if run standalone)
if __name__ == '__main__':
    print("Running data_processing.py standalone (for testing purposes)...")
    
    # Setup directories (mirroring what main_analysis_workflow would do for this module)
    cfg.BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MOISTURE_TEMP_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: Identify relevant particles
    relevant_ids = identify_target_particles_from_augmented_data(
        augmented_data_dir=cfg.AUGMENTED_PARTICLE_DATA_DIR,
        event_steps=cfg.RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS,
        target_lat_center=cfg.TARGET_LAT_CENTER,
        target_lon_center=cfg.TARGET_LON_CENTER,
        target_box_half_width_deg=cfg.TARGET_BOX_HALF_WIDTH_DEG,
        output_ids_file=cfg.RELEVANT_IDS_FILE
    )

    if relevant_ids:
        # Calculate max history hour needed based on config
        # This should ideally be passed from the main script or calculated based on cfg
        _max_arrival_step = max(cfg.RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS) if cfg.RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS else 0
        calculated_max_history_hour = _max_arrival_step + cfg.TRACK_HISTORY_WINDOW_AFTER_MAX_ARRIVAL_HOURS

        # Stage 2: Extract histories
        extract_particle_histories(
            relevant_ids=relevant_ids,
            augmented_data_dir=cfg.AUGMENTED_PARTICLE_DATA_DIR,
            filtered_output_dir=cfg.FILTERED_HOURLY_DATA_DIR,
            start_analysis_hour=cfg.ANALYSIS_START_HOUR,
            max_history_hour=calculated_max_history_hour
        )

        # Stage 3: Analyze histories
        analyze_all_particle_histories_detailed(
            relevant_ids=relevant_ids,
            filtered_input_dir=cfg.FILTERED_HOURLY_DATA_DIR,
            analyzed_output_dir=cfg.ANALYZED_PARTICLE_HISTORIES_DIR,
            max_history_hour=calculated_max_history_hour,
            dq_threshold=cfg.DQ_THRESHOLD_KG_PER_KG,
            dt_threshold=cfg.DT_THRESHOLD_KELVIN,
            change_window=cfg.CHANGE_WINDOW_HOURS,
            num_workers=cfg.NUM_WORKERS
        )

        # Stage 4 (Example): Load master df
        master_df_test = load_master_analyzed_df(cfg.ANALYZED_PARTICLE_HISTORIES_DIR, relevant_ids)
        if not master_df_test.empty:
            print("\nTest: Master DataFrame loaded successfully.")
            print(f"Shape: {master_df_test.shape}")
            print(master_df_test.head())
        else:
            print("\nTest: Master DataFrame is empty after processing.")
    else:
        print("Test: No relevant particles found.")

    print("data_processing.py test run finished.")
