# main_analysis_workflow.py
"""
Main orchestration script for the comprehensive Lagrangian particle analysis.
This script imports configurations and functions from other modules to perform:
1. Relevant particle identification.
2. Extraction of particle histories.
3. Analysis of moisture and temperature changes along trajectories.
4. Generation of various plots and animations.
5. Statistical analysis of event characteristics.
"""

import pandas as pd
from pathlib import Path
import time
import logging

import config as cfg
import data_processing
import plotting_2d
import plotting_3d
import statistical_analysis
import animations
import tracks_analysis
import MoistureTracks

def create_output_directories():
    """
    Creates all necessary output directories as defined in the configuration.
    This function is more comprehensive and directly uses cfg for paths.
    """
    print("--- Setting up Output Directories ---")
    cfg.BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Directories for data processing stages
    cfg.MOISTURE_TEMP_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RELEVANT_IDS_FILE.parent.mkdir(parents=True, exist_ok=True) # Ensures parent of file exists
    cfg.FILTERED_HOURLY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.ANALYZED_PARTICLE_HISTORIES_DIR.mkdir(parents=True, exist_ok=True)

    # Master directory for all plots and animations
    cfg.PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Specific subdirectories for different plot types
    (cfg.PLOTS_OUTPUT_DIR / cfg.INDIVIDUAL_TRAJ_PLOTS_SUBDIR).mkdir(parents=True, exist_ok=True)
    (cfg.PLOTS_OUTPUT_DIR / cfg.AGGREGATE_MAPS_MOISTURE_SUBDIR).mkdir(parents=True, exist_ok=True)
    (cfg.PLOTS_OUTPUT_DIR / cfg.AGGREGATE_MAPS_TEMP_SUBDIR).mkdir(parents=True, exist_ok=True)
    (cfg.PLOTS_OUTPUT_DIR / cfg.HOURLY_SNAPSHOTS_2D_FRAMES_SUBDIR).mkdir(parents=True, exist_ok=True)
    (cfg.PLOTS_OUTPUT_DIR / cfg.HOURLY_SNAPSHOTS_3D_FRAMES_SUBDIR).mkdir(parents=True, exist_ok=True)
    (cfg.PLOTS_OUTPUT_DIR / cfg.ANIMATIONS_SUBDIR).mkdir(parents=True, exist_ok=True)
    (cfg.PLOTS_OUTPUT_DIR / cfg.STATISTICAL_DISTRIBUTIONS_MOISTURE_SUBDIR).mkdir(parents=True, exist_ok=True)
    (cfg.PLOTS_OUTPUT_DIR / cfg.STATISTICAL_DISTRIBUTIONS_TEMP_SUBDIR).mkdir(parents=True, exist_ok=True)
    (cfg.PLOTS_OUTPUT_DIR / cfg.TARGET_AREA_ANALYSIS_SUBDIR).mkdir(parents=True, exist_ok=True)
    (cfg.PLOTS_OUTPUT_DIR / cfg.TIME_EVOLUTION_SUBDIR).mkdir(parents=True, exist_ok=True)

    print(f"All output directories established under: {cfg.BASE_OUTPUT_DIR}")


def main():
    """
    Main workflow function.
    """
    start_time_total = time.time()
    print("===== STARTING COMPREHENSIVE PARTICLE ANALYSIS WORKFLOW ====")

    # 1. Create all output directories
    create_output_directories()

    # 2. Calculate derived configuration values
    if cfg.RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS:
        latest_arrival_step = max(cfg.RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS)
        cfg.MAX_HISTORY_TRACKING_HOUR = latest_arrival_step + cfg.TRACK_HISTORY_WINDOW_AFTER_MAX_ARRIVAL_HOURS
    else: # Should not happen if properly configured, but a fallback
        cfg.MAX_HISTORY_TRACKING_HOUR = cfg.ANALYSIS_START_HOUR + 24 # Default fallback
    print(f"Max history tracking hour calculated: {cfg.MAX_HISTORY_TRACKING_HOUR}")

    relevant_ids = []
    master_df = pd.DataFrame()

    # --- Check if we can skip ahead ---
    if cfg.START_WORKFLOW_FROM_STAGE > 1:
        print(f"--- ATTEMPTING TO START FROM STAGE {cfg.START_WORKFLOW_FROM_STAGE} ---")
        if not cfg.RELEVANT_IDS_FILE.exists():
            print(f"CRITICAL: Cannot skip to stage {cfg.START_WORKFLOW_FROM_STAGE}. Required file not found: {cfg.RELEVANT_IDS_FILE}")
            return
        print(f"Loading relevant_ids from {cfg.RELEVANT_IDS_FILE}")
        relevant_ids_df = pd.read_csv(cfg.RELEVANT_IDS_FILE)
        relevant_ids = relevant_ids_df['particle_id'].tolist()
        print(f"Loaded {len(relevant_ids)} relevant IDs.")

    if cfg.START_WORKFLOW_FROM_STAGE > 4:
        if not cfg.MASTER_DF_FILE.exists():
            print(f"CRITICAL: Cannot skip to stage {cfg.START_WORKFLOW_FROM_STAGE}. Required file not found: {cfg.MASTER_DF_FILE}")
            return
        print(f"Loading master_df from {cfg.MASTER_DF_FILE}")
        master_df = pd.read_parquet(cfg.MASTER_DF_FILE)
        print(f"Loaded master DataFrame. Shape: {master_df.shape}")


    # --- STAGE 1: Identify Relevant Particle IDs ---
    if cfg.START_WORKFLOW_FROM_STAGE <= 1:
        stage_start_time = time.time()
        print("\n--- STAGE 1: Identifying Relevant Particle IDs ---")
        relevant_ids = data_processing.identify_target_particles_from_augmented_data(
            augmented_data_dir=cfg.AUGMENTED_PARTICLE_DATA_DIR,
            event_steps=cfg.RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS,
            target_lat_center=cfg.TARGET_LAT_CENTER,
            target_lon_center=cfg.TARGET_LON_CENTER,
            target_box_half_width_deg=cfg.TARGET_BOX_HALF_WIDTH_DEG,
            output_ids_file=cfg.RELEVANT_IDS_FILE
        )
        if not relevant_ids:
            print("CRITICAL: No relevant particles identified based on configuration. Exiting.")
            return
        print(f"Stage 1 completed in {time.time() - stage_start_time:.2f} seconds.")

    # --- STAGE 2: Extract Full Histories for Relevant Particles ---
    if cfg.START_WORKFLOW_FROM_STAGE <= 2:
        stage_start_time = time.time()
        print(f"\n--- STAGE 2: Extracting Histories for {len(relevant_ids)} Relevant Particles ---")
        data_processing.extract_particle_histories(
            relevant_ids=relevant_ids,
            augmented_data_dir=cfg.AUGMENTED_PARTICLE_DATA_DIR,
            filtered_output_dir=cfg.FILTERED_HOURLY_DATA_DIR,
            start_analysis_hour=cfg.ANALYSIS_START_HOUR,
            max_history_hour=cfg.MAX_HISTORY_TRACKING_HOUR # Use calculated max hour
        )
        print(f"Stage 2 completed in {time.time() - stage_start_time:.2f} seconds.")

    # --- STAGE 3: Analyze Histories for Moisture & Temperature Changes ---
    if cfg.START_WORKFLOW_FROM_STAGE <= 3:
        stage_start_time = time.time()
        print("\n--- STAGE 3: Analyzing Particle Histories (dq/dt, dT/dt, Events) ---")
        data_processing.analyze_all_particle_histories_detailed(
            relevant_ids=relevant_ids,
            filtered_input_dir=cfg.FILTERED_HOURLY_DATA_DIR,
            analyzed_output_dir=cfg.ANALYZED_PARTICLE_HISTORIES_DIR,
            max_history_hour=cfg.MAX_HISTORY_TRACKING_HOUR, # Max hour of filtered files to read
            dq_threshold=cfg.DQ_THRESHOLD_KG_PER_KG,
            dt_threshold=cfg.DT_THRESHOLD_KELVIN,
            change_window=cfg.CHANGE_WINDOW_HOURS,
            num_workers=cfg.NUM_WORKERS
        )
        print(f"Stage 3 completed in {time.time() - stage_start_time:.2f} seconds.")

    # --- STAGE 4: Load Master DataFrame for Plotting & Further Analysis ---
    if cfg.START_WORKFLOW_FROM_STAGE <= 4:
        stage_start_time = time.time()
        print("\n--- STAGE 4: Loading All Analyzed Data into Master DataFrame ---")
        master_df = data_processing.load_master_analyzed_df(
            analyzed_histories_dir=cfg.ANALYZED_PARTICLE_HISTORIES_DIR,
            relevant_ids_list=relevant_ids # Load only the histories we processed
        )
        if master_df.empty:
            print("CRITICAL: Master DataFrame is empty after loading processed histories. Cannot proceed with plotting. Exiting.")
            return
        print(f"Master DataFrame loaded. Shape: {master_df.shape}")
        # Save the master_df to a file to allow skipping this stage in the future
        print(f"Saving master DataFrame to {cfg.MASTER_DF_FILE}...")
        master_df.to_parquet(cfg.MASTER_DF_FILE)
        print(f"Stage 4 completed in {time.time() - stage_start_time:.2f} seconds.")


    # --- STAGE 5: Generate Tracks and Save Results ---
    if cfg.START_WORKFLOW_FROM_STAGE <= 5:
        print("\n--- STAGE 5: Running Independent Tracks Analysis  ---")
        tracks_stage_start_time = time.time()
        #tracks_analysis.run_tracks_analysis_for_all_steps(cfg)
        tracks_analysis.run_tracks_analysis_from_master(master_df, cfg) # Pass master_df
        print(f"Stage 5 (Tracks Analysis) completed in {time.time() - tracks_stage_start_time:.2f} seconds.")

    
    # --- STAGE 6: Generate Plots and Animations ---
    if cfg.START_WORKFLOW_FROM_STAGE <= 6:
        print("\n--- STAGE 6: Generating Visualizations and Statistical Plots ---")
        plotting_stage_start_time = time.time()

        # --- 2D Static Plots ---
        print("\n   --- Generating 2D Aggregate Maps (Moisture) ---")
        plotting_2d.plot_aggregate_map_generic(master_df, "AllEvents", 'dq_dt', "Moisture", cfg, output_suffix_ext="net_change")
        plotting_2d.plot_aggregate_map_generic(master_df, "Uptake", 'dq_dt', "Moisture", cfg, is_frequency=True, output_suffix_ext="uptake_freq")
        plotting_2d.plot_aggregate_map_generic(master_df, "Release", 'dq_dt', "Moisture", cfg, is_frequency=True, output_suffix_ext="release_freq")
        
        print("\n   --- Generating 2D Aggregate Maps (Temperature) ---")
        plotting_2d.plot_aggregate_map_generic(master_df, "AllEvents", 'dT_dt', "Temperature", cfg, output_suffix_ext="net_change")
        plotting_2d.plot_aggregate_map_generic(master_df, "Warming", 'dT_dt', "Temperature", cfg, is_frequency=True, output_suffix_ext="warming_freq")
        plotting_2d.plot_aggregate_map_generic(master_df, "Cooling", 'dT_dt', "Temperature", cfg, is_frequency=True, output_suffix_ext="cooling_freq")

        print("\n   --- Generating Composite Trajectory Density Plot ---")
        plotting_2d.plot_composite_trajectory_density(master_df, cfg) # Expects a specific func name


        # --- Statistical Analysis & Plots ---
        print("\n   --- Generating Statistical Distributions (Moisture) ---")
        statistical_analysis.plot_vertical_profile_change(master_df, 'dq_dt', "Moisture", cfg)
        statistical_analysis.plot_histogram_event_magnitudes(master_df, 'dq_dt', "Moisture", cfg)
        statistical_analysis.plot_last_event_points(master_df, relevant_ids, 'Uptake', "Moisture", cfg) # Pass relevant_ids
        
        print("\n   --- Generating Statistical Distributions (Temperature) ---")
        statistical_analysis.plot_vertical_profile_change(master_df, 'dT_dt', "Temperature", cfg)
        statistical_analysis.plot_histogram_event_magnitudes(master_df, 'dT_dt', "Temperature", cfg)
        statistical_analysis.plot_last_event_points(master_df, relevant_ids, 'Warming', "Temperature", cfg) # Pass relevant_ids

        print("\n   --- Generating Target Area Analysis Plots ---")
        statistical_analysis.analyze_and_plot_time_in_target(master_df, relevant_ids, cfg) # Pass relevant_ids
        statistical_analysis.analyze_and_plot_release_in_target(master_df, relevant_ids, "Moisture", cfg) # Pass relevant_ids
        statistical_analysis.analyze_and_plot_release_in_target(master_df, relevant_ids, "Temperature", cfg) # Pass relevant_ids

        statistical_analysis.plot_vertical_profile_in_target(master_df, 'specific_humidity', "Specific Humidity", cfg)
        statistical_analysis.plot_vertical_profile_in_target(master_df, 'temperature', "Temperature", cfg)
        statistical_analysis.plot_vertical_profile_in_target(master_df, 'dq_dt', "Net Moisture Change (dq/dt)", cfg) # Example for dq_dt
        statistical_analysis.plot_vertical_profile_in_target(master_df, 'dT_dt', "Net Temp. Change (dT/dt)", cfg)   # Example for dT_dt
        
        # Other Analysis
        # a. Profile of 6-hourly change vs. 6hr Mean Pressure (All Trajectories)
        statistical_analysis.plot_vertical_profile_change_vs_mean_pressure(master_df, "Moisture", cfg, time_window_hrs=6)
        statistical_analysis.plot_vertical_profile_change_vs_mean_pressure(master_df, "Temperature", cfg, time_window_hrs=6)

        # b. Profile of 1-hourly change vs. 1hr Mean Pressure (All Trajectories)
        statistical_analysis.plot_vertical_profile_change_vs_mean_pressure(master_df, "Moisture", cfg, time_window_hrs=1)
        statistical_analysis.plot_vertical_profile_change_vs_mean_pressure(master_df, "Temperature", cfg, time_window_hrs=1)
        
        # Also, the "Vertical Profile of Avg Net Moisture Change (Particles within Target)"
        # should now use the 1-hourly change and be binned by current pressure (as per `plot_vertical_profile_in_target`)
        # Modify the call in main_analysis_workflow for this specific plot:
        # Example: (This replaces the previous call to plot_vertical_profile_in_target for dq_dt)
        statistical_analysis.plot_vertical_profile_in_target(master_df, 'dq_dt_1hr', "Net Moisture Change (1hr)", cfg)
        statistical_analysis.plot_vertical_profile_in_target(master_df, 'dT_dt_1hr', "Net Temp. Change (1hr)", cfg)


        # c. Conditional Vertical Profiles (In Target, using 6hr change for now, could be 1hr)
        statistical_analysis.plot_conditional_vertical_profiles_in_target(master_df, "Moisture", cfg, time_window_hrs=6)
        statistical_analysis.plot_conditional_vertical_profiles_in_target(master_df, "Temperature", cfg, time_window_hrs=6)
        # You can also call this with time_window_hrs=1 if desired for 1-hr changes

        # d. Difference Profiles (Entry vs. Exit in Target)
        statistical_analysis.plot_in_target_change_profile(master_df, relevant_ids, "Moisture", cfg)
        statistical_analysis.plot_in_target_change_profile(master_df, relevant_ids, "Temperature", cfg)

        # e. q_current vs. q_lagged Scatter Plot (In Target)
        # This will now run for 2, 4, and 6-hour windows by default.
        statistical_analysis.plot_q_vs_q_lagged_in_target(master_df, "Moisture", cfg)
        statistical_analysis.plot_q_vs_q_lagged_in_target(master_df, "Temperature", cfg)    
        

        print("\n   --- Generating Time Evolution Plots ---")
        statistical_analysis.plot_time_evolution_multi_step(master_df, relevant_ids, "Moisture", cfg) # Pass relevant_ids
        statistical_analysis.plot_time_evolution_multi_step(master_df, relevant_ids, "Temperature", cfg) # Pass relevant_ids
        statistical_analysis.plot_time_evolution_full_event(master_df, relevant_ids, "Moisture", cfg) # Pass relevant_ids
        statistical_analysis.plot_time_evolution_full_event(master_df, relevant_ids, "Temperature", cfg) # Pass relevant_ids

        # --- Individual Trajectory Plots (2D and 3D examples) ---
        print("\n   --- Generating Individual Trajectory Examples ---")
        # These might be moved to plotting_2d and plotting_3d modules respectively
        plotting_2d.plot_selected_individual_2d_trajectories(master_df, relevant_ids, "Moisture", cfg)
        plotting_2d.plot_selected_individual_2d_trajectories(master_df, relevant_ids, "Temperature", cfg)
        plotting_3d.plot_selected_individual_3d_trajectories(master_df, relevant_ids, "Moisture", cfg)
        # plotting_3d.plot_selected_individual_3d_trajectories(master_df, relevant_ids, "Temperature", cfg) # If desired

        # --- Animations ---
        print("\n   --- Generating 2D and 3D Animations ---")
        animations.create_all_animations(master_df, cfg)
        
        print(f"Stage 6 (Plotting & Animations) completed in {time.time() - plotting_stage_start_time:.2f} seconds.")

    # --- STAGE 7: Generate Moisture-Releasing Tracks Plots ---
    if cfg.START_WORKFLOW_FROM_STAGE <= 7:
        print("\n--- STAGE 7: Analyzing and Plotting Moisture-Releasing Tracks ---")
        stage_start_time = time.time()
        MoistureTracks.run_moisture_tracks_analysis(master_df, cfg)
        print(f"Stage 7 completed in {time.time() - stage_start_time:.2f} seconds.")

    print(f"\n===== COMPREHENSIVE ANALYSIS WORKFLOW COMPLETED =====")
    print(f"Total execution time: {time.time() - start_time_total:.2f} seconds.")
    print(f"All outputs saved in subdirectories of: {cfg.BASE_OUTPUT_DIR}")


if __name__ == '__main__':
    # Ensure main() is called only when the script is executed directly
    main()
