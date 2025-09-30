# Hybrid-Parcel-Tracker---Post-Processing Toolbox

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A comprehensive post-processing toolbox for analyzing Lagrangian particle trajectories from atmospheric simulations. This toolkit processes pre-computed particle data (from tools like the [Hybrid-Parcel-Tracker---Main](https://github.com/ManishShukla001/Hybrid-Parcel-Tracker---Main)) to analyze moisture and temperature changes along trajectories, identify key atmospheric events, and generate extensive visualizations and statistical insights. For detailed technical documentation, workflow explanations, and visualization examples, see the [Info Page](https://manishshukla001.github.io/Hybrid-Parcel-Tracker---Post/info_page/info.html).

## Overview

This post-processing suite enables detailed analysis of atmospheric particle behavior after a hybrid particle tracking simulation. Key analyses include:

- **Event Classification**: Identification of moisture uptake/release and warming/cooling events
- **Trajectory Visualization**: 2D and 3D static plots and animations of particle paths
- **Statistical Distributions**: Vertical profiles, histograms, and geographic distributions
- **Source-Receptor Analysis**: Mapping of moisture sources and transport patterns
- **Time Evolution**: Temporal analysis of atmospheric variables aligned by particle arrival times
- **Target Area Focus**: In-depth analysis of particle behavior within specific geographical regions

## Features

### Core Functionality
- **Staged Workflow**: Modular processing from particle selection to final visualization
- **Parallel Processing**: Multi-worker support for efficient handling of large datasets
- **Flexible Input**: Supports CSV and NetCDF input formats from particle simulations
- **Event Detection**: Configurable thresholds for identifying significant atmospheric changes
- **Rich Visualizations**: 2D/3D plots, time series, and animations (GIF/MP4)

### Analysis Types
- **Aggregate Maps**: Spatial distributions of events and net changes
- **Individual Trajectories**: Detailed path visualizations with event overlays
- **Statistical Profiles**: Vertical and temporal distributions of variables
- **Tracks Analysis**: Source region identification and property evolution
- **Time Evolution**: Ensemble analysis aligned by target arrival times
- **Target Area Statistics**: Behavior analysis within defined regions

### Output Formats
- PNG plots for static visualizations
- GIF and MP4 animations
- CSV data files with processed results
- Comprehensive metadata and session logs

## Installation

### Prerequisites
- Python 3.7+
- Access to processed particle data (CSV/NetCDF format) from a particle tracking simulation

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ManishShukla001/Hybrid-Parcel-Tracker---Post.git
   cd Hybrid-Parcel-Tracker---Post
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the analysis** (see Configuration section below)

4. **Prepare your data**:
   - Place your augmented particle data files in the directory specified by `AUGMENTED_PARTICLE_DATA_DIR` in `config.py`
   - Ensure data includes: particle_id, time_step, latitude, longitude, pressure, specific_humidity, temperature

## Usage

### Quick Start

1. **Edit configuration**:
   ```python
   # config.py
   BASE_OUTPUT_DIR = Path("./your_output_directory")
   AUGMENTED_PARTICLE_DATA_DIR = Path("./your_particle_data")
   TARGET_LAT_CENTER = 8.5683  # Your target latitude
   TARGET_LON_CENTER = 78.1238  # Your target longitude
   # ... adjust other parameters as needed
   ```

2. **Run the analysis**:
   ```bash
   python main_analysis_workflow.py
   ```

### Workflow Stages

The analysis runs through several stages, controllable via `START_WORKFLOW_FROM_STAGE`:

1. **Stage 1**: Identify relevant particle IDs that pass through target region
2. **Stage 2**: Extract full histories for selected particles
3. **Stage 3**: Calculate moisture/temperature changes and classify events
4. **Stage 4**: Load master dataset for plotting
5. **Stage 5**: Run tracks analysis (source regions and pathways)
6. **Stage 6**: Generate plots, statistical analysis, and animations
7. **Stage 7**: Analyze moisture-releasing tracks

### Selective Execution

To skip to a later stage (e.g., if data is already processed):

```python
# config.py
START_WORKFLOW_FROM_STAGE = 4  # Start from data loading
```

## Configuration

All analysis parameters are controlled through `config.py`. Key settings include:

### Core Settings
- `BASE_OUTPUT_DIR`: Output directory for all results
- `AUGMENTED_PARTICLE_DATA_DIR`: Path to input particle data
- `SIMULATION_START_DATETIME`: Base datetime (e.g., `'2023-12-15 00:00:00'`)

### Target Definition
- `TARGET_LAT_CENTER`, `TARGET_LON_CENTER`: Geographic center of interest
- `TARGET_BOX_HALF_WIDTH_DEG`: Search radius around target
- `RELEVANT_PARTICLE_TARGET_ARRIVAL_STEPS`: Time steps for particle selection

### Analysis Parameters
- `DQ_THRESHOLD_KG_PER_KG`: Moisture change threshold (e.g., 0.0002 kg/kg)
- `DT_THRESHOLD_KELVIN`: Temperature change threshold (e.g., 1.0 K)
- `CHANGE_WINDOW_HOURS`: Time window for change calculations (typically 6 hours)
- `ANALYSIS_START_HOUR`: Starting point for trajectory analysis

### Visualization Settings
- `FIXED_PLOT_EXTENT_2D`: Geographic bounds for 2D plots
- `CMAP_*`: Colormap configurations
- `ANIMATION_FPS`: Animation speed
- `MAX_INDIVIDUAL_TRAJECTORIES_TO_PLOT`: Limit for detailed trajectory plots

### Performance
- `NUM_WORKERS`: Number of parallel processes

See `config.py` comments for detailed documentation of all parameters.

## Expected Outputs

Results are organized in subdirectories under `BASE_OUTPUT_DIR`:

```
your_output_directory/
├── 1_processed_data/
│   ├── filtered_hourly_data/          # Filtered particle trajectories
│   ├── analyzed_particle_histories/   # Analyzed data with changes/events
│   └── relevant_particle_ids_for_analysis.csv
├── 2_plots_and_animations/
│   ├── aggregate_maps/                 # 2D spatial distributions
│   ├── individual_trajectories/        # Individual particle paths
│   ├── statistical_distributions/      # Vertical profiles, histograms
│   ├── target_area_analysis/           # Targeted region statistics
│   ├── time_evolution/                 # Temporal analysis
│   ├── animation_frames/               # Individual animation frames
│   └── animations_output/              # GIF/MP4 animations
└── tracks_analysis_output/             # Source region analysis
```

## Dependencies

Core Python packages required (see `requirements.txt`):

- `pandas`, `numpy`: Data processing
- `matplotlib`: Plotting (with `mpl_toolkits.mplot3d`)
- `cartopy`: Geographic projections and coastlines
- `scipy`: Statistical functions
- `xarray`: NetCDF support
- `tqdm`: Progress bars
- `imageio`, `imageio-ffmpeg`: Animation creation

## Project Structure

```
├── main_analysis_workflow.py       # Main orchestration script
├── config.py                       # Configuration parameters
├── data_processing.py              # Data loading and preprocessing
├── plotting_2d.py                  # 2D visualization functions
├── plotting_3d.py                  # 3D visualization functions
├── statistical_analysis.py         # Statistical analysis suite
├── animations.py                   # Animation generation
├── tracks_analysis.py              # Tracks-based analysis
├── MoistureTracks.py               # Moisture-specific analysis
├── requirements.txt                # Python dependencies
├── info_page/                      # Documentation and diagrams
└── README.md
```

## Authors

Manish Shukla - Postdoctoral student, Indian Institute of Technology Hyderabad ([manishshukla01@live.com](mailto:manishshukla01@live.com))  
R. Maheshwaran - Assistant Professor, Indian Institute of Technology Hyderabad

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the [Hybrid-Particle-Tracker---Main](https://github.com/ManishShukla001/Hybrid-Particle-Tracker---Main) framework
- This research software was developed at the Indian Institute of Technology Hyderabad for advanced atmospheric modeling and climate research applications. Special thanks to the Copernicus Climate Change Service (C3S) for providing the ERA5 reanalysis dataset and National Centre for Medium Range Weather Forecasting (NCMWRF) for IMDAA reanalysis dataset.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation


## Support

For questions or issues, please open a GitHub issue or contact the maintainers.
