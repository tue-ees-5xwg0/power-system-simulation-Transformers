# Power System Simulation

A comprehensive Python package for simulating, analyzing, and validating power distribution networks. This project is designed for educational and research purposes, providing tools for power flow analysis, network validation, EV penetration studies, N-1 security analysis, and more.

---

## Features
- **Power Flow Simulation**: Run and analyze power flow using real or synthetic data.
- **Network Validation**: Comprehensive checks for input data, topology, and time series consistency.
- **EV Penetration Analysis**: Simulate the impact of electric vehicle charging on the grid.
- **N-1 Security Analysis**: Assess network reliability under single-component failure scenarios.
- **Tap Optimization**: Find optimal transformer tap settings for voltage regulation or loss minimization.
- **Graph Processing**: Analyze and visualize network topology, connectivity, and alternative paths.

---

## Installation

1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd power-system-simulation-Transformers
   ```
2. Install in development mode (with all optional dependencies):
   ```sh
   pip install -e .[dev,example]
   ```
   > **Note:** Re-run this command if you add new dependencies.

---

## Usage

- Place your input data (JSON, Parquet) in the appropriate folders.
- Import and use the modules in your scripts or notebooks, e.g.:
  ```python
  from power_system_simulation.validity_check import ValidatePowerSystemSimulation
  validator = ValidatePowerSystemSimulation(
      input_network_data="path/to/input.json",
      meta_data_str="path/to/meta.json",
      ev_active_power_profile="path/to/ev_profile.parquet",
      active_power_profile="path/to/active_profile.parquet",
      reactive_power_profile="path/to/reactive_profile.parquet"
  )
  ```
- See the `example/` folder for a sample notebook.

---

## Module Overview

### 1. `validity_check.py`
- Validates network data, topology, time series, and load profiles.
- Raises clear exceptions for invalid configurations.
- Ensures the grid is ready for simulation.

### 2. `ev_penetration_module.py`
- Simulates the effect of EV charging on the grid.
- Randomly assigns EVs to households and updates load profiles.
- Returns voltage and line statistics summaries.

### 3. `n1_calculation.py`
- Performs N-1 security analysis (contingency analysis).
- Finds alternative paths and assesses network reliability under failures.

### 4. `optimal_tap.py`
- Optimizes transformer tap settings for loss or voltage deviation minimization.

### 5. `graph_processor.py`
- Provides graph-based analysis of the network.
- Checks connectivity, cycles, and finds alternative paths.
- Supports visualization of the network topology.

### 6. `model_processor.py`
- Loads and processes input data for simulations.
- Summarizes node voltages and line statistics.
- Runs power flow calculations and validates results.

---

## Testing

Run all tests with:
```sh
$env:PYTHONPATH = "src"; pytest tests/
```
Or (on Linux/macOS):
```sh
PYTHONPATH=src pytest tests/
```

---

## Code Style & Quality
- Format code with:
  ```sh
  isort .
  black .
  ```
- Check code quality with:
  ```sh
  pylint power_system_simulation
  ```

---

## Folder Structure

```
power-system-simulation-Transformers/
├── src/
│   └── power_system_simulation/   # Main package code
├── tests/                         # Test files
├── example/                       # Example notebooks
├── .vscode/                       # VSCode settings
└── .github/
    └── workflows/                 # CI configurations
```

- `src/power_system_simulation/`: Main package code (modules described above)
- `tests/`: Unit and integration tests
- `example/`: Example notebooks for demonstration
- `.vscode/`: VSCode IDE settings
- `.github/workflows/`: Continuous integration (CI) configs

---

## Authors
- Andrei Dobre
- Stefan Porfir
- Diana Ionica
