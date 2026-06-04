# Treadmill Biomechanics & Energetics Analyzer

A high-performance Rust engine for batch-processing biomechanical and metabolic data from treadmill exoskeleton trials. 

This tool ingests time-series Parquet data (Ground Reaction Forces, Center of Pressure, Exoskeleton Torques, and Respirometry), applies state-estimation and gait-segmentation algorithms, and outputs cleanly aggregated databases of individual **Strides** and **Breaths**.

## 🚀 Quick Start

Because this pipeline relies on heavy linear algebra (`nalgebra`) and large file decompression (`polars`), **you must run it in release mode**. 

```bash
# Navigate to the rust project folder
cd treadmill_analyze

# Run the analyzer (Highly optimized)
cargo run --release
```

> **⚠️ WARNING: Do not run using `cargo run` (Debug mode)!** 
> Debug mode does not inline the 9x9 Kalman Filter matrix operations, which will result in massive stack memory allocations, severe slowdowns, and potential stack overflows. Always use `--release`.

## 📂 Inputs & Outputs

**Inputs:**
The script automatically searches for processed trial data in the parent directory:
`../exported_pogensee/*.parquet`
*(It automatically ignores precomputed files and previous outputs to prevent recursive duplication).*

**Outputs:**
It generates two master Parquet databases in the same directory:
1. `../exported_pogensee/strides.parquet`: Every individual left/right stride, annotated with mechanical work (Exoskeleton, Human Muscle, Achilles).
2. `../exported_pogensee/breaths.parquet`: Every valid metabolic breath, converted to Watts and scrubbed using physiological filters.

## 🧠 Pipeline Overview

1. **Center of Mass (CoM) Estimation (`com_kf.rs`)**
   Uses a 9D Kalman Filter to continuously estimate the subject's Center of Mass velocity and excursion based on bilateral 3D Ground Reaction Forces (GRFs).
2. **Gait Segmentation (`stride_analyzer.rs`)**
   Detects Heel-Strikes and Toe-Offs using a 30N vertical force threshold. It automatically estimates the instantaneous treadmill belt speed using Center of Pressure (CoP) displacement over time.
3. **Power Calculation (`energy_analyzer.rs`)**
   Calculates the true overground mechanical power by transforming the CoM velocity into the treadmill frame ($P = F_{grf} \cdot v_{overground}$).
4. **Biological Decomposition (`math_utils.rs`)**
   Isolates net human power into two components:
   * **Achilles Power**: Modeled as a passive elastic spring (matching negative loading work with positive push-off work).
   * **Muscle Power**: The remaining active metabolic cost.
5. **Metabolic Filtering (`main.rs`)**
   Converts $\dot{V}O_2$ and $\dot{V}CO_2$ to Watts using the Brockway equation. Applies strict physiological bounds (RER between 0.72 - 1.05) and a **60-second washout filter** to drop breaths that occur during or immediately after a mask leak or talking event.

## ⚙️ Performance & Concurrency Tuning

Processing hundreds of multi-megabyte Parquet files simultaneously can easily overwhelm a system. This project is specifically tuned to maximize CPU usage while strictly bounding RAM usage and preventing thread deadlocks:

* **Bounded File Parallelism (`rayon`)**: 
  The global thread pool is capped at **6 parallel files** (`.num_threads(6)`). This acts as a memory limit, preventing the OS from loading 800+ trials into RAM at once, which would cause severe memory thrashing and swap-file usage.
* **Single-Threaded Polars (`POLARS_MAX_THREADS="1"`)**: 
  Because we are parallelizing at the *file* level using Rayon, we force the Polars Parquet reader to operate single-threaded. If Polars were allowed to use Rayon internally, it would trigger a recursive "work-stealing" loop, continuously pausing current files to open new ones, resulting in an immediate stack overflow.
* **Expanded Thread Stacks**: 
  The Rayon thread stack size is explicitly increased to **8MB** (from the macOS default of 512KB) to comfortably accommodate the dataset allocations and Kalman Filter state matrices.

## 📦 Project Structure

```text
treadmill_analyze/
├── Cargo.toml               # Rust dependencies (Polars, Rayon, Nalgebra)
└── src/
    ├── main.rs              # App entry point, parallel orchestration, data I/O
    ├── com_kf.rs            # 9D Kalman Filter for CoM tracking
    ├── stride_analyzer.rs   # Heel-strike detection & treadmill speed estimation
    ├── energy_analyzer.rs   # Overground power translation
    └── math_utils.rs        # Trapezoidal integration & Achilles modeling
```
