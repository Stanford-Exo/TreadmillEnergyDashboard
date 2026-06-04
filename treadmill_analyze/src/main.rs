// File: treadmill_analyze/src/main.rs

use glob::glob;
use nalgebra::Vector3;
use polars::prelude::*;
use rayon::prelude::*;
use std::path::PathBuf;

use treadmill_analyze::energy_analyzer::EnergyAnalyzer;
use treadmill_analyze::math_utils::{extract_biological_components, trapz_1d};

#[derive(Clone, Debug)]
struct StrideRecord {
    trial_name: String,
    leg: String,
    start_time: f64,
    end_time: f64,
    mid_time: f64,
    duration: f64,
    mass_kg: f64,
    is_valid: bool,
    work_exo_pos: f64,
    work_exo_neg: f64,
    work_mus_pos: f64,
    work_mus_neg: f64,
    work_ach_pos: f64,
    work_ach_neg: f64,
}

#[derive(Clone, Debug)]
struct BreathRecord {
    trial_name: String,
    time: f64,
    vo2: f64,
    vco2: f64,
    rer: f64,
    gross_watts: f64,
    net_watts: f64,
    is_valid: bool,
}

// Explicitly casts the column to Float64 to handle Pandas integer optimizations.
fn get_f64_vec(df: &DataFrame, name: &str, len: usize) -> Vec<f64> {
    if let Ok(series) = df.column(name) {
        if let Ok(casted_series) = series.cast(&DataType::Float64) {
            if let Ok(ca) = casted_series.f64() {
                return ca.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
            }
        }
    }
    vec![0.0; len]
}

fn process_trial(file_path: &PathBuf) -> (Vec<StrideRecord>, Vec<BreathRecord>) {
    let file_name = file_path.file_stem().unwrap().to_string_lossy().to_string();
    println!("Processing: {}", file_name);

    let file = std::fs::File::open(file_path).unwrap();
    let df = ParquetReader::new(file).finish().unwrap();

    let times = get_f64_vec(&df, "time", df.height());
    let len = times.len();
    if len == 0 {
        return (vec![], vec![]);
    }

    // Identify Contact Bodies
    let cols: Vec<&str> = df.get_column_names();
    let left_prefix = cols
        .iter()
        .find(|&&c| c.contains("_l_force_y") || c.contains("left_force_y"))
        .map(|c| c.replace("_force_y", ""))
        .unwrap_or("calcn_l".to_string());
    let right_prefix = cols
        .iter()
        .find(|&&c| c.contains("_r_force_y") || c.contains("right_force_y"))
        .map(|c| c.replace("_force_y", ""))
        .unwrap_or("calcn_r".to_string());

    let l_fx = get_f64_vec(&df, &format!("{}_force_x", left_prefix), len);
    let l_fy = get_f64_vec(&df, &format!("{}_force_y", left_prefix), len);
    let l_fz = get_f64_vec(&df, &format!("{}_force_z", left_prefix), len);

    let r_fx = get_f64_vec(&df, &format!("{}_force_x", right_prefix), len);
    let r_fy = get_f64_vec(&df, &format!("{}_force_y", right_prefix), len);
    let r_fz = get_f64_vec(&df, &format!("{}_force_z", right_prefix), len);

    let l_cop_x = get_f64_vec(&df, &format!("{}_cop_x", left_prefix), len);
    let l_cop_y = get_f64_vec(&df, &format!("{}_cop_y", left_prefix), len);
    let l_cop_z = get_f64_vec(&df, &format!("{}_cop_z", left_prefix), len);

    let r_cop_x = get_f64_vec(&df, &format!("{}_cop_x", right_prefix), len);
    let r_cop_y = get_f64_vec(&df, &format!("{}_cop_y", right_prefix), len);
    let r_cop_z = get_f64_vec(&df, &format!("{}_cop_z", right_prefix), len);

    let tau_l = get_f64_vec(&df, "tauL", len);
    let vel_l = get_f64_vec(&df, "velaL", len);
    let tau_r = get_f64_vec(&df, "tauR", len);
    let vel_r = get_f64_vec(&df, "velaR", len);

    let vo2 = get_f64_vec(&df, "vo2", len);
    let vco2 = get_f64_vec(&df, "vco2", len);
    let qs_baseline = get_f64_vec(&df, "qs_baseline_w", len);

    let mut strides = Vec::new();
    let mut breaths = Vec::new();

    let initial_mass = 70.0; // Fallback
    let dt_default = 0.01;
    let mut analyzer = EnergyAnalyzer::new(initial_mass, 30.0, 0.254, Some(1.25));

    let mut last_l_strike_idx = None;
    let mut last_r_strike_idx = None;

    let mut p_exo_l_buf = Vec::new();
    let mut p_hum_l_buf = Vec::new();
    let mut p_exo_r_buf = Vec::new();
    let mut p_hum_r_buf = Vec::new();

    let mut last_invalid_breath_time = -f64::INFINITY;

    for i in 0..len {
        let t = times[i];
        let dt = if i < len - 1 {
            times[i + 1] - t
        } else {
            dt_default
        };

        let force_l = Vector3::new(l_fx[i], l_fy[i], l_fz[i]);
        let force_r = Vector3::new(r_fx[i], r_fy[i], r_fz[i]);
        let cop_l = Vector3::new(l_cop_x[i], l_cop_y[i], l_cop_z[i]);
        let cop_r = Vector3::new(r_cop_x[i], r_cop_y[i], r_cop_z[i]);

        let exo_l = tau_l[i] * vel_l[i];
        let exo_r = tau_r[i] * vel_r[i];

        let l_was_active = analyzer.stride_analyzer.left.is_active.unwrap_or(false);
        let r_was_active = analyzer.stride_analyzer.right.is_active.unwrap_or(false);

        let res = analyzer.update(t, force_l, force_r, cop_l, cop_r, dt);

        p_exo_l_buf.push(exo_l);
        p_hum_l_buf.push(res.power_left - exo_l);

        p_exo_r_buf.push(exo_r);
        p_hum_r_buf.push(res.power_right - exo_r);

        let l_is_active = analyzer.stride_analyzer.left.is_active.unwrap_or(false);
        let r_is_active = analyzer.stride_analyzer.right.is_active.unwrap_or(false);

        if l_is_active && !l_was_active {
            if let Some(start_idx) = last_l_strike_idx {
                let end_idx = i;
                let duration = times[end_idx] - times[start_idx];
                let is_valid = duration > 0.4 && duration < 1.5;

                let slice_len = end_idx - start_idx;
                let exo_slice = &p_exo_l_buf[p_exo_l_buf.len() - slice_len..];
                let hum_slice = &p_hum_l_buf[p_hum_l_buf.len() - slice_len..];

                let (mus_power, ach_power) = extract_biological_components(hum_slice, dt);
                let pos = |p: &[f64]| p.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>();
                let neg = |p: &[f64]| p.iter().map(|&x| x.min(0.0)).collect::<Vec<_>>();

                strides.push(StrideRecord {
                    trial_name: file_name.clone(),
                    leg: "Left".to_string(),
                    start_time: times[start_idx],
                    end_time: times[end_idx],
                    mid_time: (times[start_idx] + times[end_idx]) / 2.0,
                    duration,
                    mass_kg: res.mass,
                    is_valid,
                    work_exo_pos: trapz_1d(&pos(exo_slice), dt),
                    work_exo_neg: trapz_1d(&neg(exo_slice), dt),
                    work_mus_pos: trapz_1d(&pos(&mus_power), dt),
                    work_mus_neg: trapz_1d(&neg(&mus_power), dt),
                    work_ach_pos: trapz_1d(&pos(&ach_power), dt),
                    work_ach_neg: trapz_1d(&neg(&ach_power), dt),
                });
            }
            last_l_strike_idx = Some(i);
        }

        if r_is_active && !r_was_active {
            if let Some(start_idx) = last_r_strike_idx {
                let end_idx = i;
                let duration = times[end_idx] - times[start_idx];
                let is_valid = duration > 0.4 && duration < 1.5;

                let slice_len = end_idx - start_idx;
                let exo_slice = &p_exo_r_buf[p_exo_r_buf.len() - slice_len..];
                let hum_slice = &p_hum_r_buf[p_hum_r_buf.len() - slice_len..];

                let (mus_power, ach_power) = extract_biological_components(hum_slice, dt);
                let pos = |p: &[f64]| p.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>();
                let neg = |p: &[f64]| p.iter().map(|&x| x.min(0.0)).collect::<Vec<_>>();

                strides.push(StrideRecord {
                    trial_name: file_name.clone(),
                    leg: "Right".to_string(),
                    start_time: times[start_idx],
                    end_time: times[end_idx],
                    mid_time: (times[start_idx] + times[end_idx]) / 2.0,
                    duration,
                    mass_kg: res.mass,
                    is_valid,
                    work_exo_pos: trapz_1d(&pos(exo_slice), dt),
                    work_exo_neg: trapz_1d(&neg(exo_slice), dt),
                    work_mus_pos: trapz_1d(&pos(&mus_power), dt),
                    work_mus_neg: trapz_1d(&neg(&mus_power), dt),
                    work_ach_pos: trapz_1d(&pos(&ach_power), dt),
                    work_ach_neg: trapz_1d(&neg(&ach_power), dt),
                });
            }
            last_r_strike_idx = Some(i);
        }

        if i > 0 && vo2[i] > 0.0 && (vo2[i] - vo2[i - 1]).abs() > 1e-3 {
            let v = vo2[i];
            let c = vco2[i];
            let rer = if v > 0.0 { c / v } else { 0.0 };

            let gross_watts = ((3.941 * v) + (1.106 * c)) * 4.184 / 60.0;
            let net_watts = gross_watts - qs_baseline[i];

            let physio_valid = v > 200.0 && rer >= 0.72 && rer <= 1.05;

            if !physio_valid {
                last_invalid_breath_time = t;
            }

            let is_valid = physio_valid && (t - last_invalid_breath_time > 60.0);

            breaths.push(BreathRecord {
                trial_name: file_name.clone(),
                time: t,
                vo2: v,
                vco2: c,
                rer,
                gross_watts,
                net_watts,
                is_valid,
            });
        }
    }

    (strides, breaths)
}

fn main() {
    // 1. Force Polars to be strictly single-threaded internally.
    // This stops it from triggering Rayon's recursive work-stealing bug.
    std::env::set_var("POLARS_MAX_THREADS", "1");

    // 2. Build the global thread pool to strictly 6 threads.
    // This caps RAM usage so the OS doesn't thrash to the swap file.
    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .stack_size(8 * 1024 * 1024) // 8MB to be perfectly safe
        .build_global()
        .unwrap();

    println!("Finding parquet files...");
    let files: Vec<PathBuf> = glob("../exported_pogensee/*.parquet")
        .unwrap()
        .filter_map(Result::ok)
        .filter(|p| {
            let name = p.to_string_lossy();
            !name.contains("precomputed_poggensee")
                    && !name.contains("strides.parquet") // Fixed
                    && !name.contains("breaths.parquet") // Fixed
        })
        .collect();

    println!(
        "Processing {} trials using 6 parallel threads...",
        files.len()
    );

    // Map over the files in parallel using the limited thread pool
    let results: Vec<(Vec<StrideRecord>, Vec<BreathRecord>)> =
        files.par_iter().map(|f| process_trial(f)).collect();

    let mut all_strides = Vec::new();
    let mut all_breaths = Vec::new();

    for (s, b) in results {
        all_strides.extend(s);
        all_breaths.extend(b);
    }

    println!("Total Strides extracted: {}", all_strides.len());
    println!("Total Breaths extracted: {}", all_breaths.len());

    // --- Save Strides Parquet ---
    let mut df_strides = df!(
        "trial_name" => all_strides.iter().map(|s| s.trial_name.clone()).collect::<Vec<_>>(),
        "leg" => all_strides.iter().map(|s| s.leg.clone()).collect::<Vec<_>>(),
        "start_time" => all_strides.iter().map(|s| s.start_time).collect::<Vec<_>>(),
        "end_time" => all_strides.iter().map(|s| s.end_time).collect::<Vec<_>>(),
        "mid_time" => all_strides.iter().map(|s| s.mid_time).collect::<Vec<_>>(),
        "duration" => all_strides.iter().map(|s| s.duration).collect::<Vec<_>>(),
        "mass_kg" => all_strides.iter().map(|s| s.mass_kg).collect::<Vec<_>>(),
        "is_valid" => all_strides.iter().map(|s| s.is_valid).collect::<Vec<_>>(),
        "work_exo_pos" => all_strides.iter().map(|s| s.work_exo_pos).collect::<Vec<_>>(),
        "work_exo_neg" => all_strides.iter().map(|s| s.work_exo_neg).collect::<Vec<_>>(),
        "work_mus_pos" => all_strides.iter().map(|s| s.work_mus_pos).collect::<Vec<_>>(),
        "work_mus_neg" => all_strides.iter().map(|s| s.work_mus_neg).collect::<Vec<_>>(),
        "work_ach_pos" => all_strides.iter().map(|s| s.work_ach_pos).collect::<Vec<_>>(),
        "work_ach_neg" => all_strides.iter().map(|s| s.work_ach_neg).collect::<Vec<_>>(),
    )
    .unwrap();

    let strides_file = std::fs::File::create("../exported_pogensee/strides.parquet").unwrap();
    ParquetWriter::new(strides_file)
        .finish(&mut df_strides)
        .unwrap();
    println!("✅ Saved strides.parquet");

    // --- Save Breaths Parquet ---
    let mut df_breaths = df!(
        "trial_name" => all_breaths.iter().map(|b| b.trial_name.clone()).collect::<Vec<_>>(),
        "time" => all_breaths.iter().map(|b| b.time).collect::<Vec<_>>(),
        "vo2" => all_breaths.iter().map(|b| b.vo2).collect::<Vec<_>>(),
        "vco2" => all_breaths.iter().map(|b| b.vco2).collect::<Vec<_>>(),
        "rer" => all_breaths.iter().map(|b| b.rer).collect::<Vec<_>>(),
        "gross_watts" => all_breaths.iter().map(|b| b.gross_watts).collect::<Vec<_>>(),
        "net_watts" => all_breaths.iter().map(|b| b.net_watts).collect::<Vec<_>>(),
        "is_valid" => all_breaths.iter().map(|b| b.is_valid).collect::<Vec<_>>(),
    )
    .unwrap();

    let breaths_file = std::fs::File::create("../exported_pogensee/breaths.parquet").unwrap();
    ParquetWriter::new(breaths_file)
        .finish(&mut df_breaths)
        .unwrap();
    println!("✅ Saved breaths.parquet");
}
