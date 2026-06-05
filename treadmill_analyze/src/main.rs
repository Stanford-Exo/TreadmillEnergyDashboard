use glob::glob;
use nalgebra::Vector3;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

use treadmill_analyze::energy_analyzer::EnergyAnalyzer;
use treadmill_analyze::math_utils::{extract_biological_components, trapz_1d};

#[derive(Clone, Debug)]
struct GaitCycleRecord {
    trial_name: String,
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

// Safely pull numeric arrays from the Polars DataFrame
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

/// Helper to parse the base Subject/Day group key from a filename.
/// E.g., "PDS01_Day1_GA1" -> "PDS01_Day1"
fn get_group_key(filename: &str) -> String {
    if let Some(idx) = filename.rfind('_') {
        filename[..idx].to_string()
    } else {
        filename.to_string()
    }
}

/// Processes a QS trial to find the steady-state baseline metabolic cost (last 3 minutes)
fn compute_qs_baseline(file_path: &PathBuf) -> Option<f64> {
    let file = std::fs::File::open(file_path).ok()?;
    let df = ParquetReader::new(file).finish().ok()?;

    let times = get_f64_vec(&df, "time", df.height());
    let vo2 = get_f64_vec(&df, "vo2", df.height());
    let vco2 = get_f64_vec(&df, "vco2", df.height());

    let mut watts = Vec::new();
    let mut valid_times = Vec::new();

    for i in 0..times.len() {
        if vo2[i] > 0.0 && vco2[i] > 0.0 {
            let w = ((3.941 * vo2[i]) + (1.106 * vco2[i])) * 4.184 / 60.0;
            watts.push(w);
            valid_times.push(times[i]);
        }
    }

    if watts.is_empty() {
        return None;
    }

    let t_end = *valid_times.last().unwrap();
    let mut ss_watts = Vec::new();

    // Extract the final 3 minutes (180 seconds)
    for i in 0..watts.len() {
        if valid_times[i] >= t_end - 180.0 {
            ss_watts.push(watts[i]);
        }
    }

    // Fallback if the trial is shorter than 3 minutes
    if ss_watts.is_empty() {
        ss_watts = watts;
    }

    // Calculate the median value
    ss_watts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(ss_watts[ss_watts.len() / 2])
}

fn process_trial(
    file_path: &PathBuf,
    baseline_w: Option<f64>,
) -> (Vec<GaitCycleRecord>, Vec<BreathRecord>) {
    let file_name = file_path.file_stem().unwrap().to_string_lossy().to_string();
    println!("Processing: {}", file_name);

    let file = std::fs::File::open(file_path).unwrap();
    let df = ParquetReader::new(file).finish().unwrap();

    let times = get_f64_vec(&df, "time", df.height());
    let len = times.len();
    if len == 0 {
        return (vec![], vec![]);
    }

    // Dynamic Contact Body Mapping
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

    // Resolve which baseline to use (Passed in from QS, or fallback to file's own qs_baseline_w, or 70W)
    let file_qs_col = get_f64_vec(&df, "qs_baseline_w", len);
    let internal_fallback = if !file_qs_col.is_empty() && file_qs_col[0] > 0.0 {
        file_qs_col[0]
    } else {
        70.0
    };
    let resolved_baseline_w = baseline_w.unwrap_or(internal_fallback);

    let mut cycles = Vec::new();
    let mut breaths = Vec::new();

    let initial_mass = 70.0;
    let dt_default = 0.01;
    let mut analyzer = EnergyAnalyzer::new(initial_mass, 30.0, 0.254, Some(1.25));

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

        let r_was_active = analyzer.stride_analyzer.right.is_active.unwrap_or(false);
        let res = analyzer.update(t, force_l, force_r, cop_l, cop_r, dt);

        p_exo_l_buf.push(exo_l);
        p_hum_l_buf.push(res.power_left - exo_l);

        p_exo_r_buf.push(exo_r);
        p_hum_r_buf.push(res.power_right - exo_r);

        let r_is_active = analyzer.stride_analyzer.right.is_active.unwrap_or(false);

        // --- Extract Bilateral Gait Cycle (Right Heel Strike to Right Heel Strike) ---
        if r_is_active && !r_was_active {
            if let Some(start_idx) = last_r_strike_idx {
                let end_idx = i;
                let duration = times[end_idx] - times[start_idx];

                // Normal bilateral gait cycle bounds (~0.8s to 1.8s)
                let is_valid = duration > 0.6 && duration < 2.0;
                let slice_len = end_idx - start_idx;

                // Safety guard ensuring buffer is deeply populated
                if slice_len <= p_exo_l_buf.len() && slice_len <= p_exo_r_buf.len() {
                    let exo_l_slice = &p_exo_l_buf[p_exo_l_buf.len() - slice_len..];
                    let exo_r_slice = &p_exo_r_buf[p_exo_r_buf.len() - slice_len..];
                    let hum_l_slice = &p_hum_l_buf[p_hum_l_buf.len() - slice_len..];
                    let hum_r_slice = &p_hum_r_buf[p_hum_r_buf.len() - slice_len..];

                    // Process biologically active & passive costs per leg
                    let (mus_l, ach_l) = extract_biological_components(hum_l_slice, dt);
                    let (mus_r, ach_r) = extract_biological_components(hum_r_slice, dt);

                    let pos = |p: &[f64]| p.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>();
                    let neg = |p: &[f64]| p.iter().map(|&x| x.min(0.0)).collect::<Vec<_>>();

                    // Integrate combined total work for the entire gait cycle
                    cycles.push(GaitCycleRecord {
                        trial_name: file_name.clone(),
                        start_time: times[start_idx],
                        end_time: times[end_idx],
                        mid_time: (times[start_idx] + times[end_idx]) / 2.0,
                        duration,
                        mass_kg: res.mass,
                        is_valid,
                        work_exo_pos: trapz_1d(&pos(exo_l_slice), dt)
                            + trapz_1d(&pos(exo_r_slice), dt),
                        work_exo_neg: trapz_1d(&neg(exo_l_slice), dt)
                            + trapz_1d(&neg(exo_r_slice), dt),
                        work_mus_pos: trapz_1d(&pos(&mus_l), dt) + trapz_1d(&pos(&mus_r), dt),
                        work_mus_neg: trapz_1d(&neg(&mus_l), dt) + trapz_1d(&neg(&mus_r), dt),
                        work_ach_pos: trapz_1d(&pos(&ach_l), dt) + trapz_1d(&pos(&ach_r), dt),
                        work_ach_neg: trapz_1d(&neg(&ach_l), dt) + trapz_1d(&neg(&ach_r), dt),
                    });
                }
            }
            last_r_strike_idx = Some(i);
        }

        // --- Process Breaths (Only when vo2 values physically update) ---
        if i > 0 && vo2[i] > 0.0 && (vo2[i] - vo2[i - 1]).abs() > 1e-3 {
            let v = vo2[i];
            let c = vco2[i];
            let rer = if v > 0.0 { c / v } else { 0.0 };

            let gross_watts = ((3.941 * v) + (1.106 * c)) * 4.184 / 60.0;
            let net_watts = gross_watts - resolved_baseline_w;

            let physio_valid = v > 200.0 && rer >= 0.72 && rer <= 1.05;

            if !physio_valid {
                last_invalid_breath_time = t;
            }

            // Washout filter (hangover of 60 seconds after a break)
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

    (cycles, breaths)
}

fn main() {
    std::env::set_var("POLARS_MAX_THREADS", "1");

    rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .stack_size(8 * 1024 * 1024)
        .build_global()
        .unwrap();

    println!("Finding parquet files...");
    let files: Vec<PathBuf> = glob("../exported_pogensee/*.parquet")
        .unwrap()
        .filter_map(Result::ok)
        .filter(|p| {
            let name = p.to_string_lossy();
            !name.contains("precomputed_poggensee")
                && !name.contains("gait_cycles.parquet")
                && !name.contains("breaths.parquet")
        })
        .collect();

    // 1. Pass: Extract QS Baselines mappings
    println!("Computing Quiet Standing (QS) baselines...");
    let mut qs_baselines: HashMap<String, f64> = HashMap::new();

    for f in &files {
        let name = f.file_stem().unwrap().to_string_lossy();
        if name.ends_with("_QS") && !name.contains("adaptation") {
            if let Some(baseline_watts) = compute_qs_baseline(f) {
                let key = get_group_key(&name);
                qs_baselines.insert(key.clone(), baseline_watts);
                println!("  -> Extracted {} baseline: {:.1} W", key, baseline_watts);
            }
        }
    }

    // 2. Pass: Process all trials (Parallel)
    println!(
        "\nProcessing {} trials using 6 parallel threads...",
        files.len()
    );
    let results: Vec<(Vec<GaitCycleRecord>, Vec<BreathRecord>)> = files
        .par_iter()
        .map(|f| {
            let name = f.file_stem().unwrap().to_string_lossy();
            let key = get_group_key(&name);
            let baseline = qs_baselines.get(&key).copied();
            process_trial(f, baseline)
        })
        .collect();

    let mut all_cycles = Vec::new();
    let mut all_breaths = Vec::new();

    for (c, b) in results {
        all_cycles.extend(c);
        all_breaths.extend(b);
    }

    println!("\nTotal Gait Cycles extracted: {}", all_cycles.len());
    println!("Total Breaths extracted: {}", all_breaths.len());

    // --- Safely Save Gait Cycles Parquet (Prevents PyArrow crash on empty lists) ---
    if !all_cycles.is_empty() {
        let mut df_cycles = df!(
            "trial_name" => all_cycles.iter().map(|c| c.trial_name.clone()).collect::<Vec<_>>(),
            "start_time" => all_cycles.iter().map(|c| c.start_time).collect::<Vec<_>>(),
            "end_time" => all_cycles.iter().map(|c| c.end_time).collect::<Vec<_>>(),
            "mid_time" => all_cycles.iter().map(|c| c.mid_time).collect::<Vec<_>>(),
            "duration" => all_cycles.iter().map(|c| c.duration).collect::<Vec<_>>(),
            "mass_kg" => all_cycles.iter().map(|c| c.mass_kg).collect::<Vec<_>>(),
            "is_valid" => all_cycles.iter().map(|c| c.is_valid).collect::<Vec<_>>(),
            "work_exo_pos" => all_cycles.iter().map(|c| c.work_exo_pos).collect::<Vec<_>>(),
            "work_exo_neg" => all_cycles.iter().map(|c| c.work_exo_neg).collect::<Vec<_>>(),
            "work_mus_pos" => all_cycles.iter().map(|c| c.work_mus_pos).collect::<Vec<_>>(),
            "work_mus_neg" => all_cycles.iter().map(|c| c.work_mus_neg).collect::<Vec<_>>(),
            "work_ach_pos" => all_cycles.iter().map(|c| c.work_ach_pos).collect::<Vec<_>>(),
            "work_ach_neg" => all_cycles.iter().map(|c| c.work_ach_neg).collect::<Vec<_>>(),
        )
        .unwrap();

        let cycles_file =
            std::fs::File::create("../exported_pogensee/gait_cycles.parquet").unwrap();
        ParquetWriter::new(cycles_file)
            .finish(&mut df_cycles)
            .unwrap();
        println!("✅ Saved gait_cycles.parquet");
    } else {
        println!("⚠️  Skipped saving gait_cycles.parquet (0 rows extracted)");
    }

    // --- Safely Save Breaths Parquet ---
    if !all_breaths.is_empty() {
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
    } else {
        println!("⚠️  Skipped saving breaths.parquet (0 rows extracted)");
    }
}
