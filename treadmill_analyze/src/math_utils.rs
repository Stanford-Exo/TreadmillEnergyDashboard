use std::f64;

/// Simple trapezoidal integration matching numpy.trapz
pub fn trapz_1d(y: &[f64], dx: f64) -> f64 {
    if y.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..(y.len() - 1) {
        sum += y[i] + y[i + 1];
    }
    sum * dx * 0.5
}

/// Fast zero-crossing detector
fn find_zero_crossings(y: &[f64]) -> Vec<usize> {
    let mut crossings = Vec::new();
    for i in 0..(y.len() - 1) {
        if (y[i + 1] > 0.0 && y[i] <= 0.0) || (y[i + 1] < 0.0 && y[i] >= 0.0) {
            crossings.push(i);
        }
    }
    crossings
}

/// Splits human power into 'Muscle Power' and 'Achilles Power' (balanced zero-net energy).
/// Achilles acts as a spring matching the negative loading lump with the positive push-off lump.
pub fn extract_biological_components(human_power: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
    let n = human_power.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    // Double the array to smoothly handle phase wrapping
    let mut doubled = Vec::with_capacity(2 * n);
    doubled.extend_from_slice(human_power);
    doubled.extend_from_slice(human_power);

    let mut achilles_doubled = vec![0.0; 2 * n];

    // Search area: Middle of the doubled array (N/2 to 3N/2) ensures space backwards and forwards
    let search_start = n / 2;
    let search_end = n + (n / 2);

    let mut peak_val = f64::NEG_INFINITY;
    let mut peak_idx = search_start;

    for i in search_start..search_end {
        if doubled[i] > peak_val {
            peak_val = doubled[i];
            peak_idx = i;
        }
    }

    // If the absolute peak is negative or zero, there's no push-off. Achilles = 0.
    if peak_val <= 0.0 {
        return (human_power.to_vec(), vec![0.0; n]);
    }

    let crossings = find_zero_crossings(&doubled);
    let mut boundaries = vec![0];
    boundaries.extend(crossings.iter().map(|&c| c + 1));
    boundaries.push(2 * n);

    let mut pos_start = None;
    let mut pos_end = None;
    let mut neg_start = None;
    let mut neg_end = None;

    for i in 0..(boundaries.len() - 1) {
        if boundaries[i] <= peak_idx && peak_idx < boundaries[i + 1] {
            pos_start = Some(boundaries[i]);
            pos_end = Some(boundaries[i + 1]);
            if i > 0 {
                neg_start = Some(boundaries[i - 1]);
                neg_end = Some(boundaries[i]);
            }
            break;
        }
    }

    if let (Some(ps), Some(pe), Some(ns), Some(ne)) = (pos_start, pos_end, neg_start, neg_end) {
        let pos_chunk = &doubled[ps..pe];
        let neg_chunk = &doubled[ns..ne];

        let e_pos = trapz_1d(pos_chunk, dt);
        let e_neg = trapz_1d(neg_chunk, dt);

        // e_neg must actually be negative for it to be elastic storage
        if e_pos > 0.0 && e_neg < 0.0 {
            let achilles_energy = e_pos.min(e_neg.abs());
            let scale_pos = if e_pos != 0.0 {
                achilles_energy / e_pos
            } else {
                0.0
            };
            let scale_neg = if e_neg != 0.0 {
                achilles_energy / e_neg.abs()
            } else {
                0.0
            };

            for i in ps..pe {
                achilles_doubled[i] = doubled[i] * scale_pos;
            }
            for i in ns..ne {
                achilles_doubled[i] = doubled[i] * scale_neg;
            }
        }
    }

    // Fold the doubled array back onto the 0-100% boundary
    let mut achilles_power = vec![0.0; n];
    let mut muscle_power = vec![0.0; n];
    for i in 0..n {
        achilles_power[i] = achilles_doubled[i] + achilles_doubled[n + i];
        muscle_power[i] = human_power[i] - achilles_power[i];
    }

    (muscle_power, achilles_power)
}
