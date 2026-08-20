"""
Spirometry signal processing functions using BreathMetrics
"""

import traceback
import numpy as np
from scipy.interpolate import interp1d


def _get_flow_values(bm, flow_attr, indices):
    """Return pre-computed flow values if available, else sample the signal at indices."""
    flows = getattr(bm, flow_attr, None)
    if flows is not None and len(flows) > 0:
        return flows
    if len(indices) > 0:
        return bm.smoothedRespiration[indices]
    return np.array([])


def _build_outlier_mask(length, *components):
    """OR together boolean mask components into a mask of the given length.

    None entries are skipped. Components shorter than `length` are treated as
    False beyond their end (length mismatches are common because rate arrays
    have one entry per breath interval while volume arrays have one per cycle).
    """
    combined = np.zeros(length, dtype=bool)
    for comp in components:
        if comp is None:
            continue
        n = min(length, len(comp))
        combined[:n] |= np.asarray(comp[:n], dtype=bool)
    return combined


def _blank_invalid(array, invalid_mask):
    """Return a float copy of `array` with `invalid_mask` positions replaced by NaN."""
    clean = np.asarray(array, dtype=float).copy()
    clean[invalid_mask] = np.nan
    return clean


def _sd_outliers(values, n_sd):
    """Return (low_mask, high_mask) for values beyond n_sd SD from the mean.

    NaN/Inf are treated as non-outliers. Returns all-False masks if <2 finite values.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.zeros(len(values), dtype=bool), np.zeros(len(values), dtype=bool)
    mean = np.nanmean(values[valid])
    std  = np.nanstd(values[valid])
    return values < mean - n_sd * std, values > mean + n_sd * std


def _add_volume_stats(bm_attr, key_prefix, result, bm, exclude_outliers, volume_outlier_sd=3.0, flow_attr=None):
    """
    Extract volume stats from a BreathMetrics attribute and store in result.

    BreathMetrics stores volume arrays as shape (1, n) — we index [0] to get
    the flat 1-D array of per-cycle values.

    If flow_attr is provided, peak flow outliers are OR'd into the volume outlier mask.
    Absolute values are used for flow comparison since expiratory troughs are negative.
    """
    vols = getattr(bm, bm_attr, None)
    if vols is None or len(vols) == 0:
        return

    cycle_vols = vols[0]  # BreathMetrics stores volumes as (1, n); grab the inner array
    result[f'{key_prefix}_volumes'] = cycle_vols.flatten()

    valid = np.isfinite(cycle_vols)
    if not valid.any():
        return

    if not exclude_outliers:
        result[f'{key_prefix}_outliers'] = np.zeros(len(cycle_vols), dtype=bool)
        result[f'mean_{key_prefix}_volume'] = float(np.mean(cycle_vols[valid]))
        return

    # Volume-based outliers
    vol_low, vol_high = _sd_outliers(cycle_vols, volume_outlier_sd)

    # Flow-based outliers, aligned to cycle_vols length (pad with NaN beyond known flows)
    flow_low  = np.zeros(len(cycle_vols), dtype=bool)
    flow_high = np.zeros(len(cycle_vols), dtype=bool)
    if flow_attr is not None and hasattr(bm, flow_attr):
        flows_raw = np.abs(np.array(getattr(bm, flow_attr), dtype=float))
        flows_aligned = np.full(len(cycle_vols), np.nan)
        n = min(len(flows_raw), len(cycle_vols))
        flows_aligned[:n] = flows_raw[:n]
        flow_low, flow_high = _sd_outliers(flows_aligned, volume_outlier_sd)

    outlier_mask_low  = vol_low  | flow_low
    outlier_mask_high = vol_high | flow_high
    outlier_mask      = outlier_mask_low | outlier_mask_high

    result[f'{key_prefix}_outliers']      = outlier_mask
    result[f'{key_prefix}_outliers_low']  = outlier_mask_low
    result[f'{key_prefix}_outliers_high'] = outlier_mask_high

    non_outlier = valid & ~outlier_mask
    result[f'mean_{key_prefix}_volume'] = float(
        np.mean(cycle_vols[non_outlier]) if non_outlier.any() else np.mean(cycle_vols[valid])
    )


def process_breathmetrics(signal, sampling_rate, params):
    """
    Wrapper function to integrate breathMetricsClass with Streamlit.

    Parameters
    ----------
    signal : array
        Raw respiratory signal
    sampling_rate : int
        Sampling rate in Hz
    params : dict
        Processing parameters: verbose, zscore, baseline_method, simplify,
        exclude_outliers, exclude_duration_outliers

    Returns
    -------
    dict
        Results in format expected by Streamlit app
    """
    from .breathmetricsClass import bmObject

    # Extract parameters
    data_type = params.get('data_type', 'humanAirflow')
    zscore = params.get('zscore', 0)
    baseline_method = params.get('baseline_method', 'sliding')
    simplify = params.get('simplify', 1)
    verbose = params.get('verbose', 0)
    exclude_outliers = params.get('exclude_outliers', 0)
    volume_outlier_sd = float(params.get('volume_outlier_sd', 2.0))
    exclude_duration_outliers = params.get('exclude_duration_outliers', 0)
    duration_outlier_sd = float(params.get('duration_outlier_sd', 2.0))

    try:
        if signal is None or len(signal) == 0:
            raise ValueError("Signal is empty or None")

        if not isinstance(sampling_rate, (int, float)) or sampling_rate < 20 or sampling_rate > 5000:
            raise ValueError(f"Invalid sampling rate: {sampling_rate}. Must be between 20-5000 Hz")

        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 1:
            signal = signal.flatten()

        if np.any(np.isnan(signal)):
            print(f"WARNING: Signal contains {np.sum(np.isnan(signal))} NaN values, filling with 0")
            signal = np.nan_to_num(signal, nan=0.0)

        if np.any(np.isinf(signal)):
            print(f"WARNING: Signal contains {np.sum(np.isinf(signal))} infinite values, replacing with 0")
            signal = np.where(np.isinf(signal), 0.0, signal)

        valid_data_types = ['humanAirflow', 'rodentAirflow', 'humanBB', 'rodentThermocouple']
        if data_type not in valid_data_types:
            raise ValueError(f"Invalid data type '{data_type}'. Must be one of: {valid_data_types}")

        bm = bmObject(signal, sampling_rate, data_type)
        bm.estimateAllFeatures(
            zScore=zscore,
            baselineCorrectionMethod=baseline_method,
            simplify=simplify,
            verbose=verbose
        )

        # For airflow data, peaks are inhale peaks, troughs are exhale troughs.
        # For belt data, peaks are exhale onsets, troughs are inhale onsets.
        if data_type in ['humanAirflow', 'rodentAirflow']:
            current_peaks = bm.inhalePeaks
            current_troughs = bm.exhaleTroughs
            peaks_values   = _get_flow_values(bm, 'peakInspiratoryFlows',  bm.inhalePeaks)
            troughs_values = _get_flow_values(bm, 'troughExpiratoryFlows', bm.exhaleTroughs)
        else:  # In case we want to try using breathmetrics on belt data, though this isn't supported in the app
            current_peaks = bm.exhaleOnsets if hasattr(bm, 'exhaleOnsets') else bm.inhaleOnsets
            current_troughs = bm.inhaleOnsets if hasattr(bm, 'inhaleOnsets') else bm.exhaleOnsets
            peaks_values = bm.smoothedRespiration[current_peaks] if len(current_peaks) > 0 else np.array([])
            troughs_values = bm.smoothedRespiration[current_troughs] if len(current_troughs) > 0 else np.array([])

        current_peaks = np.array(current_peaks) if len(current_peaks) > 0 else np.array([])
        current_troughs = np.array(current_troughs) if len(current_troughs) > 0 else np.array([])

        result = {
            'raw': signal,
            'clean': bm.smoothedRespiration,
            'baseline_corrected': bm.baselineCorrectedRespiration,
            'time': bm.time,
            'current_peaks': current_peaks, # Can be used for peak editing
            'current_troughs': current_troughs, # Can be used for peak editing
            'auto_peaks': current_peaks.copy(),
            'auto_troughs': current_troughs.copy(),
            'peaks_values': peaks_values,
            'troughs_values': troughs_values,
            'data_type': data_type,
            'params': params.copy(),
        }

        _add_volume_stats('inhaleVolumes', 'inhale', result, bm, exclude_outliers, volume_outlier_sd,
                          flow_attr='peakInspiratoryFlows')
        _add_volume_stats('exhaleVolumes', 'exhale', result, bm, exclude_outliers, volume_outlier_sd,
                          flow_attr='troughExpiratoryFlows')

        # Tidal volumes (inhale + exhale per cycle, matching BreathMetrics' convention).
        # Each constituent volume is computed in BreathMetrics by integrating
        # abs(baseline-corrected signal) over the inhale or exhale phase, then
        # dividing by the sampling rate.
        #
        # Note: mean_inhale_volume and mean_exhale_volume are set by _add_volume_stats
        # using independent per-side denominators, so mean_tidal_volume below is *not*
        # algebraically equal to mean_inhale + mean_exhale. Callers that need the three
        # means to be internally consistent (e.g. so manual cycle removal can update
        # them in lock-step) should recompute all three from the per-cycle arrays using
        # a single shared mask.
        if 'inhale_volumes' in result and 'exhale_volumes' in result:
            inhale_vols = np.array(result['inhale_volumes']).flatten()
            exhale_vols = np.array(result['exhale_volumes']).flatten()
            min_len = min(len(inhale_vols), len(exhale_vols))
            # Tidal volume per cycle = average of inhale and exhale volumes (≈ one-way air
            # displaced per breath). BreathMetrics' convention is the *sum*, which is 2× the
            # textbook V_T and inflates downstream minute ventilation by the same factor;
            # we deliberately diverge here.
            tidal_volumes = (inhale_vols[:min_len] + exhale_vols[:min_len]) / 2.0
            result['tidal_volumes'] = tidal_volumes

            # Merge inhale/exhale outlier masks into tidal-level masks
            tidal_outliers      = _build_outlier_mask(min_len, result.get('inhale_outliers'),      result.get('exhale_outliers'))
            tidal_outliers_low  = _build_outlier_mask(min_len, result.get('inhale_outliers_low'),  result.get('exhale_outliers_low'))
            tidal_outliers_high = _build_outlier_mask(min_len, result.get('inhale_outliers_high'), result.get('exhale_outliers_high'))
            result['tidal_outliers']      = tidal_outliers
            result['tidal_outliers_low']  = tidal_outliers_low
            result['tidal_outliers_high'] = tidal_outliers_high

            valid_tidal = np.isfinite(tidal_volumes) & ~tidal_outliers
            if valid_tidal.any():
                result['mean_tidal_volume'] = float(np.mean(tidal_volumes[valid_tidal]))

        for bm_attr, key_prefix in [('inhaleDurations', 'inhale'), ('exhaleDurations', 'exhale')]:
            durations = getattr(bm, bm_attr, None)
            if durations is None or len(durations) == 0:
                continue
            result[f'{key_prefix}_durations'] = np.array(durations).flatten()
            valid = durations[0][~np.isnan(durations[0])]
            if len(valid) > 0:
                result[f'mean_{key_prefix}_duration'] = float(np.mean(valid))

        # Breathing rate — prefer inhale onsets for flow signals, fall back to troughs
        if hasattr(bm, 'inhaleOnsets') and bm.inhaleOnsets is not None and len(bm.inhaleOnsets) > 1:
            rate_markers = np.array(bm.inhaleOnsets)
        elif len(current_troughs) > 1:
            rate_markers = current_troughs
            print("WARNING: Using troughs to estimate breathing rate.")
        else:
            rate_markers = None

        if rate_markers is not None:
            marker_times = rate_markers / sampling_rate
            intervals = np.diff(marker_times)
            if len(intervals) > 0:
                # Compute duration outlier mask first so it can gate the mean interval
                dur_low, dur_high = _sd_outliers(intervals, duration_outlier_sd)
                duration_outlier_mask = dur_low | dur_high
                result['duration_outliers'] = duration_outlier_mask

                valid_intervals = intervals[~duration_outlier_mask] if exclude_duration_outliers else intervals
                mean_interval = np.mean(valid_intervals) if len(valid_intervals) > 0 else np.mean(intervals)
                result['breathing_rate_bpm'] = 60.0 / mean_interval
                result['mean_breath_interval'] = mean_interval

                breath_times = marker_times[:-1] + intervals / 2
                rates = 60.0 / intervals
                result['breath_times'] = breath_times
                result['rate_values'] = rates

                # Build outlier mask for rate, honoring both volume and duration exclusions
                # so the breathing-rate plot stays consistent with tidal_volumes_clean and
                # minute_ventilation_values_clean below.
                rate_outlier_mask = _build_outlier_mask(
                    len(rates),
                    result.get('tidal_outliers')    if exclude_outliers          else None,
                    result.get('duration_outliers') if exclude_duration_outliers else None,
                )
                result['rate_values_clean'] = _blank_invalid(rates, rate_outlier_mask)
                result['rate_interpolated'] = interp1d(
                    breath_times, rates, kind='linear', bounds_error=False, fill_value=np.nan
                )(bm.time)

                if 'tidal_volumes' in result:
                    min_len = min(len(result['tidal_volumes']), len(rates))
                    # Minute ventilation in L/min: tidal_volume (mL) * rate (per min) / 1000
                    minute_ventilation_values = result['tidal_volumes'][:min_len] * rates[:min_len] / 1000.0
                    result['minute_ventilation_values'] = minute_ventilation_values
                    result['minute_ventilation_interpolated'] = interp1d(
                        breath_times[:min_len], minute_ventilation_values,
                        kind='linear', bounds_error=False, fill_value=np.nan
                    )(bm.time)

                    # Combined invalid-cycle mask: non-finite MV OR any active outlier source
                    invalid_cycles = _build_outlier_mask(
                        min_len,
                        result.get('tidal_outliers')    if exclude_outliers          else None,
                        result.get('duration_outliers') if exclude_duration_outliers else None,
                    )
                    valid_mv_mask = np.isfinite(minute_ventilation_values) & ~invalid_cycles

                    # Clean per-cycle arrays with NaN in place of invalid cycles
                    result['tidal_volumes_clean']             = _blank_invalid(result['tidal_volumes'][:min_len], ~valid_mv_mask)
                    result['minute_ventilation_values_clean'] = _blank_invalid(minute_ventilation_values,         ~valid_mv_mask)

                    if valid_mv_mask.any():
                        result['mean_minute_ventilation'] = float(np.mean(minute_ventilation_values[valid_mv_mask]))

        # Precomputed "currently excluded" masks honoring exclude_* flags.
        # Frozen at process-time — toggling the exclude_* checkboxes without re-running
        # will not update these. Constituents (inhale_outliers, duration_outliers, etc.)
        # are still available for callers that need the raw per-source breakdown.
        dur_component = result.get('duration_outliers') if exclude_duration_outliers else None
        for key_prefix in ('inhale', 'exhale', 'tidal'):
            outliers = result.get(f'{key_prefix}_outliers')
            if outliers is None:
                continue
            vol_component = outliers if exclude_outliers else None
            result[f'{key_prefix}_excluded'] = _build_outlier_mask(len(outliers), vol_component, dur_component)

        # Scalar outlier counts — convenience for scripts that summarise across subjects.
        if 'tidal_outliers' in result:
            result['n_breaths'] = int(len(result['tidal_outliers']))
        for key in ('inhale_outliers', 'exhale_outliers', 'tidal_outliers',
                    'tidal_outliers_low', 'tidal_outliers_high', 'duration_outliers'):
            if key in result:
                result[f'n_{key}'] = int(np.sum(result[key]))

        return result

    except Exception as e:
        print(f"breathMetrics processing failed: {e}")
        traceback.print_exc()
        return None
