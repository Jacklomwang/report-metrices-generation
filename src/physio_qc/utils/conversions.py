"""
Conversions for physiological signals
Handles unit conversions like volts to mmHg for gas analyzer channels
"""


def convert_voltage_to_mmhg_o2(voltage_signal):
    """
    Convert O2 voltage signal to mmHg

    Based on calibration: 1% O2 = 0.1V, 10% O2 = 1V
    (i.e. 20% O2 = 2V)

    Parameters
    ----------
    voltage_signal : array-like
        Raw voltage values from O2 sensor

    Returns
    -------
    array-like
        O2 values in mmHg
    """
    A_O2 = (((10 - 1) / 100) / (1 - 0.1)) * 760
    B_O2 = ((10 / 100) * 760) - (A_O2 * 1)

    return A_O2 * voltage_signal + B_O2


def convert_pct_to_mmhg_o2(pct_signal):
    """
    Convert O2 percent concentration signal to mmHg

    Parameters
    ----------
    pct_signal : array-like
        O2 values in percent concentration (e.g. 20 for 20%)

    Returns
    -------
    array-like
        O2 values in mmHg
    """
    return (pct_signal / 100) * 760


def convert_voltage_to_mmhg_co2(voltage_signal):
    """
    Convert CO2 voltage signal to mmHg

    Based on calibration: 1% CO2 = 1V, 5% CO2 = 5V

    Parameters
    ----------
    voltage_signal : array-like
        Raw voltage values from CO2 sensor

    Returns
    -------
    array-like
        CO2 values in mmHg
    """
    A_CO2 = (((5 - 1) / 100) / (5 - 1)) * 760
    B_CO2 = ((5 / 100) * 760) - (A_CO2 * 5)

    return A_CO2 * voltage_signal + B_CO2


def convert_pct_to_mmhg_co2(pct_signal):
    """
    Convert CO2 percent concentration signal to mmHg

    Parameters
    ----------
    pct_signal : array-like
        CO2 values in percent concentration (e.g. 5 for 5%)

    Returns
    -------
    array-like
        CO2 values in mmHg
    """
    return (pct_signal / 100) * 760


def convert_gas_channels(df, co2_channel=None, o2_channel=None):
    """
    Convert gas analyzer voltage channels to mmHg and add as new columns

    This function looks for CO2 and O2 channels by name, converts them from
    volts to mmHg, and adds new columns with standard names that will be
    detected by the signal pattern matching.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing physiological data channels
    co2_channel : str, optional
        Name of the CO2 voltage channel. If None, will attempt to auto-detect
        from channel names containing 'A 8' or 'A8'
    o2_channel : str, optional
        Name of the O2 voltage channel. If None, will attempt to auto-detect
        from channel names containing 'A 7' or 'A7'

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'CO2(mmHg)' and 'O2(mmHg)' columns if conversions applied
    dict
        Mapping of converted channels: {'co2': 'CO2(mmHg)', 'o2': 'O2(mmHg)'}
    """
    df = df.copy()
    conversions = {}

    # Auto-detect CO2 channel if not specified
    if co2_channel is None:
        for col in df.columns:
            col_lower = col.lower()
            if "a 8" in col_lower or "a8" in col_lower:
                co2_channel = col
                break

    # Auto-detect O2 channel if not specified
    if o2_channel is None:
        for col in df.columns:
            col_lower = col.lower()
            if "a 7" in col_lower or "a7" in col_lower:
                o2_channel = col
                break

    # Detect Pct vs Volts for O2 from values: gas analyzer max is 10V,
    # so any O2 value > 10 means the channel is in percent concentration.
    # CO2 is 1:1 (Volts == Pct) so the conversion is identical either way.
    o2_is_pct = False
    if o2_channel and o2_channel in df.columns:
        if df[o2_channel].max() > 10:
            o2_is_pct = True

    # Convert CO2 if channel found
    if co2_channel and co2_channel in df.columns:
        df["CO2(mmHg)"] = convert_voltage_to_mmhg_co2(df[co2_channel])
        conversions["co2"] = "CO2(mmHg)"

    # Convert O2 if channel found (Pct vs Volts based on value range)
    if o2_channel and o2_channel in df.columns:
        if o2_is_pct:
            df["O2(mmHg)"] = convert_pct_to_mmhg_o2(df[o2_channel])
        else:
            df["O2(mmHg)"] = convert_voltage_to_mmhg_o2(df[o2_channel])
        conversions["o2"] = "O2(mmHg)"

    return df, conversions


def convert_doppler_channel(df, doppler_channel=None):
    """
    Detect Doppler channel (e.g. Biopac A6) and create a standardized Doppler column

    This function looks for a Doppler channel by name (default: 'A 6' or 'A6'),
    and copies it into a new column called 'Doppler', which will then be
    detected by SIGNAL_PATTERNS.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing physiological data channels
    doppler_channel : str, optional
        Name of the Doppler channel. If None, will attempt to auto-detect
        from channel names containing 'A 6' or 'A6'

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'Doppler' column if detected
    dict
        Mapping of converted channels: {'doppler': 'Doppler'}
    """
    df = df.copy()
    conversions = {}

    # Auto-detect Doppler channel if not specified
    if doppler_channel is None:
        for col in df.columns:
            col_lower = col.lower()
            if 'a 6' in col_lower or 'a6' in col_lower:
                doppler_channel = col
                break

    # Create standardized Doppler column
    if doppler_channel and doppler_channel in df.columns:
        df['Doppler'] = df[doppler_channel]
        conversions['doppler'] = 'Doppler'

    return df, conversions
