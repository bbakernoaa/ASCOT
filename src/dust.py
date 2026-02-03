#!/usr/bin/env python
"""
Dust Detection Algorithm using Hourly EPA AQS/AIRNOW Surface Monitors.

Architected by Aero 🍃⚡
"""

import numpy as np
import pandas as pd
import xarray as xr


def fill_gaps(da: xr.DataArray, dim: str = "time", limit: int = 1) -> xr.DataArray:
    """
    Fill 1-hour gaps in a boolean mask.

    Parameters
    ----------
    da : xr.DataArray
        Input boolean mask.
    dim : str, optional
        Dimension along which to fill gaps, by default "time".
    limit : int, optional
        Maximum gap size to fill, by default 1.
        (Currently only limit=1 is implemented via shift).

    Returns
    -------
    xr.DataArray
        Boolean mask with filled gaps.
    """
    if limit != 1:
        raise NotImplementedError(
            "Only limit=1 is currently implemented for fill_gaps."
        )

    return da | (
        da.shift({dim: 1}, fill_value=False) & da.shift({dim: -1}, fill_value=False)
    )


def start_end_duration(
    ds: xr.Dataset, column: str = "DUST", time_dim: str = "time"
) -> xr.Dataset:
    """
    Calculate the start date and duration of dust events.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with dust detection results.
    column : str, optional
        The column to use for event detection, by default "DUST".
    time_dim : str, optional
        The time dimension name, by default "time".

    Returns
    -------
    xr.Dataset
        Dataset with 'START_DATE' and 'DURATION' added.
    """
    is_event = ds[column]
    # Identify the start of each event
    starts = is_event & ~is_event.shift({time_dim: 1}, fill_value=False)
    # Identify the end of each event
    ends = is_event & ~is_event.shift({time_dim: -1}, fill_value=False)

    # Cumulative count of True values
    # We need to reset it for each event.
    # Using the lazy trick: cumsum - ffill(cumsum at starts)
    cum = is_event.astype(int).cumsum(dim=time_dim)
    start_cum = cum.where(starts).ffill(dim=time_dim)
    duration_at_hour = cum - start_cum + 1

    # Total duration is the duration at the end of the event, backfilled
    total_duration = duration_at_hour.where(ends).bfill(dim=time_dim)
    total_duration = total_duration.where(is_event)

    # Start date is the time at the start, forward filled
    event_start_time = ds[time_dim].where(starts).ffill(dim=time_dim)
    event_start_time = event_start_time.where(is_event)

    ds["START_DATE"] = event_start_time
    ds["DURATION"] = total_duration

    return ds


def dust_algorithm(
    ds: xr.Dataset,
    lower_threshold: float = 100.0,
    upper_threshold: float = 140.0,
) -> xr.Dataset:
    """
    Apply the Dust Detection Algorithm to surface monitor data.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing PM10, PM25, and WS (Wind Speed).
        Expected dimensions: (site, time) or (time, site).
    lower_threshold : float, optional
        Lower threshold for PM10, by default 100.0.
    upper_threshold : float, optional
        Upper threshold for PM10, by default 140.0.

    Returns
    -------
    xr.Dataset
        Dataset with dust detection flags and quality indicators.
    """
    # Ensure we have the required variables
    for var in ["PM10", "PM25", "WS"]:
        if var not in ds.data_vars:
            # Try to find case-insensitive match or common variations
            found = False
            for v in ds.data_vars:
                if v.upper() == var or (var == "PM25" and v == "PM2.5"):
                    ds = ds.rename({v: var})
                    found = True
                    break
            if not found and var == "PM25":
                # If PM25 is missing, fill with NaN to allow algorithm to run
                # (with lower confidence)
                ds["PM25"] = xr.full_like(ds["PM10"], np.nan)

    # 1. Scientific Logic: Ratios and integrity
    ds["RATIO"] = ds.PM25 / ds.PM10

    # 2. Rolling Statistics (3-hour window)
    # Aero Protocol: Use xarray's rolling which supports dask
    pm10_rmean = ds.PM10.rolling(time=3, center=True).mean()
    pm10_rmax = ds.PM10.rolling(time=3, center=True).max()
    ws_rmax = ds.WS.rolling(time=3, center=True).max()

    # Fill gaps caused by rolling window (bfill/ffill)
    pm10_rmean = pm10_rmean.bfill(dim="time").ffill(dim="time")
    pm10_rmax = pm10_rmax.bfill(dim="time").ffill(dim="time")
    ws_rmax = ws_rmax.bfill(dim="time").ffill(dim="time")

    # 3. Setting Thresholds
    pm10_lower_thr = lower_threshold
    pm10_upper_thr = 180.0
    pm10_98_thr = 85.0

    # 4. Dust Levels (G-series and T-series)
    g1 = (pm10_rmean > pm10_lower_thr) & (pm10_rmax >= pm10_upper_thr)
    g2 = (pm10_rmean > pm10_lower_thr) & (pm10_rmax >= upper_threshold)
    g3 = (pm10_rmean > pm10_98_thr) & (pm10_rmax >= upper_threshold)

    t1 = g2 & (ds.RATIO <= 0.35)
    t2 = g2 & (ds.RATIO <= 0.26)
    t3 = g2 & (ds.RATIO <= 0.20)

    # Wind Speed refined levels
    ws_threshold = 7.3
    g2_ws = g2 & (ws_rmax > ws_threshold)
    t2_ws = t2 & (ws_rmax > ws_threshold)
    g3_ws = g3 & (ws_rmax > ws_threshold)
    t3_ws = t3 & (ws_rmax > ws_threshold)

    # 5. Fill 1-hr gaps
    g1 = fill_gaps(g1)
    g2 = fill_gaps(g2)
    g3 = fill_gaps(g3)
    t1 = fill_gaps(t1)
    t2 = fill_gaps(t2)
    t3 = fill_gaps(t3)
    g2_ws = fill_gaps(g2_ws)
    t2_ws = fill_gaps(t2_ws)
    g3_ws = fill_gaps(g3_ws)
    t3_ws = fill_gaps(t3_ws)

    # 6. Final Dust Product
    dust = t2 | g2 | g2_ws | t2_ws

    # 7. Method classification
    method = xr.full_like(dust, "NONE", dtype="U10")
    method = xr.where(dust & g2 & t2 & ~t2_ws & ~g2_ws, "T2", method)
    method = xr.where(dust & g2 & ~t2 & ~t2_ws & ~g2_ws, "G2", method)
    method = xr.where(dust & g2 & t2 & t2_ws & g2_ws, "T2+WS", method)
    method = xr.where(dust & g2 & ~t2 & ~t2_ws & g2_ws, "G2+WS", method)

    # Add to dataset
    ds["G1"] = g1
    ds["G2"] = g2
    ds["G3"] = g3
    ds["T1"] = t1
    ds["T2"] = t2
    ds["T3"] = t3
    ds["G2_WS"] = g2_ws
    ds["T2_WS"] = t2_ws
    ds["G3_WS"] = g3_ws
    ds["T3_WS"] = t3_ws
    ds["DUST"] = dust
    ds["Method"] = method

    # 8. Start Date and Duration calculation
    ds = start_end_duration(ds, column="DUST")

    # 9. 72-hour duration filter
    too_long = ds.DURATION > 72.0
    for var in [
        "DUST",
        "G1",
        "G2",
        "G3",
        "T1",
        "T2",
        "T3",
        "G2_WS",
        "T2_WS",
        "G3_WS",
        "T3_WS",
    ]:
        ds[var] = xr.where(too_long, False, ds[var])
    ds["Method"] = xr.where(too_long, "NONE", ds["Method"])
    ds["START_DATE"] = xr.where(too_long, np.datetime64("NaT"), ds["START_DATE"])
    ds["DURATION"] = xr.where(too_long, np.nan, ds["DURATION"])

    # 10. Quality Control
    ds = get_quality(ds)

    # Provenance
    history = ds.attrs.get("history", "")
    now = pd.Timestamp.now()
    ds.attrs["history"] = history + f" [{now}] Applied Aero Dust Algorithm."

    return ds


def get_quality(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate quality flags for dust detection.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dust detection results.

    Returns
    -------
    xr.Dataset
        Dataset with 'QC' flag added.
    """
    pm10_rmax = (
        ds.PM10.rolling(time=3, center=True).max().bfill(dim="time").ffill(dim="time")
    )

    qc = xr.full_like(ds.DUST, 0, dtype=int)

    # Low confidence (QC=1)
    qc = xr.where(ds.Method == "G2", 1, qc)
    qc = xr.where(ds.Method == "G2+WS", 1, qc)
    qc = xr.where(ds.Method == "T2", 1, qc)

    # Medium confidence (QC=2)
    qc = xr.where((ds.Method == "G2") & (pm10_rmax > 200.0), 2, qc)
    qc = xr.where((ds.Method == "G2+WS") & (pm10_rmax > 180.0), 2, qc)
    qc = xr.where((ds.Method == "T2") & (pm10_rmax > 180.0), 2, qc)

    # High confidence (QC=3)
    qc = xr.where((ds.Method == "G2") & (pm10_rmax > 300.0), 3, qc)
    qc = xr.where((ds.Method == "G2+WS") & (pm10_rmax > 250.0), 3, qc)
    qc = xr.where((ds.Method == "T2") & (pm10_rmax > 250.0), 3, qc)
    qc = xr.where(ds.Method == "T2+WS", 3, qc)

    # No dust = 0
    qc = xr.where(~ds.DUST, 0, qc)

    ds["QC"] = qc
    return ds


def get_monthly_quantile(ds: xr.Dataset, quantile: float, col: str) -> xr.DataArray:
    """
    Calculate monthly quantiles for a variable.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    quantile : float
        Quantile to calculate (0.0 to 1.0).
    col : str
        Variable name.

    Returns
    -------
    xr.DataArray
        DataArray with monthly quantiles.
    """
    # Group by month and calculate quantile
    return ds[col].groupby("time.month").quantile(quantile)


def patch_co(
    ds: xr.Dataset, col: str, co_col: str = "CO", threshold: float = 0.5
) -> xr.DataArray:
    """
    Apply CO threshold patching to a boolean mask.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    col : str
        The boolean mask variable name.
    co_col : str, optional
        The CO variable name, by default "CO".
    threshold : float, optional
        The CO threshold, by default 0.5.

    Returns
    -------
    xr.DataArray
        Patched boolean mask.
    """
    # Original logic used shifts from -2 to 12.
    mask = ds[col] & (ds[co_col] > threshold)
    patched = mask.copy()
    for s in range(-2, 13):
        if s == 0:
            continue
        patched = patched | mask.shift(time=s, fill_value=False)
    return patched & ds[col]


def get_and_clean_obs(
    source: str = "airnow",
    start: str = "2017-06-01",
    end: str = "2017-06-02",
    path: str = ".",
    **kwargs,
) -> xr.Dataset:
    """
    Fetch and clean observation data from AQS or AirNow.

    Parameters
    ----------
    source : str, optional
        Data source, 'airnow' or 'aqs', by default "airnow".
    start : str, optional
        Start date, by default "2017-06-01".
    end : str, optional
        End date, by default "2017-06-02".
    path : str, optional
        Data directory, by default ".".
    **kwargs
        Additional arguments passed to monetio.add_data.

    Returns
    -------
    xr.Dataset
        Cleaned dataset.
    """
    import monetio.util as util
    from monetio.obs import airnow, aqs

    dates = pd.date_range(start=start, end=end, freq="h")

    if source.lower() == "airnow":
        df = airnow.add_data(dates, **kwargs)
    else:
        df = aqs.add_data(dates, param=["PM10", "WIND", "RHDP", "PM2.5"], **kwargs)

    if df.empty:
        raise ValueError(f"No data found for {source} between {start} and {end}")

    # Handling both long and wide formats from monetio
    if "obs" in df.columns:
        # Basic cleaning for long format
        df.loc[df.obs < 0, "obs"] = np.nan
        # Use monetio's long_to_wide
        df = util.long_to_wide(df)
    else:
        # For wide format, clean common variables
        for col in ["PM10", "PM2.5", "WS", "CO"]:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan

    # Rename PM2.5 to PM25 for consistency
    if "PM2.5" in df.columns:
        df.rename(columns={"PM2.5": "PM25"}, inplace=True)

    # Convert to Xarray
    if "time" not in df.columns and df.index.name != "time":
        if "time_local" in df.columns:
            df.rename(columns={"time_local": "time"}, inplace=True)

    # Ensure siteid is present
    if "siteid" not in df.columns:
        if "site" in df.columns:
            df.rename(columns={"site": "siteid"}, inplace=True)

    # Groupby siteid and time, then to xarray
    # Drop duplicates to be safe
    df = df.drop_duplicates(subset=["time", "siteid"])
    ds = df.set_index(["time", "siteid"]).sort_index().to_xarray()

    # Fix lat/lon to be 1D coordinates of siteid
    for coord in ["latitude", "longitude"]:
        if coord in ds.data_vars:
            # Take the first non-null value for each site
            # Using first() after groupby siteid
            # Or just take the mean across time dimension if it's (time, siteid)
            ds[coord] = ds[coord].mean(dim="time", skipna=True)
            ds = ds.set_coords(coord)

    return ds


def main():
    """Run the Dust Detection Algorithm from the command line."""
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        description="Dust Detection Algorithm",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--start", help="start date; YYYY-MM-DD", required=True)
    parser.add_argument("-e", "--end", help="end date; YYYY-MM-DD", required=True)
    parser.add_argument("-d", "--data", help="airnow or aqs", default="airnow")
    parser.add_argument("-o", "--output", help="output filename", required=True)
    parser.add_argument("--chunk", help="Chunk by siteid for Dask", action="store_true")

    args = parser.parse_args()

    print(f"Fetching {args.data} data...")
    ds = get_and_clean_obs(source=args.data, start=args.start, end=args.end)

    if args.chunk:
        ds = ds.chunk({"siteid": 100})
        print("Data chunked with Dask.")

    print("Running Dust Algorithm...")
    ds = dust_algorithm(ds)

    # For output, we might want to convert back to dataframe or save as netCDF
    if args.output.endswith(".csv"):
        print(f"Saving to CSV: {args.output}")
        # Only keep dust events to match original behavior
        df = ds.to_dataframe().reset_index()
        dust_events = df[df.DUST]
        dust_events.to_csv(args.output, index=False)
    else:
        print(f"Saving to NetCDF: {args.output}")
        ds.to_netcdf(args.output)

    print("Complete!")


if __name__ == "__main__":
    main()
