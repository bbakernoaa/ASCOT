#!/usr/bin/env python
"""
Dust Detection Algorithm using Hourly EPA AQS/AIRNOW Surface Monitors.

Architected by Aero 🍃⚡
"""

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# Disable Pandas 3.0+ automatic string inference to maintain Dask compatibility.
# This prevents metadata mismatches between empty and non-empty partitions.
if hasattr(pd.options, "future") and hasattr(pd.options.future, "infer_string"):
    pd.options.future.infer_string = False


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


def fetch_isd_lite(dates: pd.DatetimeIndex, box: list[float]) -> Optional[xr.Dataset]:
    """
    Fetch ISD-Lite data for a given date range and bounding box.

    Robustly handles missing files and metadata issues.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Date range to fetch.
    box : list of float
        Bounding box [latmin, lonmin, latmax, lonmax].

    Returns
    -------
    xr.Dataset or None
        Fetched dataset or None if no data found.
    """
    try:
        from monetio.obs.ish_lite import ISH
        from monetio.readers.ish_lite import ISHLiteReader
    except ImportError:
        print("monetio not installed. Cannot fetch met data.")
        return None

    try:
        ish = ISH()
        ish.dates = dates
        ish.read_ish_history()
        dfloc = ish.subset_sites(
            latmin=box[0], lonmin=box[1], latmax=box[2], lonmax=box[3]
        )

        urls = ish.build_urls(sites=dfloc)
        if urls.empty:
            print("No ISD-Lite stations found in box.")
            return None

        # Robust aggregation: skip files that fail to load (e.g. 404)
        import dask

        @dask.delayed
        def safe_read_csv(url):
            try:
                return ish.read_csv(url)
            except Exception as e:
                print(f"Skipping {url}: {e}")
                return pd.DataFrame()

        # Use dask.compute on the list of delayed objects to get real DataFrames
        url_list = urls["name"] if "name" in urls else urls
        dfs = dask.compute(*[safe_read_csv(url) for url in url_list])
        df_ish = pd.concat([df for df in dfs if not df.empty], ignore_index=True)

        if df_ish.empty:
            print("No ISD-Lite data successfully fetched.")
            return None

        # Harmonize and convert to xarray
        reader = ISHLiteReader()
        df_ish = reader.harmonize(df_ish)
        # Filter to requested dates
        df_ish = df_ish.loc[(df_ish.time >= dates.min()) & (df_ish.time <= dates.max())]
        # Merge with station info
        df_ish = pd.merge(
            df_ish, dfloc, how="left", left_on="siteid", right_on="station_id"
        ).rename(columns={"ctry": "country"})

        return reader.to_xarray(df_ish)
    except Exception as e:
        print(f"Error fetching ISD-Lite data: {e}")
        return None


def add_met_to_airnow(ds: xr.Dataset) -> xr.Dataset:
    """
    Supplement AirNow dataset with ISD-Lite meteorological data.

    Parameters
    ----------
    ds : xr.Dataset
        Input AirNow dataset.

    Returns
    -------
    xr.Dataset
        Dataset with supplemented meteorological variables.
    """
    from scipy.spatial import cKDTree

    # Get date range from ds
    if "time" not in ds.dims:
        return ds

    dates = pd.to_datetime(ds.time.values)
    start = dates.min()
    end = dates.max()

    # Get bounding box
    latmin = ds.latitude.min().item()
    latmax = ds.latitude.max().item()
    lonmin = ds.longitude.min().item()
    lonmax = ds.longitude.max().item()
    box = [latmin - 1, lonmin - 1, latmax + 1, lonmax + 1]

    # Fetch ISD-Lite data
    print(f"Fetching ISD-Lite data for box {box}...")
    ish_dates = pd.date_range(start, end, freq="h")
    try:
        ds_ish = fetch_isd_lite(ish_dates, box)
    except Exception as e:
        print(f"Unexpected error during ISD-Lite fetch: {e}")
        ds_ish = None

    if ds_ish is None:
        print("No ISD-Lite data found.")
        return ds

    # Build KDTree for nearest neighbor
    if "latitude" not in ds_ish.coords or "longitude" not in ds_ish.coords:
        print("ISD-Lite dataset missing latitude/longitude.")
        return ds

    # Extract unique station locations from ds_ish
    ish_sites = ds_ish.siteid.values
    ish_coords = np.column_stack([ds_ish.latitude.values, ds_ish.longitude.values])
    tree = cKDTree(ish_coords)

    # For each site in ds, find nearest ISD site
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        print("Missing latitude/longitude in AirNow dataset.")
        return ds

    airnow_coords = np.column_stack([ds.latitude.values, ds.longitude.values])
    dist, idx = tree.query(airnow_coords)

    nearest_ish_sites = ish_sites[idx]

    met_vars = ["temp", "dew_pt_temp", "ws", "press"]
    met_ds_list = []

    for var in met_vars:
        if var in ds_ish.data_vars:
            # Selection for each site
            var_data = ds_ish[var].sel(siteid=nearest_ish_sites)
            var_data["siteid"] = ds.siteid
            met_ds_list.append(
                var_data.rename(var.upper() if var != "ws" else "WS_MET")
            )

    if not met_ds_list:
        return ds

    ds_met = xr.merge(met_ds_list)

    # Reindex time to match AirNow exactly
    ds_met = ds_met.reindex(time=ds.time, method="nearest")

    # Calculate RH
    if "TEMP" in ds_met and "DEW_PT_TEMP" in ds_met:
        t = ds_met.TEMP
        td = ds_met.DEW_PT_TEMP
        # August-Roche-Magnus formula
        rh = 100 * (
            np.exp((17.625 * td) / (243.04 + td)) / np.exp((17.625 * t) / (243.04 + t))
        )
        ds_met["RH"] = rh

    # Merge into original ds
    if "WS" in ds.data_vars:
        ds["WS"] = ds["WS"].fillna(ds_met["WS_MET"])
    else:
        if "WS_MET" in ds_met:
            ds["WS"] = ds_met["WS_MET"]

    # Add other met vars
    for v in ds_met.data_vars:
        if v not in ds.data_vars:
            ds[v] = ds_met[v]

    # Provenance
    history_attr = ds.attrs.get("history", "")
    now = pd.Timestamp.now()
    ds.attrs["history"] = (
        history_attr + f" [{now}] Supplemented met data from ISD-Lite."
    )

    return ds


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
    upper_threshold: float = 150.0,
    dynamic_threshold: bool = False,
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
    if dynamic_threshold:
        # Calculate rolling 30-day mean and std
        # Ensure window is not larger than the dataset length
        window_size = min(24 * 30, len(ds.time))
        pm10_rolling = ds.PM10.rolling(time=window_size, min_periods=1, center=False)
        pm10_mean = pm10_rolling.mean()
        pm10_std = pm10_rolling.std()
        pm10_lower_thr = pm10_mean + 2 * pm10_std
        # Fill NaN with static threshold if rolling window has no data
        pm10_lower_thr = pm10_lower_thr.fillna(lower_threshold)
    else:
        pm10_lower_thr = lower_threshold

    pm10_upper_thr = 180.0
    pm10_98_thr = 85.0

    # 4. Dust Levels (G-series and T-series)
    g1 = (pm10_rmean > pm10_lower_thr) & (pm10_rmax >= pm10_upper_thr)
    g2 = (pm10_rmean > pm10_lower_thr) & (pm10_rmax >= upper_threshold)
    g3 = (pm10_rmean > pm10_98_thr) & (pm10_rmax >= upper_threshold)

    t1 = g2 & (ds.RATIO <= 0.40)
    t2 = g2 & (ds.RATIO <= 0.25)
    t3 = g2 & (ds.RATIO <= 0.15)

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
    # More robust detection: Require a low PM2.5/PM10 ratio (T2)
    # and either the PM10 threshold (G2) or high wind speeds (G2_WS).
    # If PM2.5 is missing, we rely on PM10 and Wind Speed (lower confidence).

    # Strictly require ratio < 0.25 if available
    ratio_check = (ds.RATIO < 0.25) | ds.RATIO.isnull()

    dust = (t2 | g2_ws) & ratio_check

    # Add RH dependence if available
    if "RH" in ds.data_vars:
        dust = dust & (ds.RH < 40.0)

    # Add Temperature filter (exclude freezing temps to avoid road salt)
    if "TEMP" in ds.data_vars:
        dust = dust & (ds.TEMP > 0.0)

    # 7. Method classification
    method = xr.full_like(dust, "NONE", dtype="U10")
    # T2: High PM10 + Low Ratio
    method = xr.where(dust & t2 & ~g2_ws, "T2", method)
    # G2+WS: High PM10 + High Wind (and low ratio if available)
    method = xr.where(dust & ~t2 & g2_ws, "G2+WS", method)
    # T2+WS: High PM10 + Low Ratio + High Wind
    method = xr.where(dust & t2 & g2_ws, "T2+WS", method)
    # G2: Just high PM10 (should be rare with above logic but kept for completeness)
    method = xr.where(dust & g2 & ~t2 & ~g2_ws, "G2", method)

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
    with_met: bool = False,
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
    with_met : bool, optional
        Whether to supplement with met data, by default False.
    **kwargs
        Additional arguments passed to monetio.load.

    Returns
    -------
    xr.Dataset
        Cleaned dataset.
    """
    import monetio

    dates = pd.date_range(start=start, end=end, freq="h")

    if source.lower() == "airnow":
        ds = monetio.load("airnow", dates=dates, as_xarray=True, **kwargs)
    else:
        ds = monetio.load(
            "aqs",
            dates=dates,
            param=["PM10", "WIND", "RHDP", "PM2.5"],
            as_xarray=True,
            **kwargs,
        )

    if ds is None:
        raise ValueError(f"No data found for {source} between {start} and {end}")

    # Fix for StringDtype compatibility (Pandas 3.0+)
    for var in ds.variables:
        if isinstance(ds[var].dtype, pd.StringDtype):
            ds[var] = ds[var].astype(object)

    # Manually filter by box if provided
    if "box" in kwargs:
        b = kwargs["box"]
        ds = ds.where(
            (ds.latitude >= b[0])
            & (ds.longitude >= b[1])
            & (ds.latitude <= b[2])
            & (ds.longitude <= b[3]),
            drop=True,
        )

    # Cleaning for xarray
    for var in ["PM10", "PM2.5", "WS", "CO"]:
        if var in ds.data_vars:
            ds[var] = ds[var].where(ds[var] >= 0)

    # Rename PM2.5 to PM25 for consistency
    if "PM2.5" in ds.data_vars:
        ds = ds.rename({"PM2.5": "PM25"})

    # Ensure only dedicated PM10 monitors are used
    if "PM10" in ds.data_vars:
        # 1. Drop sites that have no PM10 data at all
        has_pm10 = ds.PM10.notnull().any(dim="time").compute()
        ds = ds.sel(siteid=has_pm10)

        # 2. Drop sites where PM10 is suspiciously correlated with PM2.5
        # (Indicates calculated PM10 from PM2.5 only sensors)
        if "PM25" in ds.data_vars:
            # Check for constant ratio or identical values
            ratio = ds.PM25 / ds.PM10
            # We use a small epsilon for float comparison
            is_constant = ratio.std(dim="time") < 1e-5
            is_same = (ds.PM10 == ds.PM25).all(dim="time")

            # Compute suspicious mask
            suspicious = (is_constant | is_same).compute()
            if suspicious.any():
                print(
                    f"Dropping {int(suspicious.sum())} sites with suspicious "
                    "PM10/PM2.5 correlation."
                )
                ds = ds.sel(siteid=~suspicious)

    if with_met:
        ds = add_met_to_airnow(ds)

    # Resample to hourly to ensure consistency for the rolling windows
    # Use '1h' (lowercase) for Pandas 3.0+ compatibility
    if "time" in ds.dims:
        ds = ds.resample(time="1h").mean()

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
    parser.add_argument(
        "--with-met", help="Supplement with ISD-Lite met data", action="store_true"
    )
    parser.add_argument(
        "--dynamic-threshold",
        help="Use dynamic PM10 threshold (30-day mean + 2sigma)",
        action="store_true",
    )

    args = parser.parse_args()

    if args.dynamic_threshold:
        fetch_start = (pd.to_datetime(args.start) - pd.Timedelta(days=30)).strftime(
            "%Y-%m-%d"
        )
        print(f"Dynamic threshold requested. Fetching data from {fetch_start}...")
    else:
        fetch_start = args.start

    print(f"Fetching {args.data} data...")
    ds = get_and_clean_obs(
        source=args.data, start=fetch_start, end=args.end, with_met=args.with_met
    )

    if args.chunk:
        ds = ds.chunk({"siteid": 100})
        print("Data chunked with Dask.")

    print("Running Dust Algorithm...")
    ds = dust_algorithm(ds, dynamic_threshold=args.dynamic_threshold)

    # Subset to requested period for output
    ds = ds.sel(time=slice(args.start, args.end))

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
