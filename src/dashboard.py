#!/usr/bin/env python
"""
Panel Dashboard for Dust Detection results.

Architected by Aero 🍃⚡
"""

import os

import pandas as pd
import panel as pn
import xarray as xr

from dust import dust_algorithm, get_and_clean_obs
from viz import plot_dust_interactive, plot_dust_timeseries

# Enable Panel extensions
pn.extension(sizing_mode="stretch_width")


def build_dashboard(days: int = 180) -> pn.Column:
    """
    Build the dust detection dashboard for the last N days.

    Parameters
    ----------
    days : int, optional
        Number of days of data to fetch, by default 180.

    Returns
    -------
    pn.Column
        The Panel dashboard layout.
    """
    print(f"Fetching data for the last {days} days...")
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)

    try:
        ds = get_and_clean_obs(
            source="airnow",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Fallback to a smaller period if requested period fails
        fallback_days = 30 if days > 30 else 7
        print(f"Attempting to fetch last {fallback_days} days instead...")
        start_date = end_date - pd.Timedelta(days=fallback_days)
        ds = get_and_clean_obs(
            source="airnow",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )

    # Use Dask for processing
    ds = ds.chunk({"siteid": 100})

    print("Running Dust Algorithm...")
    ds = dust_algorithm(ds)

    # Compute for summary stats
    print("Computing summary statistics...")
    # Count unique site-time combinations with valid PM10 data
    # This fulfills the "unique sites in space and time" requirement
    total_obs = int(ds.PM10.notnull().sum().compute())
    has_dust_mask = ds.DUST.any(dim="time").compute()
    dust_sites = int(has_dust_mask.sum())

    print("Creating visualizations...")
    # 1. Interactive map (Track B) - Daily Max
    # Resample to daily max for the map as requested "view the data values per day"
    ds_daily = ds.resample(time="1D").max(dim="time")

    # Safety check to ensure ds_daily is a Dataset
    if isinstance(ds_daily, xr.DataArray):
        ds_daily = ds_daily.to_dataset()

    interactive_map = plot_dust_interactive(ds_daily, var="QC")

    # 2. Spaghetti Time Series plot
    # We use hourly data for the time series but only for sites that had dust
    ts_plot = plot_dust_timeseries(ds, var="PM10")

    # Indicators for Sidebar
    indicators = pn.Column(
        pn.indicators.Number(
            name="Site Count (Space-Time)",
            value=total_obs,
            format="{value}",
            font_size="24pt",
            title_size="12pt",
        ),
        pn.indicators.Number(
            name="Sites with Dust Events",
            value=dust_sites,
            format="{value}",
            colors=[(0.1, "green"), (1, "red")],
            font_size="24pt",
            title_size="12pt",
        ),
        sizing_mode="stretch_width",
    )

    # Sidebar content
    sidebar = [
        pn.pane.Markdown("## About DustTrace"),
        pn.pane.Markdown(
            """
            DustTrace is a Pangeo-native pipeline for wind-blown dust detection.
            It uses hourly PM10 concentrations and PM2.5/PM10 ratios from AirNow,
            supplemented by ISD-Lite meteorological data (Temperature, Humidity, Wind)
            to identify and visualize dust events across North America.
            """
        ),
        pn.pane.Markdown("## Summary Statistics"),
        indicators,
        pn.pane.Markdown("## Data Sources"),
        pn.pane.Markdown(
            """
            Results are derived from the following open data repositories:

            * **Air Quality Data**: [EPA AirNow](https://www.airnow.gov/)
            * **Meteorological Data**: [NOAA NCEI ISD-Lite](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)
            """
        ),
        pn.pane.Markdown("---"),
        pn.pane.Markdown(
            """
            **Technical Reference**

            This dashboard is built using [Panel](https://panel.holoviz.org/) and
            [HvPlot](https://hvplot.holoviz.org/), following the Aero Protocol for
            backend-agnostic geospatial visualization.

            *Powered by Aero 🍃⚡*
            """
        ),
    ]

    # Main area content
    main = [
        pn.pane.Markdown(
            f"### Monitoring Period: {start_date.date()} to {end_date.date()}"
        ),
        pn.Card(
            interactive_map,
            title="Daily Max Dust Confidence (QC: 0=None, 1=Low, 2=Med, 3=High)",
            sizing_mode="stretch_both",
        ),
        pn.Card(
            ts_plot,
            title="PM10 Spaghetti Plot (Sites with Detected Dust)",
            sizing_mode="stretch_both",
        ),
    ]

    # Custom Header
    # Using NOAA Blue palette (#002D62)
    header = pn.Row(
        pn.pane.Markdown(
            "# DustTrace: Wind-Blown Dust Detection",
            styles={"color": "white", "margin": "10px"},
        ),
        styles={"background": "#002D62"},
        sizing_mode="stretch_width",
    )

    # Compose Layout
    dashboard = pn.Column(
        header,
        pn.Row(
            pn.Column(
                *sidebar,
                width=320,
                sizing_mode="stretch_height",
                styles={"background": "#f8f9fa", "padding": "10px"},
            ),
            pn.Column(*main, sizing_mode="stretch_both", styles={"padding": "10px"}),
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )

    return dashboard


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("docs", exist_ok=True)

    print("Building dashboard...")
    # For testing, we'll use 7 days to keep it fast
    # But in the GitHub Action we can set this to 30
    days_to_run = int(os.environ.get("DASHBOARD_DAYS", 7))
    db = build_dashboard(days=days_to_run)

    output_path = "docs/index.html"
    print(f"Saving dashboard to {output_path}...")
    # Using embed=True so the time slider works in the static HTML file
    db.save(output_path, title="DustTrace Dashboard", embed=True)
    print("Complete!")
