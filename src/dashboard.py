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
from viz import plot_dust_heatmap, plot_dust_interactive

# Enable Panel extensions
pn.extension(sizing_mode="stretch_width")


def build_dashboard(days: int = 90) -> pn.Column:
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
            with_met=True,
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
            with_met=True,
        )

    # Use Dask for processing
    ds = ds.chunk({"siteid": 100})

    print("Running Dust Algorithm...")
    ds = dust_algorithm(ds)

    # Compute for summary stats
    print("Computing summary statistics...")
    # Filter for confidence level > 1 (QC > 1) as requested
    # Compute mask first to avoid dask boolean indexing error in xarray
    mask = (ds.QC > 1).compute()
    ds_conf = ds.where(mask, drop=True)

    if "time" in ds_conf.dims and ds_conf.time.size > 0:
        # Total sites with dust events (QC > 1)
        dust_sites = int(ds_conf.siteid.size)

        # Total hours with dust events (QC > 1)
        # Count unique hours where at least one site has an event
        dust_hours = int(ds_conf.time.size)

        # Total days with dust events (QC > 1)
        # Redefined: At least 3 sites must report dust on the same day
        sites_per_day = mask.resample(time="1D").max(dim="time").sum(dim="siteid")
        dust_days = int((sites_per_day >= 3).sum())

        # Regional Events: At least 6 sites on the same day
        regional_events = int((sites_per_day >= 6).sum())
    else:
        dust_sites = 0
        dust_hours = 0
        dust_days = 0
        regional_events = 0

    print("Creating visualizations...")
    # 1. Interactive map (Track B) - Daily Max
    # Resample to daily max for the map as requested "view the data values per day"
    ds_daily = ds.resample(time="1D").max(dim="time")

    # Safety check to ensure ds_daily is a Dataset
    if isinstance(ds_daily, xr.DataArray):
        ds_daily = ds_daily.to_dataset()

    interactive_map = plot_dust_interactive(ds_daily, var="QC")

    # 2. Regional Timeline Heatmap
    # We use hourly data for the heatmap but only for sites that had dust
    ts_plot = plot_dust_heatmap(ds, var="PM10")

    # Link selections for bidirectional interactivity
    # This allows box selection on the map to filter the heatmap and vice versa
    try:
        from holoviews.selection import link_selections

        linker = link_selections.instance()
        linked_plots = linker(interactive_map + ts_plot).cols(1)
        map_final = linked_plots[0]
        ts_final = linked_plots[1]
    except Exception as e:
        print(f"Error linking selections: {e}")
        map_final = interactive_map
        ts_final = ts_plot

    # Indicators for Sidebar
    indicators = pn.Column(
        pn.indicators.Number(
            name="Days with Dust Events",
            value=dust_days,
            format="{value}",
            font_size="20pt",
            title_size="10pt",
        ),
        pn.indicators.Number(
            name="Hours with Dust Events",
            value=dust_hours,
            format="{value}",
            font_size="20pt",
            title_size="10pt",
        ),
        pn.indicators.Number(
            name="Sites with Dust Events",
            value=dust_sites,
            format="{value}",
            colors=[(0.1, "green"), (1, "red")],
            font_size="20pt",
            title_size="10pt",
        ),
        pn.indicators.Number(
            name="Significant Regional Events",
            value=regional_events,
            format="{value}",
            font_size="20pt",
            title_size="10pt",
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
        pn.pane.Markdown(
            "*Statistics reflect high-confidence events (QC > 1). "
            "Days with Dust Events requires ≥3 sites. "
            "Regional Events requires ≥6 sites.*"
        ),
        indicators,
        pn.pane.Markdown("> **Note**: QC=0 indicates no dust detected."),
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
        pn.pane.Markdown(f"### Monitoring Period: {start_date.date()} to {end_date.date()}"),
        pn.Card(
            map_final,
            title="Daily Max Dust Confidence (QC: 0=None, 1=Low, 2=Med, 3=High)",
            sizing_mode="stretch_both",
        ),
        pn.Card(
            ts_final,
            title="Regional Timeline: PM10 Heatmap (Use Box Select on map to filter sites)",
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
