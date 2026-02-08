"""
Generate a dashboard from processed dust detection results.
"""

import os

import panel as pn
import xarray as xr

from viz import plot_dust_interactive, plot_dust_timeseries


def build_dashboard(nc_file, output_html):
    print(f"Loading {nc_file}...")
    ds = xr.open_dataset(nc_file)

    ds_daily = ds.resample(time="1D").max(dim="time")
    if isinstance(ds_daily, xr.DataArray):
        ds_daily = ds_daily.to_dataset()

    interactive_map = plot_dust_interactive(ds_daily, var="QC")
    ts_plot = plot_dust_timeseries(ds, var="PM10")

    dashboard = pn.Column(
        pn.pane.Markdown("# Dust Detection Test Results"),
        pn.Row(
            pn.Card(
                interactive_map,
                title="Daily Max Confidence (QC)",
                sizing_mode="stretch_both",
            ),
            pn.Card(ts_plot, title="PM10 Time Series", sizing_mode="stretch_both"),
        ),
    )

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    dashboard.save(output_html, embed=True)
    print(f"Dashboard saved to {output_html}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        build_dashboard(
            sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "dashboard.html"
        )
    else:
        print("Usage: python generate_dashboard.py <input_nc> <output_html>")
