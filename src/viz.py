#!/usr/bin/env python
"""
Visualization module for Dust Detection results.

Follows the Aero Protocol Two-Track Rule:
Track A: Static (Matplotlib + Cartopy)
Track B: Interactive (HvPlot / GeoViews)

Architected by Aero 🍃⚡
"""

from typing import Any, Optional, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

# Try to import interactive tools
try:
    import holoviews as hv
    import hvplot.pandas  # noqa: F401
    import hvplot.xarray  # noqa: F401
except ImportError:
    hv = None


def plot_dust_static(
    ds: xr.Dataset,
    time: Optional[Union[str, pd.Timestamp]] = None,
    var: str = "DUST",
    ax: Optional[plt.Axes] = None,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    **kwargs: Any,
) -> plt.Axes:
    """
    Create a static map of dust events using Matplotlib and Cartopy.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing dust detection results and site coordinates.
    time : str or pd.Timestamp, optional
        The time slice to plot. If None, the first time step is used.
    var : str, optional
        The variable to plot, by default "DUST".
    ax : plt.Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    projection : ccrs.Projection, optional
        Cartopy projection, by default ccrs.PlateCarree().
    **kwargs : Any
        Additional arguments passed to ax.scatter.

    Returns
    -------
    plt.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": projection})

    if time is not None:
        data = ds.sel(time=time)
    else:
        data = ds.isel(time=0)

    # Add features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES, linestyle=":", edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, linestyle="-", edgecolor="black")

    # Ensure latitude and longitude are available
    if "latitude" not in data or "longitude" not in data:
        # Try to get from coordinates if they are there
        if "latitude" in ds.coords and "longitude" in ds.coords:
            lat = ds.coords["latitude"]
            lon = ds.coords["longitude"]
        else:
            raise ValueError("Dataset must contain 'latitude' and 'longitude'.")
    else:
        lat = data.latitude
        lon = data.longitude

    # Filter for valid lat/lon
    valid_mask = lat.notnull() & lon.notnull()
    valid_sites = data.siteid.where(valid_mask, drop=True)
    data_valid = data.sel(siteid=valid_sites)

    lat_valid = lat.sel(siteid=valid_sites)
    lon_valid = lon.sel(siteid=valid_sites)

    # Plot points where the variable is True or has values
    events = data_valid.where(data_valid[var] > 0, drop=True)
    non_events = data_valid.where(data_valid[var] <= 0, drop=True)

    # Plot non-events
    if len(non_events.siteid) > 0:
        ax.scatter(
            lon_valid.sel(siteid=non_events.siteid),
            lat_valid.sel(siteid=non_events.siteid),
            c="lightgray",
            s=10,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            label="No Dust",
        )

    # Plot events
    if len(events.siteid) > 0:
        # Convert to dataframe for easier sorting
        events_df = events[[var]].assign_coords(latitude=lat_valid, longitude=lon_valid).to_dataframe().reset_index()
        # Sort by var so higher values are plotted last (on top)
        events_df = events_df.sort_values(by=var, ascending=True)

        im = ax.scatter(
            events_df.longitude,
            events_df.latitude,
            c=events_df[var],
            s=50,
            cmap="autumn_r",
            edgecolor="black",
            transform=ccrs.PlateCarree(),
            label="Dust Event",
            **kwargs,
        )
        plt.colorbar(im, ax=ax, label=var, shrink=0.6)

    ax.legend(loc="lower right")

    current_time = pd.to_datetime(data.time.values).strftime("%Y-%m-%d %H:%M")
    ax.set_title(f"Dust Detection: {var}\nTime: {current_time}")

    return ax


def plot_dust_heatmap(
    ds: xr.Dataset,
    var: str = "PM10",
    **kwargs: Any,
) -> Any:
    """
    Create a heatmap of PM10 concentrations at sites with detected dust.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing dust detection results.
    var : str, optional
        The variable to plot on color axis, by default "PM10".
    **kwargs : Any
        Additional arguments passed to hvplot.

    Returns
    -------
    hv.Element
        The interactive HoloViews object.
    """
    if hv is None:
        raise ImportError("holoviews and hvplot are required for interactive plotting.")

    # Only show sites that had at least one dust event
    has_event = ds.DUST.any(dim="time")
    ds_to_plot = ds.sel(siteid=has_event)

    if ds_to_plot.siteid.size == 0:
        return hv.Div("No dust events detected in this period.")

    # Convert to dataframe
    # Include DUST and QC for context
    cols_to_keep = [var]
    if "DUST" in ds_to_plot:
        cols_to_keep.append("DUST")
    if "QC" in ds_to_plot:
        cols_to_keep.append("QC")

    df = ds_to_plot[cols_to_keep].to_dataframe().reset_index()
    df["time"] = pd.to_datetime(df["time"])

    # Heatmap
    # We use heatmap to show PM10 over time for each site
    heatmap = df.hvplot.heatmap(
        x="time",
        y="siteid",
        C=var,
        cmap="viridis",
        title=f"Regional Timeline of {var} (Heatmap)",
        xlabel="Time",
        ylabel="Site ID",
        clabel=f"{var} Concentration",
        rot=45,
        **kwargs,
    )

    return heatmap


def plot_dust_interactive(
    ds: xr.Dataset,
    var: str = "DUST",
    rasterize: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Create an interactive dashboard using HvPlot.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing dust detection results.
    var : str, optional
        The variable to plot, by default "DUST".
    rasterize : bool, optional
        Whether to rasterize the plot for large datasets, by default False.
    **kwargs : Any
        Additional arguments passed to hvplot.

    Returns
    -------
    hv.Element
        The interactive HoloViews object.
    """
    if hv is None:
        raise ImportError("holoviews and hvplot are required for interactive plotting.")

    # Select columns to plot
    cols = ["latitude", "longitude", var, "PM10", "Method"]
    available_cols = [c for c in cols if c in ds or c in ds.coords]

    # Filtering sites to reduce size
    has_event = (ds[var] > 0).any(dim="time")
    ds_to_plot = ds.sel(siteid=has_event)

    if ds_to_plot.siteid.size == 0:
        ds_to_plot = ds.isel(time=slice(0, 1))

    # Convert to dataframe
    df = ds_to_plot[available_cols].to_dataframe().reset_index()
    df["time"] = pd.to_datetime(df["time"])

    # Ensure higher confidence values are plotted on top
    if var in df.columns:
        df = df.sort_values(by=var, ascending=True)

    plot = df.hvplot.points(
        x="longitude",
        y="latitude",
        c=var,
        geo=True,
        tiles="OSM",
        cmap="autumn_r",
        hover_cols=["siteid", "time", "PM10", "Method"],
        title=f"Dust Detection: {var}",
        groupby="time",
        dynamic=False,
        xlim=(-125, -65),
        ylim=(25, 50),
        **kwargs,
    )

    if rasterize:
        from holoviews.operation.datashader import rasterize as hv_rasterize

        return hv_rasterize(plot)

    return plot


def plot_dust_timeseries(
    ds: xr.Dataset,
    var: str = "PM10",
    **kwargs: Any,
) -> Any:
    """
    Create a spaghetti time series plot of sites with detected dust.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing dust detection results.
    var : str, optional
        The variable to plot on Y-axis, by default "PM10".
    **kwargs : Any
        Additional arguments passed to hvplot.

    Returns
    -------
    hv.Element
        The interactive HoloViews object.
    """
    if hv is None:
        raise ImportError("holoviews and hvplot are required for interactive plotting.")

    # Only show sites that had at least one dust event
    has_event = ds.DUST.any(dim="time")
    ds_to_plot = ds.sel(siteid=has_event)

    if ds_to_plot.siteid.size == 0:
        return hv.Div("No dust events detected in this period.")

    # Convert to dataframe
    # Include DUST and QC for markers
    cols_to_keep = [var]
    if "DUST" in ds_to_plot:
        cols_to_keep.append("DUST")
    if "QC" in ds_to_plot:
        cols_to_keep.append("QC")

    df = ds_to_plot[cols_to_keep].to_dataframe().reset_index()
    df["time"] = pd.to_datetime(df["time"])

    # Spaghetti plot
    lines = df.hvplot.line(
        x="time",
        y=var,
        by="siteid",
        alpha=0.4,
        legend=False,
        title=f"Time Series of {var} at Dust-Affected Sites (Markers show detections)",
        ylabel=f"{var} concentration",
        **kwargs,
    )

    # Markers for dust events (QC > 0 or DUST is True)
    if "DUST" in df.columns:
        dust_df = df[df.DUST].copy()
        if not dust_df.empty:
            markers = dust_df.hvplot.scatter(
                x="time",
                y=var,
                by="siteid",
                c="QC" if "QC" in dust_df.columns else "red",
                cmap="autumn_r",
                size=40,
                marker="circle",
                alpha=0.8,
                hover_cols=["siteid", "time", "QC"] if "QC" in dust_df.columns else ["siteid", "time"],
                legend=False,
            )
            return lines * markers

    return lines
