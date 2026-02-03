import numpy as np
import pandas as pd
import xarray as xr

from viz import plot_dust_interactive, plot_dust_static, plot_dust_timeseries


def create_mock_ds_with_coords(n_hours=24):
    """Create mock dataset with lat/lon for testing viz."""
    times = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    sites = ["Site1", "Site2"]

    ds = xr.Dataset(
        data_vars={
            "PM10": (("time", "siteid"), np.random.rand(n_hours, 2) * 100),
            "DUST": (
                ("time", "siteid"),
                np.random.choice([True, False], size=(n_hours, 2)),
            ),
            "QC": (("time", "siteid"), np.random.randint(0, 4, size=(n_hours, 2))),
            "Method": (("time", "siteid"), np.full((n_hours, 2), "T2", dtype="U10")),
            "latitude": (("siteid",), [34.0, 35.0]),
            "longitude": (("siteid",), [-118.0, -117.0]),
        },
        coords={
            "time": times,
            "siteid": sites,
        },
    )
    ds = ds.set_coords(["latitude", "longitude"])
    return ds


def test_plot_dust_static():
    """Test static plot creation."""
    ds = create_mock_ds_with_coords()
    ax = plot_dust_static(ds)
    assert ax is not None


def test_plot_dust_interactive():
    """Test interactive plot creation."""
    ds = create_mock_ds_with_coords()
    plot = plot_dust_interactive(ds)
    assert plot is not None


def test_plot_dust_timeseries():
    """Test time series plot creation."""
    ds = create_mock_ds_with_coords()
    # Force a dust event
    ds["DUST"].values[0, 0] = True
    plot = plot_dust_timeseries(ds)
    assert plot is not None
