import numpy as np
import pandas as pd
import xarray as xr

from dust import dust_algorithm


def create_mock_ds(n_hours=24):
    """Create mock dataset for testing."""
    times = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    sites = ["Site1", "Site2"]

    # Create a dust-like event
    pm10_vals = np.ones((n_hours, 2)) * 50
    # Dust event at Site1
    event_start = 10
    event_end = min(15, n_hours)
    pm10_vals[event_start:event_end, 0] = 300

    pm25_vals = pm10_vals * 0.1  # Low ratio, typical for dust

    ws_vals = np.ones((n_hours, 2)) * 5
    ws_vals[event_start:event_end, 0] = 10  # High wind speed

    ds = xr.Dataset(
        data_vars={
            "PM10": (("time", "siteid"), pm10_vals),
            "PM25": (("time", "siteid"), pm25_vals),
            "WS": (("time", "siteid"), ws_vals),
        },
        coords={
            "time": times,
            "siteid": sites,
        },
    )
    return ds


def test_dust_algorithm_eager_vs_lazy():
    """Verify that results are identical for NumPy and Dask backends."""
    ds_numpy = create_mock_ds()
    ds_dask = ds_numpy.chunk({"time": 12})

    res_numpy = dust_algorithm(ds_numpy)
    res_dask = dust_algorithm(ds_dask)

    # Check that Dask result is still lazy
    assert hasattr(res_dask.DUST.data, "dask")

    # Compare results
    xr.testing.assert_allclose(
        res_numpy.DUST.astype(int), res_dask.DUST.compute().astype(int)
    )
    xr.testing.assert_allclose(res_numpy.QC, res_dask.QC.compute())

    # Ensure dust was actually detected in our mock event
    assert res_numpy.DUST.sel(siteid="Site1").isel(time=12)
    assert res_numpy.QC.sel(siteid="Site1").isel(time=12) >= 2


def test_dust_algorithm_provenance():
    """Check if history attribute is updated."""
    ds = create_mock_ds()
    res = dust_algorithm(ds)
    assert "Applied Aero Dust Algorithm" in res.attrs["history"]


def test_dust_algorithm_missing_pm25():
    """Test behavior when PM25 is missing."""
    ds = create_mock_ds().drop_vars("PM25")
    res = dust_algorithm(ds)
    assert "PM25" in res.data_vars
    assert res.PM25.isnull().all()
    # Should still run but might have different detection results
    assert "DUST" in res.data_vars


def test_dust_algorithm_duration_filter():
    """Test that events > 72h are filtered out."""
    # Create an 80h event
    n_hours = 100
    ds = create_mock_ds(n_hours=n_hours)

    # Modify Site1 to have 80h of high PM10
    pm10_vals = ds.PM10.values
    pm10_vals[10:90, 0] = 300
    ds["PM10"] = (("time", "siteid"), pm10_vals)

    res = dust_algorithm(ds)

    # Should be filtered out because duration is 80h > 72h
    assert not res.DUST.sel(siteid="Site1").any()
    assert np.isnan(res.DURATION.sel(siteid="Site1")).all()
