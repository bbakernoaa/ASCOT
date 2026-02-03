from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from dust import add_met_to_airnow, dust_algorithm, fill_gaps


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


def test_fill_gaps_eager_vs_lazy():
    """Verify fill_gaps logic twice: Eager and Lazy."""
    data = xr.DataArray(
        [True, False, True, False, False, True, True],
        dims="time",
        coords={"time": pd.date_range("2023-01-01", periods=7, freq="h")},
    )

    # Eager
    res_eager = fill_gaps(data)
    expected = [True, True, True, False, False, True, True]
    np.testing.assert_array_equal(res_eager.values, expected)

    # Lazy
    data_lazy = data.chunk({"time": 3})
    res_lazy = fill_gaps(data_lazy)
    assert hasattr(res_lazy.data, "dask")
    np.testing.assert_array_equal(res_lazy.compute().values, expected)


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


def test_dust_algorithm_rh():
    """Verify RH dependence in dust algorithm."""
    # Create synthetic dataset
    time = pd.date_range("2023-01-01", periods=10, freq="h")
    siteid = ["site1"]
    ds = xr.Dataset(
        {
            "PM10": (("time", "siteid"), np.full((10, 1), 200.0)),
            "PM25": (("time", "siteid"), np.full((10, 1), 20.0)),
            "WS": (("time", "siteid"), np.full((10, 1), 10.0)),
            "RH": (("time", "siteid"), np.full((10, 1), 50.0)),  # Above 40
        },
        coords={"time": time, "siteid": siteid},
    )

    # Run algorithm
    ds_out = dust_algorithm(ds)
    # Should be False because RH > 40
    assert not ds_out.DUST.any()

    # Set RH < 40
    ds["RH"] = (("time", "siteid"), np.full((10, 1), 30.0))
    ds_out = dust_algorithm(ds)
    # Should be True
    assert ds_out.DUST.any()


def test_dynamic_threshold():
    """Verify dynamic threshold logic in dust algorithm."""
    # Create synthetic dataset with 40 days of data
    np.random.seed(42)
    time = pd.date_range("2023-01-01", periods=24 * 40, freq="h")
    siteid = ["site1"]
    # Mean = 50, Std = 5
    pm10 = np.random.normal(50, 5, (24 * 40, 1))
    pm10 = np.maximum(pm10, 0)
    # Add a peak at the end that is above dynamic threshold but below 100
    pm10[-5:] = 80

    ds = xr.Dataset(
        {
            "PM10": (("time", "siteid"), pm10),
            "PM25": (("time", "siteid"), pm10 * 0.1),
            "WS": (("time", "siteid"), np.full((24 * 40, 1), 10.0)),
        },
        coords={"time": time, "siteid": siteid},
    )

    # Static threshold 100
    # Also set upper_threshold lower than our peak (80) so it can trigger
    ds_static = dust_algorithm(ds, lower_threshold=100.0, upper_threshold=70.0)
    assert not ds_static.DUST.any()

    # Dynamic threshold
    ds_dynamic = dust_algorithm(
        ds, lower_threshold=100.0, upper_threshold=70.0, dynamic_threshold=True
    )
    assert ds_dynamic.DUST.any()


def test_add_met_to_airnow():
    """Test supplementation of AirNow data with ISD-Lite met data."""
    # Create mock AirNow dataset
    time = pd.date_range("2023-01-01", periods=2, freq="h")
    ds = xr.Dataset(
        {"PM10": (("time", "siteid"), [[10], [20]])},
        coords={
            "time": time,
            "siteid": ["site1"],
            "latitude": (("siteid",), [40.0]),
            "longitude": (("siteid",), [-100.0]),
        },
    )
    ds = ds.set_coords(["latitude", "longitude"])

    # Mock ish_lite.add_data
    mock_df_ish = pd.DataFrame(
        {
            "time": [time[0], time[1]],
            "siteid": ["ISD1", "ISD1"],
            "temp": [20.0, 21.0],
            "dew_pt_temp": [10.0, 11.0],
            "ws": [5.0, 6.0],
        }
    )

    # Mock ish_lite.ISH.read_ish_history
    mock_history = pd.DataFrame(
        {"station_id": ["ISD1"], "latitude": [40.1], "longitude": [-100.1]}
    )

    with (
        patch("monetio.obs.ish_lite.add_data", return_value=mock_df_ish),
        patch("monetio.obs.ish_lite.ISH") as mock_ish_cls,
    ):
        mock_ish_inst = mock_ish_cls.return_value
        mock_ish_inst.read_ish_history.return_value = mock_history

        ds_out = add_met_to_airnow(ds)

        assert "TEMP" in ds_out
        assert "DEW_PT_TEMP" in ds_out
        assert "RH" in ds_out
        assert "WS" in ds_out
        assert not ds_out.TEMP.isnull().any()
