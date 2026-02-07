import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from dust import main


def test_cli_csv_output(tmp_path):
    """Test CLI with CSV output."""
    output_file = tmp_path / "test_output.csv"

    # Mock data ingestion
    # We provide enough data for the algorithm to run
    times = pd.date_range("2023-01-01", periods=10, freq="h")
    mock_ds = xr.Dataset(
        {
            "PM10": (("time", "siteid"), np.full((10, 1), 200.0)),
            "PM25": (("time", "siteid"), np.full((10, 1), 40.0)),
            "WS": (("time", "siteid"), np.full((10, 1), 10.0)),
        },
        coords={"time": times, "siteid": ["site1"]},
    )

    with patch("dust.get_and_clean_obs", return_value=mock_ds):
        # Mock sys.argv
        test_args = [
            "dust.py",
            "-s",
            "2023-01-01",
            "-e",
            "2023-01-01",
            "-d",
            "airnow",
            "-o",
            str(output_file),
        ]
        with patch.object(sys, "argv", test_args):
            main()

    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert not df.empty
    assert "DUST" in df.columns
    assert df.DUST.all()


def test_cli_netcdf_output(tmp_path):
    """Test CLI with NetCDF output."""
    output_file = tmp_path / "test_output.nc"

    times = pd.date_range("2023-01-01", periods=10, freq="h")
    mock_ds = xr.Dataset(
        {
            "PM10": (("time", "siteid"), np.full((10, 1), 200.0)),
            "PM25": (("time", "siteid"), np.full((10, 1), 40.0)),
            "WS": (("time", "siteid"), np.full((10, 1), 10.0)),
        },
        coords={"time": times, "siteid": ["site1"]},
    )

    with patch("dust.get_and_clean_obs", return_value=mock_ds):
        test_args = [
            "dust.py",
            "-s",
            "2023-01-01",
            "-e",
            "2023-01-01",
            "-d",
            "airnow",
            "-o",
            str(output_file),
        ]
        with patch.object(sys, "argv", test_args):
            main()

    assert output_file.exists()
    ds = xr.open_dataset(output_file)
    assert "DUST" in ds.data_vars
    assert ds.DUST.all()


def test_cli_dynamic_threshold(tmp_path):
    """Test CLI with dynamic threshold flag."""
    output_file = tmp_path / "test_output.nc"

    # Needs more data for dynamic threshold (30 days)
    times = pd.date_range("2022-12-01", periods=24 * 40, freq="h")
    mock_ds = xr.Dataset(
        {
            "PM10": (("time", "siteid"), np.full((24 * 40, 1), 200.0)),
            "PM25": (("time", "siteid"), np.full((24 * 40, 1), 40.0)),
            "WS": (("time", "siteid"), np.full((24 * 40, 1), 10.0)),
        },
        coords={"time": times, "siteid": ["site1"]},
    )

    with patch("dust.get_and_clean_obs", return_value=mock_ds) as mock_get:
        test_args = [
            "dust.py",
            "-s",
            "2023-01-01",
            "-e",
            "2023-01-05",
            "-d",
            "airnow",
            "-o",
            str(output_file),
            "--dynamic-threshold",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that fetch start date was shifted back 30 days
        args, kwargs = mock_get.call_args
        assert kwargs["start"] == "2022-12-02"  # 2023-01-01 minus 30 days is 2022-12-02

    assert output_file.exists()
