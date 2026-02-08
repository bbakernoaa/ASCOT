"""
Script to reproduce the Feb 2023 Dust Detection test.
Architected by Aero 🍃⚡
"""

import os

import pandas as pd

from dust import dust_algorithm, get_and_clean_obs


def run_feb_2023_test():
    start = "2023-02-12"
    end = "2023-02-28"

    # Fetch 7 days of lookback for dynamic threshold
    fetch_start = (pd.to_datetime(start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"Fetching data for {start} to {end}...")
    ds = get_and_clean_obs(
        source="airnow",
        start=fetch_start,
        end=end,
        with_met=True,
        box=[25, -125, 50, -70],
    )

    print("Running dust algorithm...")
    ds = dust_algorithm(ds, dynamic_threshold=True)

    # Subset to requested period
    ds = ds.sel(time=slice(start, end))

    # Save results
    os.makedirs("data", exist_ok=True)
    output_path = "data/feb_2023_test_results.nc"

    # Convert object types for NetCDF compatibility
    for var in ds.variables:
        if ds[var].dtype == object:
            ds[var] = ds[var].astype(str)

    ds.to_netcdf(output_path, engine="h5netcdf")
    print(f"Results saved to {output_path}")

    # Print Summary
    dust_count = int(ds.DUST.sum())
    sites_count = int(ds.DUST.any(dim="time").sum())
    print("\nSummary:")
    print(f"Total Dust Event-Hours: {dust_count}")
    print(f"Sites with Events: {sites_count}")


if __name__ == "__main__":
    run_feb_2023_test()
