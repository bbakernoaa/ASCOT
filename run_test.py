import xarray as xr
import pandas as pd
from dust import get_and_clean_obs, dust_algorithm
import os

start = "2023-02-12"
end = "2023-02-28"
source = "airnow"
with_met = True
dynamic_threshold = True
box = [25, -125, 50, -70]
fetch_start = (pd.to_datetime(start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

print(f"Fetching data from {fetch_start} to {end}...")
ds = get_and_clean_obs(source=source, start=fetch_start, end=end, with_met=with_met, box=box)
print("Running dust algorithm...")
ds_res = dust_algorithm(ds, dynamic_threshold=dynamic_threshold)
ds_res = ds_res.sel(time=slice(start, end))

for var in ds_res.variables:
    if ds_res[var].dtype == object:
        ds_res[var] = ds_res[var].astype(str)
ds_res.to_netcdf("feb_2023_test.nc", engine="h5netcdf")
print("Saved to feb_2023_test.nc")
