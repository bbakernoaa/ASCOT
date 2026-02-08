import xarray as xr
import pandas as pd
import numpy as np

ds = xr.open_dataset("feb_2023_test.nc")
dust_mask = ds.DUST.compute()
total_dust_hours = int(dust_mask.sum())

print(f"Total DUST event-hours: {total_dust_hours}")

# QC Distribution for DUST events
dust_qc = ds.QC.compute().where(dust_mask).to_series().dropna().value_counts().sort_index()
print("\nQC Level Distribution for DUST events:")
print(dust_qc)

# Method Distribution for DUST events
method_counts = ds.Method.compute().where(dust_mask).to_series().dropna().value_counts()
print("\nMethod Distribution for DUST events:")
print(method_counts)

# Data availability for DUST events (Corrected logic)
print("\nData availability for DUST events (Corrected):")
pm25_missing = int((ds.PM25.isnull() & dust_mask).sum().compute())
ws_missing = int((ds.WS.isnull() & dust_mask).sum().compute())
print(f"Events with missing PM2.5: {pm25_missing} ({pm25_missing/total_dust_hours*100:.1f}%)")
print(f"Events with missing Wind Speed: {ws_missing} ({ws_missing/total_dust_hours*100:.1f}%)")

# PM10 stats for QC=1
qc1_mask = (ds.QC == 1) & dust_mask
if qc1_mask.any():
    pm10_qc1 = ds.PM10.compute().where(qc1_mask).to_series().dropna()
    print("\nPM10 Statistics for QC=1 (Low Confidence) events:")
    print(pm10_qc1.describe())
else:
    print("\nNo QC=1 events found.")

# Breakdown of Method for QC=1
method_qc1 = ds.Method.compute().where(qc1_mask).to_series().dropna().value_counts()
print("\nMethod Distribution for QC=1 events:")
print(method_qc1)
