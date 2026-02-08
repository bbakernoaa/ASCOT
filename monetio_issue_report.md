# Potential Issue with Monetio ISD-Lite Loader

While running the DustTrace pipeline for February 2023, I encountered issues with the `monetio.load("ish_lite", ...)` function.

## Issues Identified:
1. **Missing Files (404 Not Found)**: Several ISD-Lite station files for 2023 are missing from the NCEI server (e.g., `https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/2023/720984-99999-2023.gz`). Monetio's default reader does not gracefully handle these 404 errors, causing the entire batch fetch to fail.
2. **Metadata Mismatch in Dask**: The current implementation of `ISHLiteReader` uses `dd.from_delayed` which can fail if some partitions are empty (due to skipped files) and the metadata is not explicitly provided or correctly inferred.

## Applied Workaround in DustTrace:
I have implemented a robust fetching logic in `src/dust.py` (`add_met_to_airnow`) that:
- Iterates through the URLs and uses `dask.delayed` with a try-except block to skip failing URLs.
- Uses `pd.concat` on computed delayed objects to avoid metadata issues in Dask dataframes when handling potentially empty partitions.

## Recommendation:
The `monetio` package should consider adding an option to `load()` to ignore missing files or handle 404 errors without failing the entire request.
