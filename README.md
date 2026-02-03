# ASCOT 🍃⚡
**ARL Surface Wind-Blown Dust Detection Algorithm** 

ASCOT (Automated Surface Coordinate Observation Tracking) is a high-performance scientific pipeline for detecting wind-blown dust events using surface monitor data (EPA AQS and AirNow).

Architected by **Aero**, this version is optimized for the Pangeo ecosystem, supporting both eager execution (NumPy) and lazy, distributed computation (Dask).

## 🚀 Features
- **Pangeo-Native:** Built on `xarray`, `dask`, and `monetio`.
- **Backend Agnostic:** Seamlessly switch between local NumPy computation and distributed Dask clusters.
- **Strict Provenance:** Automatically tracks data transformations in metadata.
- **Latest Monetio:** Integrated with the `monetio` develop branch for improved data access.

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/ascot.git
cd ascot

# Install dependencies
pip install .
```

For development:
```bash
pip install -e ".[dev]"
pre-commit install
```

## 📖 Usage

### Command Line Interface
Run the dust detection algorithm on AirNow data:
```bash
python dust.py -s 2023-07-01 -e 2023-07-07 -d airnow -o dust_events.csv
```

To enable Dask chunking for large datasets:
```bash
python dust.py -s 2023-01-01 -e 2023-12-31 -d aqs -o annual_dust.nc --chunk
```

### Python API
```python
import xarray as xr
from dust import dust_algorithm

# Load your dataset
ds = xr.open_dataset("surface_obs.nc")

# Apply the algorithm (works lazily if ds is dask-backed)
ds_dust = dust_algorithm(ds)

# Results are only computed when needed
print(ds_dust.DUST.values)
```

## 🧪 Testing
We use `pytest` for validation. Our tests ensure consistent results between NumPy and Dask backends.
```bash
pytest
```

## 📜 Quality Control (QC) Flags
The algorithm provides confidence levels for detected dust:
- `0`: No dust detected.
- `1`: Low confidence.
- `2`: Medium confidence.
- `3`: High confidence (detected by multiple methods or high PM10 concentrations).

## 🛡 License
NOAA Air Resources Laboratory
