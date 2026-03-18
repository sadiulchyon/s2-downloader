# Sentinel-2 Explorer Phase 1

A lightweight Streamlit app for:
- uploading an AOI as GeoJSON
- searching Sentinel-2 L2A scenes from Earth Search
- previewing True Color, False Color, SWIR, NDVI, MNDWI, and BSI
- exporting AOI-clipped GeoTIFFs as a ZIP

## Files to upload to GitHub

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `.gitignore`
- `README.md`

## Quick start locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload all files from this folder.
3. Push to GitHub.
4. In Streamlit Community Cloud, create a new app from that repo.
5. Set the main file path to `app.py`.
6. Deploy.

## Current scope

This is the phase 1 MVP only.

Included:
- native Sentinel-2 preview
- AOI-based scene search
- multi-date export for native bands, composites, and indices

Not included yet:
- SEN2SR execution
- S2DR3 execution
- Google Drive export
- per-user auth
- non-GeoJSON AOI uploads

## Notes

- Very large AOIs may be slow.
- The app reads Sentinel-2 cloud-optimized GeoTIFF assets directly from STAC results.
- Exports are clipped to the uploaded AOI and zipped with a `manifest.csv`.
