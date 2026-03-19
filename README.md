# Sentinel-2 Explorer Phase 1

A lightweight Streamlit app for:
- uploading an AOI as GeoJSON
- searching Sentinel-2 L2A scenes from Earth Search
- previewing True Color, False Color, SWIR, NDVI, MNDWI, and BSI
- exporting AOI-clipped GeoTIFFs as a ZIP

## Quick start locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Very large AOIs may be slow.
- The app reads Sentinel-2 cloud-optimized GeoTIFF assets directly from STAC results.
- Exports are clipped to the uploaded AOI and zipped with a `manifest.csv`.
