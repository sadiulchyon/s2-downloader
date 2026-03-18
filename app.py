from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from pyproj import Transformer
from pystac_client import Client
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.warp import Resampling, reproject
from shapely.geometry import mapping, shape
from shapely.ops import unary_union, transform as shapely_transform

st.set_page_config(page_title="S2 Explorer", page_icon="🛰️", layout="wide")

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
PREVIEW_ASSETS = {
    "True Color": ["red", "green", "blue"],
    "False Color": ["nir", "red", "green"],
    "SWIR Composite": ["swir22", "nir", "red"],
}
INDEX_ASSETS = ["red", "green", "blue", "nir", "swir16", "swir22"]
EXPORT_OPTIONS = [
    "red",
    "green",
    "blue",
    "nir",
    "swir16",
    "swir22",
    "True Color",
    "False Color",
    "SWIR Composite",
    "NDVI",
    "MNDWI",
    "BSI",
]


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype="float32")
    valid = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[valid] = a[valid] / b[valid]
    return out.astype("float32")


def parse_geojson(uploaded_file) -> dict:
    payload = json.loads(uploaded_file.getvalue().decode("utf-8"))
    if payload.get("type") == "FeatureCollection":
        geoms = [shape(feat["geometry"]) for feat in payload.get("features", []) if feat.get("geometry")]
        if not geoms:
            raise ValueError("No valid geometry found in the uploaded GeoJSON.")
        merged = unary_union(geoms)
    elif payload.get("type") == "Feature":
        merged = shape(payload["geometry"])
    else:
        merged = shape(payload)

    if merged.is_empty:
        raise ValueError("Uploaded geometry is empty.")

    return mapping(merged)


def reproject_geom(geom_geojson: dict, dst_crs) -> dict:
    src_geom = shape(geom_geojson)
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    dst_geom = shapely_transform(transformer.transform, src_geom)
    return mapping(dst_geom)


@st.cache_data(show_spinner=False)
def search_scenes(geom_geojson_str: str, start_iso: str, end_iso: str, max_cloud: int) -> List[dict]:
    geom_geojson = json.loads(geom_geojson_str)
    client = Client.open(EARTH_SEARCH_URL)
    search = client.search(
        collections=[COLLECTION],
        intersects=geom_geojson,
        datetime=f"{start_iso}/{end_iso}",
        query={"eo:cloud_cover": {"lte": max_cloud}},
        max_items=200,
    )
    items = list(search.items())
    items.sort(key=lambda item: item.datetime or date.min)
    return [item.to_dict() for item in items]


def item_table(items: List[dict]) -> pd.DataFrame:
    rows = []
    for item in items:
        props = item.get("properties", {})
        rows.append(
            {
                "date": str(props.get("datetime", ""))[:10],
                "cloud_%": props.get("eo:cloud_cover"),
                "id": item.get("id"),
                "tile": props.get("grid:code", ""),
            }
        )
    return pd.DataFrame(rows)


def get_asset_href(item: dict, asset_key: str) -> str:
    return item["assets"][asset_key]["href"]


def get_base_profile(item: dict, geom_geojson: dict) -> Tuple[dict, np.ndarray]:
    href = get_asset_href(item, "red")
    with rasterio.open(href) as src:
        dst_geom = reproject_geom(geom_geojson, src.crs)
        arr, transform = mask(src, [dst_geom], crop=True, nodata=np.nan, filled=True)
        base = arr[0].astype("float32")
        if src.nodata is not None:
            base[base == src.nodata] = np.nan
        profile = {
            "height": base.shape[0],
            "width": base.shape[1],
            "transform": transform,
            "crs": src.crs,
            "dtype": "float32",
            "count": 1,
            "nodata": np.nan,
        }
        return profile, base


def polygon_mask_array(profile: dict, geom_geojson: dict) -> np.ndarray:
    dst_geom = reproject_geom(geom_geojson, profile["crs"])
    mask_arr = geometry_mask(
        [dst_geom],
        transform=profile["transform"],
        invert=True,
        out_shape=(profile["height"], profile["width"]),
    )
    return mask_arr


def read_band_on_base_grid(item: dict, asset_key: str, profile: dict, inside_mask: np.ndarray) -> np.ndarray:
    href = get_asset_href(item, asset_key)
    destination = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    destination[~inside_mask] = np.nan
    return destination.astype("float32")


def load_scene_arrays(item: dict, geom_geojson: dict, required_assets: List[str]) -> Tuple[dict, Dict[str, np.ndarray]]:
    profile, _ = get_base_profile(item, geom_geojson)
    inside_mask = polygon_mask_array(profile, geom_geojson)
    arrays = {}
    for asset_key in sorted(set(required_assets)):
        arrays[asset_key] = read_band_on_base_grid(item, asset_key, profile, inside_mask)
    return profile, arrays


def compute_index(arrays: Dict[str, np.ndarray], name: str) -> np.ndarray:
    red = arrays["red"]
    green = arrays["green"]
    blue = arrays["blue"]
    nir = arrays["nir"]
    swir16 = arrays["swir16"]

    if name == "NDVI":
        return _safe_divide(nir - red, nir + red)
    if name == "MNDWI":
        return _safe_divide(green - swir16, green + swir16)
    if name == "BSI":
        return _safe_divide((swir16 + red) - (nir + blue), (swir16 + red) + (nir + blue))
    raise ValueError(f"Unsupported index: {name}")


def percentile_stretch(stack: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    out = np.zeros_like(stack, dtype="float32")
    for i in range(stack.shape[0]):
        band = stack[i]
        valid = np.isfinite(band)
        if not valid.any():
            continue
        lo, hi = np.nanpercentile(band[valid], [p_low, p_high])
        if hi <= lo:
            out[i] = np.nan_to_num(band)
            continue
        scaled = (band - lo) / (hi - lo)
        out[i] = np.clip(np.nan_to_num(scaled, nan=0.0), 0, 1)
    return np.moveaxis(out, 0, -1)


def scene_label(item: dict) -> str:
    dt = str(item.get("properties", {}).get("datetime", ""))[:10]
    cloud = item.get("properties", {}).get("eo:cloud_cover", "?")
    return f"{dt} | cloud {cloud}% | {item.get('id', '')}"


def write_single_band_geotiff(buffer: io.BytesIO, arr: np.ndarray, profile: dict):
    with rasterio.open(
        buffer,
        "w",
        driver="GTiff",
        height=profile["height"],
        width=profile["width"],
        count=1,
        dtype="float32",
        crs=profile["crs"],
        transform=profile["transform"],
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(arr.astype("float32"), 1)


def write_three_band_geotiff(buffer: io.BytesIO, arr: np.ndarray, profile: dict):
    with rasterio.open(
        buffer,
        "w",
        driver="GTiff",
        height=profile["height"],
        width=profile["width"],
        count=3,
        dtype="float32",
        crs=profile["crs"],
        transform=profile["transform"],
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(arr.astype("float32"))


def build_export_zip(selected_items: List[dict], geom_geojson: dict, outputs: List[str]) -> bytes:
    required_assets = INDEX_ASSETS
    manifest_rows = []
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in selected_items:
            profile, arrays = load_scene_arrays(item, geom_geojson, required_assets)
            date_str = str(item.get("properties", {}).get("datetime", ""))[:10]
            scene_id = item.get("id", "scene")
            scene_prefix = f"exports/{date_str}_{scene_id}"

            for output_name in outputs:
                tif_buffer = io.BytesIO()
                if output_name in INDEX_ASSETS:
                    write_single_band_geotiff(tif_buffer, arrays[output_name], profile)
                    filename = f"{scene_prefix}/{output_name}.tif"
                    kind = "band"
                elif output_name in PREVIEW_ASSETS:
                    stack = np.stack([arrays[key] for key in PREVIEW_ASSETS[output_name]], axis=0)
                    write_three_band_geotiff(tif_buffer, stack, profile)
                    filename = f"{scene_prefix}/{output_name.lower().replace(' ', '_')}.tif"
                    kind = "composite"
                else:
                    index_arr = compute_index(arrays, output_name)
                    write_single_band_geotiff(tif_buffer, index_arr, profile)
                    filename = f"{scene_prefix}/{output_name.lower()}.tif"
                    kind = "index"

                zf.writestr(filename, tif_buffer.getvalue())
                manifest_rows.append(
                    {
                        "date": date_str,
                        "scene_id": scene_id,
                        "output": output_name,
                        "type": kind,
                        "file": filename,
                    }
                )

        manifest = pd.DataFrame(manifest_rows)
        zf.writestr("exports/manifest.csv", manifest.to_csv(index=False))

    zip_buffer.seek(0)
    return zip_buffer.read()


def render_preview(item: dict, geom_geojson: dict, preview_choice: str):
    required_assets = INDEX_ASSETS
    profile, arrays = load_scene_arrays(item, geom_geojson, required_assets)

    if preview_choice in PREVIEW_ASSETS:
        stack = np.stack([arrays[key] for key in PREVIEW_ASSETS[preview_choice]], axis=0)
        img = percentile_stretch(stack)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.set_title(preview_choice)
        ax.axis("off")
        return fig

    data = compute_index(arrays, preview_choice)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(data, cmap="viridis")
    ax.set_title(preview_choice)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def init_state():
    if "items" not in st.session_state:
        st.session_state.items = []
    if "aoi_geojson" not in st.session_state:
        st.session_state.aoi_geojson = None


init_state()

st.title("Sentinel-2 Explorer")
st.caption("Phase 1 MVP: upload AOI, search scenes, preview RGB or indices, and export AOI-clipped GeoTIFFs.")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("AOI GeoJSON", type=["geojson", "json"])
    start_date = st.date_input("Start date", value=date(2024, 1, 1))
    end_date = st.date_input("End date", value=date.today())
    max_cloud = st.slider("Max cloud cover (%)", min_value=0, max_value=100, value=20, step=5)
    search_clicked = st.button("Search scenes", type="primary", use_container_width=True)

if uploaded is not None:
    try:
        st.session_state.aoi_geojson = parse_geojson(uploaded)
    except Exception as exc:
        st.error(f"AOI could not be read: {exc}")
        st.stop()

if search_clicked:
    if st.session_state.aoi_geojson is None:
        st.warning("Upload a GeoJSON AOI first.")
    elif start_date > end_date:
        st.warning("Start date must be earlier than end date.")
    else:
        with st.spinner("Searching Sentinel-2 scenes..."):
            st.session_state.items = search_scenes(
                json.dumps(st.session_state.aoi_geojson),
                start_date.isoformat(),
                end_date.isoformat(),
                max_cloud,
            )

items = st.session_state.items

a, b = st.columns([1.2, 1])
with a:
    st.subheader("Scenes")
    if items:
        table = item_table(items)
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("Run a search to populate available scenes.")

with b:
    st.subheader("Preview")
    if items:
        labels = [scene_label(item) for item in items]
        selected_label = st.selectbox("Choose a scene", options=labels)
        selected_item = items[labels.index(selected_label)]
        preview_choice = st.radio(
            "Layer",
            options=["True Color", "False Color", "SWIR Composite", "NDVI", "MNDWI", "BSI"],
            horizontal=False,
        )
        with st.spinner("Rendering preview..."):
            fig = render_preview(selected_item, st.session_state.aoi_geojson, preview_choice)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Preview appears after a scene search.")

st.subheader("Export selected dates")
if items:
    export_labels = [scene_label(item) for item in items]
    selected_export_labels = st.multiselect(
        "Scenes to export",
        options=export_labels,
        default=export_labels[:1],
    )
    export_outputs = st.multiselect(
        "Bands / composites / indices",
        options=EXPORT_OPTIONS,
        default=["True Color", "NDVI", "MNDWI"],
    )
    if st.button("Build ZIP export", use_container_width=True):
        if not selected_export_labels or not export_outputs:
            st.warning("Choose at least one scene and one output.")
        else:
            selected_items = [items[export_labels.index(lbl)] for lbl in selected_export_labels]
            with st.spinner("Building AOI-clipped GeoTIFF ZIP..."):
                zip_bytes = build_export_zip(selected_items, st.session_state.aoi_geojson, export_outputs)
            st.download_button(
                label="Download export ZIP",
                data=zip_bytes,
                file_name="s2_phase1_exports.zip",
                mime="application/zip",
                use_container_width=True,
            )
else:
    st.info("Export options appear after a scene search.")

with st.expander("Current limitations"):
    st.markdown(
        """
        - This phase 1 app accepts GeoJSON only.
        - Exports are native Sentinel-2 bands, indices, and simple composites.
        - SEN2SR and S2DR3 are intentionally excluded from phase 1 hosting.
        - Very large AOIs can be slow because the app resamples multiple COG bands on demand.
        """
    )
