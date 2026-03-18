from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

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
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
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


def _merge_shapes(geoms: List):
    if not geoms:
        raise ValueError("No valid geometry found in the uploaded AOI file.")
    merged = unary_union(geoms)
    if merged.is_empty:
        raise ValueError("Uploaded geometry is empty.")
    return mapping(merged)


def parse_geojson_bytes(raw_bytes: bytes) -> dict:
    payload = json.loads(raw_bytes.decode("utf-8"))
    if payload.get("type") == "FeatureCollection":
        geoms = [shape(feat["geometry"]) for feat in payload.get("features", []) if feat.get("geometry")]
        return _merge_shapes(geoms)
    if payload.get("type") == "Feature":
        return _merge_shapes([shape(payload["geometry"])])
    return _merge_shapes([shape(payload)])


def _parse_kml_coords(text: str) -> List[Tuple[float, float]]:
    coords = []
    for token in text.replace("\n", " ").replace("\t", " ").split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        lon = float(parts[0])
        lat = float(parts[1])
        coords.append((lon, lat))
    return coords


def parse_kml_bytes(raw_bytes: bytes) -> dict:
    root = ET.fromstring(raw_bytes)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    polygons = []

    for poly in root.findall(".//kml:Polygon", ns):
        outer_el = poly.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
        if outer_el is None or not (outer_el.text or "").strip():
            continue
        shell = _parse_kml_coords(outer_el.text or "")
        if len(shell) < 4:
            continue

        holes = []
        for inner_el in poly.findall(".//kml:innerBoundaryIs/kml:LinearRing/kml:coordinates", ns):
            inner_coords = _parse_kml_coords(inner_el.text or "")
            if len(inner_coords) >= 4:
                holes.append(inner_coords)

        polygon = Polygon(shell=shell, holes=holes)
        if not polygon.is_empty:
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if isinstance(polygon, Polygon):
                polygons.append(polygon)
            elif isinstance(polygon, MultiPolygon):
                polygons.extend(list(polygon.geoms))

    if not polygons:
        raise ValueError("No polygon geometry found in the uploaded KML. Use a polygon AOI in EPSG:4326 coordinates.")

    return _merge_shapes(polygons)


def parse_uploaded_aoi(uploaded_file) -> dict:
    raw_bytes = uploaded_file.getvalue()
    name = (uploaded_file.name or "").lower()
    if name.endswith(".kml"):
        return parse_kml_bytes(raw_bytes)
    if name.endswith(".geojson") or name.endswith(".json"):
        return parse_geojson_bytes(raw_bytes)

    try:
        return parse_geojson_bytes(raw_bytes)
    except Exception:
        return parse_kml_bytes(raw_bytes)


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
        max_items=300,
    )
    items = list(search.items())
    items.sort(key=lambda item: (str(item.datetime or date.min), item.id or ""))
    return [item.to_dict() for item in items]


def group_items_by_date(items: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = {}
    for item in items:
        dt = str(item.get("properties", {}).get("datetime", ""))[:10]
        grouped.setdefault(dt, []).append(item)

    rows = []
    for dt in sorted(grouped.keys()):
        group = grouped[dt]
        clouds = [it.get("properties", {}).get("eo:cloud_cover") for it in group]
        clouds = [c for c in clouds if c is not None]
        tiles = sorted({it.get("properties", {}).get("grid:code", "") for it in group if it.get("properties", {}).get("grid:code")})
        rows.append(
            {
                "date": dt,
                "items": group,
                "n_scenes": len(group),
                "tiles": tiles,
                "cloud_min": min(clouds) if clouds else None,
                "cloud_max": max(clouds) if clouds else None,
            }
        )
    return rows


def date_table(groups: List[dict]) -> pd.DataFrame:
    rows = []
    for g in groups:
        rows.append(
            {
                "date": g["date"],
                "n_scenes": g["n_scenes"],
                "cloud_min_%": g["cloud_min"],
                "cloud_max_%": g["cloud_max"],
                "tiles": ", ".join(g["tiles"]),
            }
        )
    return pd.DataFrame(rows)


def get_asset_href(item: dict, asset_key: str) -> str:
    return item["assets"][asset_key]["href"]


def get_base_profile(items_for_date: List[dict], geom_geojson: dict) -> Tuple[dict, np.ndarray]:
    href = get_asset_href(items_for_date[0], "red")
    with rasterio.open(href) as src:
        dst_geom = reproject_geom(geom_geojson, src.crs)
        arr, transform = mask(src, [dst_geom], crop=True, filled=False)
        base = arr[0].astype("float32")
        if np.ma.isMaskedArray(arr):
            base[arr.mask[0]] = np.nan
        elif src.nodata is not None:
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


def mosaic_band_on_base_grid(items_for_date: List[dict], asset_key: str, profile: dict, inside_mask: np.ndarray) -> np.ndarray:
    destination = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")

    for item in items_for_date:
        href = get_asset_href(item, asset_key)
        temp = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")
        with rasterio.open(href) as src:
            src_nodata = src.nodata
            if src_nodata is None:
                src_nodata = 0
            reproject(
                source=rasterio.band(src, 1),
                destination=temp,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=profile["transform"],
                dst_crs=profile["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        valid = np.isfinite(temp)
        destination[np.isnan(destination) & valid] = temp[np.isnan(destination) & valid]

    destination[~inside_mask] = np.nan
    return destination.astype("float32")


def load_date_arrays(items_for_date: List[dict], geom_geojson: dict, required_assets: List[str]) -> Tuple[dict, Dict[str, np.ndarray]]:
    profile, _ = get_base_profile(items_for_date, geom_geojson)
    inside_mask = polygon_mask_array(profile, geom_geojson)
    arrays = {}
    for asset_key in sorted(set(required_assets)):
        arrays[asset_key] = mosaic_band_on_base_grid(items_for_date, asset_key, profile, inside_mask)
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


def group_label(group: dict) -> str:
    cloud_min = group["cloud_min"]
    cloud_max = group["cloud_max"]
    if cloud_min is None or cloud_max is None:
        cloud_text = "cloud n/a"
    elif cloud_min == cloud_max:
        cloud_text = f"cloud {cloud_min}%"
    else:
        cloud_text = f"cloud {cloud_min}-{cloud_max}%"
    return f"{group['date']} | {group['n_scenes']} scene(s) | {cloud_text}"


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


def build_export_zip(selected_groups: List[dict], geom_geojson: dict, outputs: List[str]) -> bytes:
    required_assets = INDEX_ASSETS
    manifest_rows = []
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for group in selected_groups:
            profile, arrays = load_date_arrays(group["items"], geom_geojson, required_assets)
            date_str = group["date"]
            scene_prefix = f"exports/{date_str}"

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
                        "n_scenes": group["n_scenes"],
                        "tiles": ", ".join(group["tiles"]),
                        "output": output_name,
                        "type": kind,
                        "file": filename,
                    }
                )

        manifest = pd.DataFrame(manifest_rows)
        zf.writestr("exports/manifest.csv", manifest.to_csv(index=False))

    zip_buffer.seek(0)
    return zip_buffer.read()


def render_preview(group: dict, geom_geojson: dict, preview_choice: str):
    required_assets = INDEX_ASSETS
    profile, arrays = load_date_arrays(group["items"], geom_geojson, required_assets)

    if preview_choice in PREVIEW_ASSETS:
        stack = np.stack([arrays[key] for key in PREVIEW_ASSETS[preview_choice]], axis=0)
        img = percentile_stretch(stack)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.set_title(f"{preview_choice} | {group['date']}")
        ax.axis("off")
        return fig

    data = compute_index(arrays, preview_choice)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(data, cmap="viridis")
    ax.set_title(f"{preview_choice} | {group['date']}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def init_state():
    if "scene_items" not in st.session_state:
        st.session_state["scene_items"] = []
    if "aoi_geojson" not in st.session_state:
        st.session_state.aoi_geojson = None


init_state()

st.title("Sentinel-2 Explorer")
st.caption("Phase 1 MVP: upload AOI, search scenes, preview mosaicked dates, and export AOI-clipped GeoTIFFs.")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("AOI file", type=["geojson", "json", "kml"], help="Supported formats: GeoJSON, JSON, KML")
    start_date = st.date_input("Start date", value=date(2024, 1, 1))
    end_date = st.date_input("End date", value=date.today())
    max_cloud = st.slider("Max cloud cover (%)", min_value=0, max_value=100, value=20, step=5)
    search_clicked = st.button("Search scenes", type="primary", use_container_width=True)

if uploaded is not None:
    try:
        st.session_state.aoi_geojson = parse_uploaded_aoi(uploaded)
    except Exception as exc:
        st.error(f"AOI could not be read: {exc}")
        st.stop()

if search_clicked:
    if st.session_state.aoi_geojson is None:
        st.warning("Upload a GeoJSON or KML AOI first.")
    elif start_date > end_date:
        st.warning("Start date must be earlier than end date.")
    else:
        with st.spinner("Searching Sentinel-2 scenes..."):
            st.session_state["scene_items"] = search_scenes(
                json.dumps(st.session_state.aoi_geojson),
                start_date.isoformat(),
                end_date.isoformat(),
                max_cloud,
            )

items = st.session_state["scene_items"]
groups = group_items_by_date(items) if items else []

a, b = st.columns([1.2, 1])
with a:
    st.subheader("Dates")
    if groups:
        table = date_table(groups)
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("Run a search to populate available dates.")

with b:
    st.subheader("Preview")
    if groups:
        labels = [group_label(group) for group in groups]
        selected_label = st.selectbox("Choose a date", options=labels)
        selected_group = groups[labels.index(selected_label)]
        st.caption(f"Tiles in mosaic: {', '.join(selected_group['tiles']) if selected_group['tiles'] else 'n/a'}")
        preview_choice = st.radio(
            "Layer",
            options=["True Color", "False Color", "SWIR Composite", "NDVI", "MNDWI", "BSI"],
            horizontal=False,
        )
        with st.spinner("Rendering mosaicked preview..."):
            fig = render_preview(selected_group, st.session_state.aoi_geojson, preview_choice)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Preview appears after a scene search.")

st.subheader("Export selected dates")
if groups:
    export_labels = [group_label(group) for group in groups]
    selected_export_labels = st.multiselect(
        "Dates to export",
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
            st.warning("Choose at least one date and one output.")
        else:
            selected_groups = [groups[export_labels.index(lbl)] for lbl in selected_export_labels]
            with st.spinner("Building AOI-clipped GeoTIFF ZIP..."):
                zip_bytes = build_export_zip(selected_groups, st.session_state.aoi_geojson, export_outputs)
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
        - This phase 1 app accepts GeoJSON, JSON, and polygon KML files.
        - Dates are mosaicked from all matching Sentinel-2 scenes returned for that date.
        - Exports are native Sentinel-2 bands, indices, and simple composites.
        - SEN2SR and S2DR3 are intentionally excluded from phase 1 hosting.
        - Very large AOIs can be slow because the app resamples multiple COG bands on demand.
        """
    )
