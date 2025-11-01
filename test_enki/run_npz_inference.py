import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf


HERE = Path(__file__).resolve()
DEFAULTS = {
    "npz_path": HERE.parents[3] / "kesif_alani.derivatives.npz",

    "model_path": HERE.parents[1] / "model_enki" / "enki0_0.keras",
    "output_dir": Path.cwd() / "enki_npz_inference",
    "tile_size": 350,
    "stride": 175,
    "batch_size": 32,
    "class_names": ["Negative", "Positive"],
    "normalize": True,
    "confidence_threshold": 0.99,
    "include_classes": ["Positive"],
    "min_margin": 0.1,
    "nms_iou_threshold": 0.3,
    "nms_max_detections": None,
}


def load_rgb_from_npz(npz_path: Path) -> Tuple[np.ndarray, Optional[dict]]:
    with np.load(npz_path, allow_pickle=True) as data:
        if "rgb" not in data.files:
            raise KeyError(f"'rgb' array not found in {npz_path}")
        rgb = data["rgb"]
        metadata = data["_metadata"].item() if "_metadata" in data.files else None
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected 'rgb' array with shape (3,H,W); got {rgb.shape}")
    rgb = np.transpose(rgb, (1, 2, 0))  # -> (H,W,3)
    return rgb.astype(np.float32), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Enki model inference on RGB data stored inside an NPZ file."
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=DEFAULTS["npz_path"],
        help="Path to the NPZ file (expects an array named 'rgb' in (3,H,W) format).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULTS["model_path"],
        help="Path to the Keras model to use for inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULTS["output_dir"],
        help="Directory where prediction CSV/JSON files will be written.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULTS["tile_size"],
        help="Square tile size that matches the model input.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULTS["stride"],
        help="Stride to slide the window over the source image.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULTS["batch_size"],
        help="Number of tiles evaluated together.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=DEFAULTS["class_names"],
        help="Ordered list of class names corresponding to the model output indices.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Divide tiles by 255.0 before inference. Enable if your training pipeline used normalized inputs.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULTS["confidence_threshold"],
        help="Threshold applied to the highest class probability when building the filtered JSON output.",
    )
    parser.add_argument(
        "--include-classes",
        nargs="+",
        default=DEFAULTS["include_classes"],
        help="Class names to include in vector outputs (use 'ALL' to keep everything).",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULTS["min_margin"],
        help="Minimum gap between the best and second-best class probabilities.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=DEFAULTS["nms_iou_threshold"],
        help="IoU threshold for non-max suppression (set <=0 to disable).",
    )
    parser.add_argument(
        "--nms-max-detections",
        type=int,
        default=DEFAULTS["nms_max_detections"],
        help="Maximum number of detections to keep after NMS (default: unlimited).",
    )
    parser.add_argument(
        "--reference-raster",
        type=Path,
        help="Optional raster to read georeferencing from. Defaults to the path stored in NPZ metadata if available.",
    )
    parser.add_argument(
        "--gpkg-path",
        type=Path,
        help="If provided, export detections to this GeoPackage. Defaults to <output_dir>/<stem>_detections.gpkg when georeferencing is available.",
    )
    parser.add_argument(
        "--skip-vector",
        action="store_true",
        help="Skip GeoPackage export even if georeferencing information is available.",
    )
    return parser.parse_args()


@tf.keras.utils.register_keras_serializable(package="compat")
class CompatInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, batch_shape=None, **kwargs):
        if batch_shape is not None and "batch_input_shape" not in kwargs:
            kwargs["batch_input_shape"] = tuple(batch_shape)
        super().__init__(*args, **kwargs)


def load_model_compat(model_path: Path) -> tf.keras.Model:
    try:
        import keras

        return keras.models.load_model(model_path)
    except Exception:
        pass
    load_attempts = [
        {},
        {"compile": False, "safe_mode": False},
        {"compile": False, "custom_objects": {"InputLayer": CompatInputLayer}},
        {"custom_objects": {"InputLayer": CompatInputLayer}},
        {"compile": False, "safe_mode": False, "custom_objects": {"InputLayer": CompatInputLayer}},
    ]

    last_error: Exception | None = None
    for params in load_attempts:
        try:
            return tf.keras.models.load_model(model_path, **params)
        except Exception as exc:
            last_error = exc
            continue
    try:
        return load_model_from_archive(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {model_path}. "
            "Ensure the file exists and is compatible with your TensorFlow/Keras version. "
            f"Last error: {last_error}; archive error: {exc}"
        ) from exc


def load_model_from_archive(model_path: Path) -> tf.keras.Model:
    with zipfile.ZipFile(model_path) as archive:
        config = json.loads(archive.read("config.json").decode("utf-8"))
        fix_batch_shape(config)
        model_json = json.dumps(config)
        model = tf.keras.models.model_from_json(model_json)
        with archive.open("model.weights.h5") as weights_file:
            with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
                tmp.write(weights_file.read())
                tmp.flush()
                model.load_weights(tmp.name)
    return model


def fix_batch_shape(node):
    if isinstance(node, dict):
        if node.get("class_name") == "InputLayer":
            cfg = node.setdefault("config", {})
            if "batch_shape" in cfg and "batch_input_shape" not in cfg:
                cfg["batch_input_shape"] = cfg.pop("batch_shape")
        dtype_value = node.get("dtype")
        if isinstance(dtype_value, dict) and dtype_value.get("class_name") == "DTypePolicy":
            node["dtype"] = dtype_value.get("config", {}).get("name", "float32")
        cfg = node.get("config")
        if isinstance(cfg, dict):
            dtype_value = cfg.get("dtype")
            if isinstance(dtype_value, dict) and dtype_value.get("class_name") == "DTypePolicy":
                cfg["dtype"] = dtype_value.get("config", {}).get("name", "float32")
        for value in node.values():
            fix_batch_shape(value)
    elif isinstance(node, list):
        for item in node:
            fix_batch_shape(item)


def compute_positions(length: int, tile: int, stride: int) -> List[int]:
    if length < tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    last_start = length - tile
    if positions[-1] != last_start:
        positions.append(last_start)
    return positions


def iter_tiles(
    image: np.ndarray, tile_size: int, stride: int
) -> Iterable[Tuple[Tuple[int, int], np.ndarray]]:
    height, width, _ = image.shape
    row_positions = compute_positions(height, tile_size, stride)
    col_positions = compute_positions(width, tile_size, stride)
    for top in row_positions:
        for left in col_positions:
            patch = image[top : top + tile_size, left : left + tile_size]
            if patch.shape[0] != tile_size or patch.shape[1] != tile_size:
                continue
            yield (top, left), patch


def batched_predictions(
    model: tf.keras.Model,
    tile_iter: Iterable[Tuple[Tuple[int, int], np.ndarray]],
    batch_size: int,
    normalize: bool,
) -> Iterable[Tuple[Tuple[int, int], np.ndarray]]:
    batch_meta: List[Tuple[int, int]] = []
    batch_tiles: List[np.ndarray] = []

    for meta, tile in tile_iter:
        batch_meta.append(meta)
        batch_tiles.append(tile)
        if len(batch_tiles) == batch_size:
            batch = prepare_batch(batch_tiles, normalize)
            preds = model.predict(batch, verbose=0)
            for meta_item, probs in zip(batch_meta, preds):
                yield meta_item, probs
            batch_meta.clear()
            batch_tiles.clear()

    if batch_tiles:
        batch = prepare_batch(batch_tiles, normalize)
        preds = model.predict(batch, verbose=0)
        for meta_item, probs in zip(batch_meta, preds):
            yield meta_item, probs


def prepare_batch(tiles: Sequence[np.ndarray], normalize: bool) -> np.ndarray:
    batch = np.stack(tiles, axis=0).astype(np.float32)
    if normalize:
        batch /= 255.0
    return batch


def build_outputs(
    predictions: Iterable[Tuple[Tuple[int, int], np.ndarray]],
    class_names: Sequence[str],
    tile_size: int,
) -> pd.DataFrame:
    records = []
    for (top, left), probs in predictions:
        best_idx = int(np.argmax(probs))
        best_score = float(probs[best_idx])
        center_row = top + tile_size / 2.0
        center_col = left + tile_size / 2.0
        row = {
            "top": top,
            "left": left,
            "bottom": top + tile_size,
            "right": left + tile_size,
            "center_row": center_row,
            "center_col": center_col,
            "predicted_class": class_names[best_idx],
            "confidence": best_score,
        }
        for idx, name in enumerate(class_names):
            row[f"prob_{name}"] = float(probs[idx])
        records.append(row)
    df = pd.DataFrame.from_records(records)
    return df


def compute_margins(df: pd.DataFrame) -> pd.Series:
    prob_columns = [c for c in df.columns if c.startswith("prob_")]
    if len(prob_columns) < 2:
        return pd.Series(np.zeros(len(df)), index=df.index)

    probs = df[prob_columns].to_numpy(dtype=np.float32)
    top = probs.max(axis=1)
    if probs.shape[1] == 1:
        second = np.zeros_like(top)
    else:
        second = np.partition(probs, -2, axis=1)[:, -2]
    return pd.Series(top - second, index=df.index)


def apply_nms(
    detections: pd.DataFrame,
    iou_threshold: float,
    max_detections: Optional[int],
) -> pd.DataFrame:
    if detections.empty or iou_threshold is None or iou_threshold <= 0:
        return detections

    boxes = detections[["top", "left", "bottom", "right"]].to_numpy(dtype=np.float32)
    scores = detections["confidence"].to_numpy(dtype=np.float32)
    order = scores.argsort()[::-1]

    keep_indices: List[int] = []
    suppressed = np.zeros(len(detections), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep_indices.append(idx)
        if max_detections is not None and len(keep_indices) >= max_detections:
            break

        top1, left1, bottom1, right1 = boxes[idx]
        area1 = max(0.0, bottom1 - top1) * max(0.0, right1 - left1)

        for jdx in order:
            if suppressed[jdx] or jdx == idx:
                continue

            top2, left2, bottom2, right2 = boxes[jdx]
            inter_top = max(top1, top2)
            inter_left = max(left1, left2)
            inter_bottom = min(bottom1, bottom2)
            inter_right = min(right1, right2)
            inter_h = max(0.0, inter_bottom - inter_top)
            inter_w = max(0.0, inter_right - inter_left)
            inter_area = inter_h * inter_w

            if inter_area <= 0:
                continue

            area2 = max(0.0, bottom2 - top2) * max(0.0, right2 - left2)
            union = area1 + area2 - inter_area
            iou = inter_area / union if union > 0 else 0.0

            if iou > iou_threshold:
                suppressed[jdx] = True

    return detections.iloc[keep_indices].copy()


def resolve_reference_raster(
    npz_path: Path, metadata: Optional[dict], override: Optional[Path]
) -> Optional[Path]:
    candidates: List[Path] = []
    if override is not None:
        candidates.append(override)
    if metadata:
        meta_path = metadata.get("input_path")
        if isinstance(meta_path, str) and meta_path:
            candidates.append(Path(meta_path))
            candidates.append(npz_path.parent / Path(meta_path).name)
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def load_georeferencing(raster_path: Path) -> Tuple[Optional["Affine"], Optional["CRS"]]:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required for GeoPackage export.") from exc

    with rasterio.open(raster_path) as src:
        return src.transform, src.crs


def export_to_geopackage(
    detections: pd.DataFrame,
    transform,
    crs,
    gpkg_path: Path,
) -> None:
    if detections.empty:
        print("No detections above threshold; skipping GeoPackage export.")
        return

    try:
        import geopandas as gpd
        from shapely.geometry import Point, box
    except ImportError as exc:
        raise RuntimeError(
            "geopandas and shapely are required for GeoPackage export."
        ) from exc

    import rasterio.transform

    polygons = []
    points = []
    center_xs = []
    center_ys = []

    for row in detections.itertuples(index=False):
        top = float(row.top)
        left = float(row.left)
        bottom = float(row.bottom)
        right = float(row.right)

        minx, maxy = rasterio.transform.xy(transform, top, left, offset="ul")
        maxx, miny = rasterio.transform.xy(transform, bottom - 1, right - 1, offset="lr")

        geom = box(minx, miny, maxx, maxy)
        polygons.append(geom)

        center_col = left + (right - left) / 2.0
        center_row = top + (bottom - top) / 2.0
        center_x, center_y = transform * (center_col, center_row)
        center_xs.append(center_x)
        center_ys.append(center_y)
        points.append(Point(center_x, center_y))

    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    polygon_df = detections.copy()
    polygon_df["center_x"] = center_xs
    polygon_df["center_y"] = center_ys
    gdf_polygons = gpd.GeoDataFrame(polygon_df, geometry=polygons, crs=crs)
    gdf_polygons.to_file(gpkg_path, layer="detections_tiles", driver="GPKG")

    gdf_points = gpd.GeoDataFrame(
        polygon_df.drop(columns="geometry", errors="ignore"),
        geometry=points,
        crs=crs,
    )
    gdf_points.to_file(
        gpkg_path,
        layer="detections_points",
        driver="GPKG",
        mode="a",
    )

    print(f"GeoPackage saved to {gpkg_path}")


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb, metadata = load_rgb_from_npz(args.npz_path)

    model = load_model_compat(args.model_path)

    tiles = iter_tiles(rgb, args.tile_size, args.stride)
    predictions = list(
        batched_predictions(model, tiles, args.batch_size, args.normalize)
    )

    df = build_outputs(predictions, args.class_names, args.tile_size)
    csv_path = output_dir / (args.npz_path.stem + "_predictions.csv")
    df.to_csv(csv_path, index=False)

    threshold = args.confidence_threshold
    high_conf_mask = df["confidence"] >= threshold
    include_classes = args.include_classes
    if len(include_classes) == 1 and include_classes[0].upper() == "ALL":
        include_classes = df["predicted_class"].unique().tolist()

    mask_class = high_conf_mask & df["predicted_class"].isin(include_classes)
    after_class = int(mask_class.sum())
    detections = df[mask_class].copy()
    detections["margin"] = compute_margins(detections)
    detections = detections[detections["margin"] >= args.min_margin].copy()
    after_margin = len(detections)
    detections = apply_nms(
        detections,
        args.nms_iou_threshold,
        args.nms_max_detections,
    )
    after_nms = len(detections)
    detections_path = (
        output_dir / f"{args.npz_path.stem}_detections_conf_{threshold:.2f}.json"
    )
    detections_path.write_text(
        json.dumps(detections.to_dict(orient="records"), indent=2)
    )

    summary_path = output_dir / (args.npz_path.stem + "_summary.json")
    summary_payload = {
        "npz_path": str(args.npz_path.resolve()),
        "model_path": str(args.model_path.resolve()),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "num_tiles": len(df),
        "class_names": args.class_names,
        "confidence_threshold": threshold,
        "include_classes": include_classes,
        "min_margin": args.min_margin,
        "nms_iou_threshold": args.nms_iou_threshold,
        "nms_max_detections": args.nms_max_detections,
        "num_after_confidence": int(high_conf_mask.sum()),
        "num_after_class": after_class,
        "num_after_margin": int(after_margin),
        "num_after_nms": int(after_nms),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print(f"Predictions saved to {csv_path}")
    print(f"Summary saved to {summary_path}")

    if not args.skip_vector:
        reference_raster = resolve_reference_raster(
            args.npz_path, metadata, args.reference_raster
        )
        if reference_raster is None:
            print(
                "Reference raster not found; skipping GeoPackage export. "
                "Provide --reference-raster to enable vector output."
            )
        else:
            try:
                transform, crs = load_georeferencing(reference_raster)
                gpkg_path = (
                    args.gpkg_path
                    if args.gpkg_path is not None
                    else output_dir
                    / f"{args.npz_path.stem}_detections_conf_{threshold:.2f}.gpkg"
                )
                export_to_geopackage(detections, transform, crs, gpkg_path)
            except Exception as exc:
                print(f"GeoPackage export failed: {exc}")


if __name__ == "__main__":
    main()
