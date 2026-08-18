"""Mesh export helpers for NeuroFlux STL generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MeshProfile:
    name: str
    sigma: float
    taubin: int


@dataclass(frozen=True)
class StlExportOptions:
    sigma: float = 0.0
    taubin: int = 0
    max_faces: int = 300_000
    hollow: bool = False
    wall_mm: float = 5.0
    sulci_enhance: float = 0.4
    hc_iter: int = 10
    flat_base: bool = False
    upsample: int = 1
    combined_only: bool = True
    combined: bool = True
    scale: float = 1.0
    label_schema: str = "full"
    surface_mode: str = "sdf"
    combined_mode: str = "external_shell"
    nozzle_mm: float = 0.4
    layer_height_mm: float = 0.2


_PROFILES = {
    "csf": MeshProfile("csf", sigma=0.8, taubin=12),
    "gm": MeshProfile("gm", sigma=0.35, taubin=4),
    "wm": MeshProfile("wm", sigma=0.75, taubin=10),
    "deep_gm": MeshProfile("deep_gm", sigma=0.65, taubin=8),
    "brainstem": MeshProfile("brainstem", sigma=0.8, taubin=12),
    "cerebellum": MeshProfile("cerebellum", sigma=0.4, taubin=4),
    "bone": MeshProfile("bone", sigma=0.25, taubin=3),
    "unknown": MeshProfile("unknown", sigma=0.6, taubin=8),
}

_FULL_LABELS = {
    1: "csf",
    2: "gm",
    3: "wm",
    4: "deep_gm",
    5: "brainstem",
    6: "cerebellum",
}

_HEMI_LABELS = {
    1: "csf",
    2: "gm",
    3: "gm",
    4: "wm",
    5: "wm",
    6: "wm",
    7: "deep_gm",
    8: "deep_gm",
    9: "brainstem",
    10: "cerebellum",
}

_FS_CSF = {4, 5, 14, 15, 24, 31, 43, 44, 63, 72}
_FS_GM = {3, 42}
_FS_WM = {2, 41, 77, 78, 79, 251, 252, 253, 254, 255}
_FS_DEEP_GM = {
    10, 11, 12, 13, 17, 18, 26, 28, 30,
    49, 50, 51, 52, 53, 54, 58, 60, 62,
}
_FS_BRAINSTEM = {16, 170, 173, 174, 175, 178}
_FS_CEREBELLUM = {6, 7, 8, 45, 46, 47}


def normalize_label_schema(label_schema: str | None) -> str:
    """Return the canonical segmentation label schema name."""
    schema = (label_schema or "full").strip().lower()
    aliases = {
        "seg_full": "full",
        "whole": "full",
        "whole_brain": "full",
        "seg_hemi": "hemi",
        "hemisphere": "hemi",
        "hemi": "hemi",
        "seg_fs": "fs",
        "fs": "fs",
        "freesurfer": "fs",
        "ct_bone": "ct",
        "bone": "ct",
        "ct": "ct",
    }
    schema = aliases.get(schema, schema)
    return schema if schema in {"full", "hemi", "fs", "ct"} else "full"


def normalize_surface_mode(surface_mode: str | None) -> str:
    """Return the canonical surface extraction mode."""
    mode = (surface_mode or "sdf").strip().lower()
    aliases = {
        "distance": "sdf",
        "distance_field": "sdf",
        "signed_distance": "sdf",
        "legacy": "volume",
        "gaussian": "volume",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in {"sdf", "volume"} else "sdf"


def normalize_combined_mode(combined_mode: str | None) -> str:
    """Return the canonical strategy for combined STL exports."""
    mode = (combined_mode or "external_shell").strip().lower()
    aliases = {
        "shell": "external_shell",
        "union": "external_shell",
        "external": "external_shell",
        "multi": "multipart",
        "multi_part": "multipart",
        "per_tissue": "multipart",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in {"external_shell", "multipart"} else "external_shell"


def resolve_mesh_profile(label: int, label_schema: str | None = "full") -> MeshProfile:
    """Resolve a label to the smoothing profile used for mesh extraction."""
    label = int(label)
    schema = normalize_label_schema(label_schema)

    if schema == "full":
        profile_key = _FULL_LABELS.get(label, "unknown")
    elif schema == "hemi":
        profile_key = _HEMI_LABELS.get(label, "unknown")
    elif schema == "ct":
        profile_key = "bone" if label > 0 else "unknown"
    else:
        if label in _FS_CSF:
            profile_key = "csf"
        elif label in _FS_GM:
            profile_key = "gm"
        elif label in _FS_WM:
            profile_key = "wm"
        elif label in _FS_DEEP_GM:
            profile_key = "deep_gm"
        elif label in _FS_BRAINSTEM:
            profile_key = "brainstem"
        elif label in _FS_CEREBELLUM:
            profile_key = "cerebellum"
        else:
            profile_key = "unknown"

    return _PROFILES[profile_key]


def combine_mesh_profiles(profiles: Iterable[MeshProfile]) -> MeshProfile:
    """Choose a detail-preserving combined profile from selected labels."""
    profiles = list(profiles)
    if not profiles:
        return _PROFILES["unknown"]
    return MeshProfile(
        name="combined",
        sigma=min(p.sigma for p in profiles),
        taubin=min(p.taubin for p in profiles),
    )


def stl_options_from_body(body: dict) -> StlExportOptions:
    """Parse and clamp STL export options from an API request body."""
    return StlExportOptions(
        sigma=float(body.get("stl_sigma", 0.0)),
        taubin=int(body.get("stl_taubin", 0)),
        max_faces=max(1_000, int(body.get("stl_max_faces", 300_000))),
        hollow=bool(body.get("stl_hollow", False)),
        wall_mm=max(0.1, float(body.get("stl_wall_mm", 5.0))),
        sulci_enhance=max(0.0, float(body.get("stl_sulci_enhance", 0.4))),
        hc_iter=max(0, int(body.get("stl_hc_iter", 10))),
        flat_base=bool(body.get("stl_flat_base", False)),
        upsample=max(1, min(3, int(body.get("stl_upsample", 1)))),
        combined_only=bool(body.get("stl_combined_only", body.get("combined_only", True))),
        combined=bool(body.get("stl_combined", True)),
        scale=max(0.1, min(10.0, float(body.get("stl_scale", 1.0)))),
        label_schema=normalize_label_schema(body.get("label_schema", "full")),
        surface_mode=normalize_surface_mode(body.get("stl_surface_mode", "sdf")),
        combined_mode=normalize_combined_mode(body.get("stl_combined_mode", "external_shell")),
        nozzle_mm=max(0.1, min(2.0, float(body.get("stl_nozzle_mm", 0.4)))),
        layer_height_mm=max(0.02, min(1.0, float(body.get("stl_layer_height_mm", 0.2)))),
    )


def mesh_quality_report(mesh) -> dict:
    """Return JSON-serialisable mesh quality metrics for export diagnostics."""
    report = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "surface_area_mm2": round(float(mesh.area), 3),
        "volume_mm3": round(float(mesh.volume), 3) if mesh.is_watertight else None,
    }
    try:
        parts = mesh.split(only_watertight=False)
        report["components"] = int(len(parts))
    except Exception:
        report["components"] = None
    try:
        report["bounds_mm"] = [[round(float(v), 3) for v in row] for row in mesh.bounds]
        report["extents_mm"] = [round(float(v), 3) for v in mesh.extents]
    except Exception:
        report["bounds_mm"] = None
        report["extents_mm"] = None
    try:
        edge_lengths = np.asarray(mesh.edges_unique_length, dtype=np.float64)
        if edge_lengths.size:
            report["edge_mm"] = {
                "min": round(float(edge_lengths.min()), 4),
                "median": round(float(np.median(edge_lengths)), 4),
                "p95": round(float(np.percentile(edge_lengths, 95)), 4),
            }
    except Exception:
        report["edge_mm"] = None
    return report


def iter_stl_export(
    seg_arr: np.ndarray,
    affine: np.ndarray,
    label_items: list[tuple[str, int]],
    out_dir: str,
    timestamp: str,
    excluded_components: set[int] | None = None,
    options: StlExportOptions | None = None,
):
    """Yield progress dictionaries while generating STL meshes."""
    try:
        import trimesh as _trimesh
        from scipy.ndimage import distance_transform_edt, gaussian_filter
        from scipy.ndimage import label as nd_label
        from skimage.measure import marching_cubes
        from trimesh import smoothing as tri_smooth
    except ImportError as exc:
        yield {"status": "error", "error": f"Missing dependency: {exc}"}
        return

    options = options or StlExportOptions()
    excluded_components = excluded_components or set()
    label_schema = normalize_label_schema(options.label_schema)
    surface_mode = normalize_surface_mode(options.surface_mode)
    combined_mode = normalize_combined_mode(options.combined_mode)

    vox2mm = affine[:3, :3]
    origin = affine[:3, 3]
    voxel_sizes = np.abs(np.linalg.norm(vox2mm, axis=0))
    positive_voxels = voxel_sizes[voxel_sizes > 0]
    mean_voxel_size = float(np.mean(positive_voxels)) if positive_voxels.size else 1.0

    yield {"pct": 3, "msg": "Preparing mesh pipeline..."}

    component_labels = None
    if excluded_components:
        selected_mask = np.zeros_like(seg_arr, dtype=np.uint8)
        for _, label in label_items:
            selected_mask[seg_arr == int(label)] = 1
        component_labels, _ = nd_label(selected_mask)

    def _mask_for_label(label: int):
        mask = seg_arr == int(label)
        if component_labels is not None:
            mask = mask.copy()
            for component_id in excluded_components:
                mask[component_labels == component_id] = False
        return mask

    def _effective_profile(label: int | None, name: str, base_profile: MeshProfile) -> dict:
        sigma = options.sigma if options.sigma > 0 else base_profile.sigma
        taubin = options.taubin if options.taubin > 0 else base_profile.taubin
        return {
            "label": label,
            "name": name,
            "sigma": sigma,
            "taubin": taubin,
            "profile": base_profile.name,
        }

    def _make_mesh_gen(binary_mask, tissue_params):
        from scipy.ndimage import binary_erosion, generate_binary_structure

        name = tissue_params.get("name", "tissue")
        voxel_count = int(binary_mask.sum())
        if voxel_count < 100:
            return None

        mask = binary_mask.astype(np.float32)

        hollow_inner = None
        if options.hollow:
            struct = generate_binary_structure(3, 1)
            erode_r = max(1, int(round(options.wall_mm / mean_voxel_size)))
            eroded = binary_erosion(
                mask > 0,
                structure=struct,
                iterations=erode_r,
                border_value=0,
            )
            if eroded.sum() > 100:
                inner_sm = gaussian_filter(
                    eroded.astype(np.float32),
                    sigma=max(0.25, tissue_params.get("sigma", 0.8)),
                )
                if inner_sm.max() >= 0.3:
                    try:
                        inner_verts, inner_faces, _, _ = marching_cubes(inner_sm, level=0.5)
                        inner_mm = (inner_verts @ vox2mm.T) + origin
                        hollow_inner = _trimesh.Trimesh(
                            vertices=inner_mm,
                            faces=inner_faces,
                            process=True,
                        )
                        hollow_inner.invert()
                    except Exception:
                        hollow_inner = None

        upsample = options.upsample
        if upsample > 1:
            from scipy.ndimage import zoom as nd_zoom

            yield 5, f"{name} - upsampling x{upsample}..."
            mask = nd_zoom(mask, upsample, order=1)

        raw_sigma = max(0.0, float(tissue_params.get("sigma", 0.8)))
        sampling_mm = [
            float(voxel_sizes[i]) / upsample if voxel_sizes[i] > 0 else 1.0 / upsample
            for i in range(3)
        ]

        def _sigma_axes(sigma_mm: float):
            return [max(0.01, float(sigma_mm) / max(0.01, s)) for s in sampling_mm]

        if surface_mode == "sdf":
            yield 8, f"{name} - signed distance field..."
            mask_bool = mask >= 0.5
            if not mask_bool.any() or mask_bool.all():
                return None
            inside = distance_transform_edt(mask_bool, sampling=sampling_mm)
            outside = distance_transform_edt(~mask_bool, sampling=sampling_mm)
            surface_field = (inside - outside).astype(np.float32, copy=False)
            if raw_sigma > 0:
                surface_field = gaussian_filter(surface_field, sigma=_sigma_axes(raw_sigma))

            if options.sulci_enhance > 0:
                coarse_sigma = _sigma_axes(max(raw_sigma * 2.5, mean_voxel_size))
                coarse = gaussian_filter(surface_field, sigma=coarse_sigma)
                surface_field = surface_field + options.sulci_enhance * 0.35 * (
                    surface_field - coarse
                )
            if surface_field.max() <= 0 or surface_field.min() >= 0:
                return None
            surface_level = 0.0
        else:
            yield 8, f"{name} - volume surface field..."
            sigma_ax = _sigma_axes(max(0.15, raw_sigma))
            surface_field = gaussian_filter(mask, sigma=sigma_ax)
            if surface_field.max() < 0.3:
                return None

            if options.sulci_enhance > 0:
                coarse_sigma = [s * 2.5 for s in sigma_ax]
                coarse = gaussian_filter(surface_field, sigma=coarse_sigma)
                surface_field = np.clip(
                    surface_field + options.sulci_enhance * (surface_field - coarse),
                    0.0,
                    1.0,
                )
                if surface_field.max() < 0.3:
                    return None
            surface_level = 0.5

        yield 30, f"{name} - marching cubes..."
        try:
            verts, faces_mc, _, _ = marching_cubes(surface_field, level=surface_level)
        except Exception:
            return None

        verts_mm = (verts @ (vox2mm / upsample).T) + origin
        mesh = _trimesh.Trimesh(vertices=verts_mm, faces=faces_mc, process=True)
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            largest_faces = max(len(part.faces) for part in parts)
            min_faces = max(24, int(largest_faces * 0.01))
            kept_parts = [part for part in parts if len(part.faces) >= min_faces]
            if kept_parts:
                mesh = (
                    _trimesh.util.concatenate(kept_parts)
                    if len(kept_parts) > 1 else kept_parts[0]
                )

        raw_taubin = int(tissue_params.get("taubin", 0))
        tau_iter = 0 if raw_taubin <= 0 else max(1, min(50, raw_taubin))
        if tau_iter > 0:
            yield 52, f"{name} - taubin smoothing ({tau_iter} iter)..."
            try:
                tri_smooth.filter_taubin(mesh, lamb=0.45, nu=0.48, iterations=tau_iter)
            except Exception:
                pass

        if options.hc_iter > 0:
            yield 60, f"{name} - HC smoothing ({options.hc_iter} iter)..."
            try:
                tri_smooth.filter_humphrey(
                    mesh,
                    alpha=0.08,
                    beta=0.55,
                    iterations=options.hc_iter,
                )
            except Exception:
                pass

        yield 68, f"{name} - mesh repair..."
        try:
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            _trimesh.repair.fix_normals(mesh)
        except Exception:
            pass
        for _ in range(3):
            if mesh.is_watertight:
                break
            try:
                _trimesh.repair.fill_holes(mesh)
            except Exception:
                break
        try:
            _trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

        if options.hollow and hollow_inner is not None:
            try:
                _trimesh.repair.fix_normals(hollow_inner)
                mesh = _trimesh.util.concatenate([mesh, hollow_inner])
                _trimesh.repair.fix_normals(mesh)
            except Exception:
                pass

        yield 78, f"{name} - decimation ({len(mesh.faces):,} faces)..."
        if len(mesh.faces) > options.max_faces:
            decimated = False
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=options.max_faces)
                decimated = True
            except Exception:
                pass
            if not decimated:
                try:
                    pitch = float(np.max(mesh.extents)) / ((options.max_faces / 2) ** 0.5)
                    mesh = mesh.simplify_vertex_clustering(pitch)
                    if len(mesh.faces) > options.max_faces * 1.5:
                        mesh = mesh.simplify_quadric_decimation(face_count=options.max_faces)
                except Exception:
                    pass

        if options.flat_base and len(mesh.vertices) > 3:
            try:
                verts = mesh.vertices.copy()
                z_5th = float(np.percentile(verts[:, 2], 5))
                verts[verts[:, 2] < z_5th, 2] = z_5th
                mesh.vertices = verts
                try:
                    mesh.remove_degenerate_faces()
                    _trimesh.repair.fix_normals(mesh)
                except Exception:
                    pass
            except Exception:
                pass

        if abs(options.scale - 1.0) > 0.01:
            mesh.vertices *= options.scale

        yield 94, f"{name} - {len(mesh.faces):,} faces"
        return mesh

    def _drive(binary_mask, tissue_params, start_pct, end_pct):
        span = max(1, end_pct - start_pct)
        gen = _make_mesh_gen(binary_mask, tissue_params)
        mesh = None
        try:
            while True:
                local_pct, msg = next(gen)
                yield start_pct + int(local_pct / 100 * span), msg
        except StopIteration as exc:
            mesh = exc.value
        return mesh

    saved = []
    reports = {}
    generated_meshes = {}
    profiles = [resolve_mesh_profile(label, label_schema) for _, label in label_items]

    if not options.combined_only:
        total = len(label_items)
        for index, (key, label) in enumerate(label_items):
            mask = _mask_for_label(label)
            if mask.sum() == 0:
                continue

            base_profile = resolve_mesh_profile(label, label_schema)
            params = _effective_profile(label, key, base_profile)
            start_pct = 10 + int(index / (total + 1) * 70)
            end_pct = 10 + int((index + 1) / (total + 1) * 70)
            driver = _drive(mask, params, start_pct, end_pct)
            try:
                while True:
                    pct, msg = next(driver)
                    yield {"pct": pct, "msg": msg}
            except StopIteration as exc:
                mesh = exc.value

            if mesh is not None:
                out_path = os.path.join(out_dir, f"{timestamp}_{key}.stl")
                mesh.export(out_path)
                saved.append(out_path)
                reports[out_path] = mesh_quality_report(mesh)
                generated_meshes[key] = mesh

    if options.combined and label_items:
        if combined_mode == "multipart" and len(label_items) > 1:
            meshes = []
            total = len(label_items)
            for index, (key, label) in enumerate(label_items):
                mesh = generated_meshes.get(key)
                if mesh is not None:
                    meshes.append(mesh)
                    continue

                mask = _mask_for_label(label)
                if mask.sum() == 0:
                    continue
                base_profile = resolve_mesh_profile(label, label_schema)
                params = _effective_profile(label, key, base_profile)
                start_pct = 10 + int(index / max(1, total) * 70)
                end_pct = 10 + int((index + 1) / max(1, total) * 70)
                driver = _drive(mask, params, start_pct, end_pct)
                try:
                    while True:
                        pct, msg = next(driver)
                        yield {"pct": pct, "msg": msg}
                except StopIteration as exc:
                    mesh = exc.value
                if mesh is not None:
                    meshes.append(mesh)

            if meshes:
                yield {"pct": 88, "msg": "Saving multipart STL..."}
                mesh_combined = _trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
                try:
                    _trimesh.repair.fix_normals(mesh_combined)
                except Exception:
                    pass
                out_path = os.path.join(out_dir, f"{timestamp}_combined.stl")
                mesh_combined.export(out_path)
                saved.append(out_path)
                reports[out_path] = mesh_quality_report(mesh_combined)
        else:
            combined_mask = np.zeros_like(seg_arr, dtype=bool)
            for _, label in label_items:
                combined_mask |= seg_arr == int(label)
            if component_labels is not None:
                combined_mask = combined_mask.copy()
                for component_id in excluded_components:
                    combined_mask[component_labels == component_id] = False

            total = len(label_items) + 1 if not options.combined_only else 1
            combined_index = len(label_items) if not options.combined_only else 0
            start_pct = 10 + int(combined_index / total * 70)
            combined_profile = combine_mesh_profiles(profiles)
            params = _effective_profile(None, "combined", combined_profile)
            driver = _drive(combined_mask, params, start_pct, 82)
            try:
                while True:
                    pct, msg = next(driver)
                    yield {"pct": pct, "msg": msg}
            except StopIteration as exc:
                mesh_combined = exc.value

            if mesh_combined is not None:
                yield {"pct": 88, "msg": "Saving STL..."}
                out_path = os.path.join(out_dir, f"{timestamp}_combined.stl")
                mesh_combined.export(out_path)
                saved.append(out_path)
                reports[out_path] = mesh_quality_report(mesh_combined)

    yield {"pct": 95, "msg": f"{len(saved)} file(s) ready."}
    yield {
        "status": "done",
        "saved": saved,
        "dir": out_dir,
        "reports": reports,
        "printer": {
            "nozzle_mm": options.nozzle_mm,
            "layer_height_mm": options.layer_height_mm,
            "scale": options.scale,
        },
        "surface_mode": surface_mode,
        "combined_mode": combined_mode,
    }
