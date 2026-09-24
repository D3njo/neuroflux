"""Mesh export helpers for NeuroFlux STL generation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


class MeshGenerationError(Exception):
    """Raised when a mesh cannot be generated (e.g. hollow shell too thin)."""


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
    smoothing: str = "profile"
    sulcal_recovery: bool = True


_PROFILES = {
    "csf": MeshProfile("csf", sigma=0.8, taubin=12),
    "gm": MeshProfile("gm", sigma=0.35, taubin=4),
    "wm": MeshProfile("wm", sigma=0.75, taubin=10),
    "cc": MeshProfile("cc", sigma=0.35, taubin=4),
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
    6: "cc",
    7: "deep_gm",
    8: "deep_gm",
    9: "brainstem",
    10: "cerebellum",
}

_FS_CSF = {4, 5, 14, 15, 24, 31, 43, 44, 63, 72}
_FS_GM = {3, 42}
_FS_WM = {2, 41, 77, 78, 79}
_FS_CC = {251, 252, 253, 254, 255}
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


def normalize_smoothing(smoothing: str | None) -> str:
    """Return ``profile`` or ``custom``; unknown values fall back to ``profile``."""
    mode = (smoothing or "profile").strip().lower()
    return mode if mode in {"profile", "custom"} else "profile"


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
        elif label in _FS_CC:
            profile_key = "cc"
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


def effective_smoothing_params(
    options: StlExportOptions,
    base_profile: MeshProfile,
) -> tuple[float, int]:
    """Return (sigma, taubin) honouring ``stl_smoothing`` profile vs custom."""
    if normalize_smoothing(options.smoothing) == "custom":
        return float(options.sigma), int(options.taubin)
    return float(base_profile.sigma), int(base_profile.taubin)


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
        smoothing=normalize_smoothing(body.get("stl_smoothing", "profile")),
        sulcal_recovery=bool(body.get("stl_sulcal_recovery", True)),
    )


def _finite_number(value, digits: int):
    """Round a finite number, or return None for NaN and infinities."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def json_ready(value):
    """Convert an export event into values ``json.dumps`` can emit as strict JSON.

    Browser ``JSON.parse`` rejects ``NaN`` and ``Infinity``. A done event that
    fails to parse is dropped, so the STL preview never opens even though the
    file was already written.
    """
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def mesh_quality_report(mesh, *, require_watertight: bool = False) -> dict:
    """Return JSON-serialisable mesh quality metrics for export diagnostics."""
    try:
        watertight = bool(mesh.is_watertight)
    except Exception:
        watertight = None
    report = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": watertight,
        "surface_area_mm2": _finite_number(getattr(mesh, "area", None), 3),
        "volume_mm3": None,
    }
    if watertight:
        try:
            report["volume_mm3"] = _finite_number(mesh.volume, 3)
        except Exception:
            report["volume_mm3"] = None
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
                "min": _finite_number(edge_lengths.min(), 4),
                "median": _finite_number(np.median(edge_lengths), 4),
                "p95": _finite_number(np.percentile(edge_lengths, 95), 4),
            }
    except Exception:
        report["edge_mm"] = None
    if require_watertight and report["watertight"] is False:
        report["error"] = (
            "Mesh is not watertight (required when hollow or flat_base is enabled)"
        )
    return report


def _inferior_world_direction(affine: np.ndarray) -> np.ndarray:
    """Unit vector in world millimetres pointing toward patient inferior.

    The superior–inferior voxel axis is not always world Z. Use that column
    of the affine (negated when the axis increases toward superior).
    """
    fallback = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    try:
        from nibabel.orientations import aff2axcodes

        aff = np.asarray(affine, dtype=np.float64)
        codes = aff2axcodes(aff)
    except Exception:
        return fallback

    for i, code in enumerate(codes):
        if code not in ("S", "I"):
            continue
        direction = aff[:3, i].astype(np.float64, copy=True)
        if code == "S":
            direction *= -1.0
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            return fallback
        return direction / norm
    return fallback


# Binary marching-cubes sits on voxel faces. A blur of about one acquired
# voxel takes that staircase out. Sulci are wider, so the unsharp band has
# to start above the lattice or the high-pass paints the steps back in.
_LATTICE_SIGMA_VOX = 0.85
_SULCUS_MIN_VOX = 2.6
_GAUSSIAN_TRUNCATE = 4.0
# Inradius (mm) below which the lattice blur sags onto the isosurface.
# Cortical banks around 2–3 mm sit under this; a solid surface does not,
# because the same 1 mm window sees thicker tissue underneath.
_CORTEX_HALF_MM = 1.35
# Keep a pinched wall this far above the SDF zero level so marching cubes
# cannot open a tunnel through it. Thick surfaces never reach this clamp.
_THIN_SDF_FLOOR = 0.60


def _surface_gaussian_sigmas(
    raw_sigma_mm: float,
    sampling_mm,
    sulci_enhance: float,
    voxel_size_mm,
) -> tuple[list[float], list[float] | None]:
    """Return ``(fine, coarse)`` Gaussian sigmas in voxels of the meshed grid.

    ``fine`` is at least ``_LATTICE_SIGMA_VOX`` acquired voxels, including
    after upsampling, so a requested sigma of 0 still cannot leave cubic
    steps. ``coarse`` is the sulcal low-pass and is always wider than that.
    """
    sampling = [max(1e-3, float(s)) for s in sampling_mm]
    voxels = [max(1e-3, float(v)) for v in voxel_size_mm]
    scale = [voxels[i] / sampling[i] for i in range(3)]
    requested = [max(0.0, float(raw_sigma_mm)) / sampling[i] for i in range(3)]
    fine = [max(_LATTICE_SIGMA_VOX * scale[i], requested[i]) for i in range(3)]
    if sulci_enhance <= 0:
        return fine, None
    coarse = [
        max(fine[i] * 2.5, _SULCUS_MIN_VOX * scale[i]) for i in range(3)
    ]
    return fine, coarse


def _support_thin_cortex(
    surface_field: np.ndarray,
    mask_bool: np.ndarray,
    inside_mm: np.ndarray,
    sampling_mm,
) -> np.ndarray:
    """Stop the lattice blur from opening tunnels in thin cortex.

    ``inside_mm`` is the distance to the mask boundary. A thick surface has
    a larger distance one voxel in, so the clamp does not run there and the
    stair removal stays put. A cortical wall does not, and its smoothed
    field would otherwise graze zero.
    """
    from scipy.ndimage import maximum_filter

    if not np.any(mask_bool):
        return surface_field
    radii = [max(1, int(round(1.0 / max(float(s), 1e-3)))) for s in sampling_mm]
    local = maximum_filter(inside_mm, size=tuple(2 * r + 1 for r in radii))
    # Only lift voxels still inside the smoothed surface. Negative voxels are
    # already outside; forcing them up to the floor extrudes spikes on gyri.
    thin = (
        mask_bool
        & (local <= _CORTEX_HALF_MM)
        & (surface_field >= 0)
        & (surface_field < _THIN_SDF_FLOOR)
    )
    if not np.any(thin):
        return surface_field
    field = np.array(surface_field, dtype=np.float32, copy=True)
    field[thin] = np.float32(_THIN_SDF_FLOOR)
    return field


# Each gyrus has its own white-matter core, so two cores always have a sulcus
# between them. At 1 mm SynthSeg often labels that sulcus grey matter. The
# mesh then shows a smooth cap with isolated CSF pits instead of folds.
# Grey matter whose neighbours map to WM points this far apart sits on the
# midline between two cores.
_SULCUS_CORE_GAP_MM = 4.0
# Gap relative to the distance to WM. A gyral crown sees one core tip, so its
# neighbours map to nearby WM points; a fused sulcus sees two facing banks.
_SULCUS_GAP_RATIO = 1.4
# Keep this much grey matter on the WM so the sulcus fundus stays cortex.
_SULCUS_MIN_CORTEX_MM = 1.0

# SynthSeg partial-volume map written next to seg_full (uint8, P x 255).
NONTISSUE_MAP_NAME = "pv_nontissue.nii.gz"
# Cortex voxels at least this likely to be CSF/background are sulcal partial
# volume that argmax still gave to cortex. They also carry the thin bridges
# that close a sulcus into a row of pits.
_PV_NONTISSUE_MIN = 0.3
# Only open voxels this far below a closing hull, so the outer contour and
# with it the printed dimensions stay where the hard labels put them.
_PV_HULL_RADIUS_MM = 5.0
_PV_MIN_HULL_DEPTH_MM = 2.5

_SULCAL_TISSUE = {
    "full": (frozenset({2}), frozenset({3})),
    "hemi": (frozenset({2, 3}), frozenset({4, 5, 6})),
    "fs": (frozenset(_FS_GM), frozenset(_FS_WM | _FS_CC)),
}


def recover_fused_sulci(
    seg_arr: np.ndarray,
    affine: np.ndarray,
    selected_labels: Iterable[int],
    label_schema: str | None = "full",
) -> np.ndarray | None:
    """Return grey-matter voxels that belong to a sulcus between two WM cores.

    Only voxels on a sheet that opens onto the exterior are returned, so the
    result deepens real grooves and never hollows out the cortex. Returns
    ``None`` when the schema has no GM/WM split or nothing needs carving.
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import label as nd_label

    tissue = _SULCAL_TISSUE.get(normalize_label_schema(label_schema))
    if tissue is None:
        return None
    gm_labels, wm_labels = tissue
    selected = {int(lbl) for lbl in selected_labels}
    if not selected & gm_labels:
        return None

    seg_arr = np.asarray(seg_arr)
    gm_full = np.isin(seg_arr, list(gm_labels))
    wm_full = np.isin(seg_arr, list(wm_labels))
    if not gm_full.any() or not wm_full.any():
        return None

    coords = np.argwhere(gm_full | wm_full)
    lo = np.maximum(coords.min(axis=0) - 2, 0)
    hi = np.minimum(coords.max(axis=0) + 3, seg_arr.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    gm = gm_full[box]
    wm = wm_full[box]
    selected_mask = np.isin(seg_arr[box], list(selected))

    vox = np.abs(np.linalg.norm(np.asarray(affine, dtype=np.float64)[:3, :3], axis=0))
    vox = np.where(vox > 0, vox, 1.0)
    dist_wm, nearest = distance_transform_edt(
        ~wm, sampling=vox, return_indices=True,
    )
    nearest = nearest.astype(np.int32, copy=False)

    carve = np.zeros(gm.shape, dtype=bool)
    for axis in range(3):
        lo_side = [slice(None)] * 3
        hi_side = [slice(None)] * 3
        lo_side[axis] = slice(0, -1)
        hi_side[axis] = slice(1, None)
        lo_side, hi_side = tuple(lo_side), tuple(hi_side)
        both_gm = gm[lo_side] & gm[hi_side]
        gap_sq = np.zeros(both_gm.shape, dtype=np.float32)
        for k in range(3):
            step = (nearest[k][lo_side] - nearest[k][hi_side]).astype(np.float32)
            step *= np.float32(vox[k])
            gap_sq += step * step
        gap = np.sqrt(gap_sq)
        depth = np.minimum(dist_wm[lo_side], dist_wm[hi_side])
        hit = (
            both_gm
            & (gap >= _SULCUS_CORE_GAP_MM)
            & (gap >= _SULCUS_GAP_RATIO * depth)
            & (depth >= _SULCUS_MIN_CORTEX_MM)
        )
        carve[lo_side] |= hit
        carve[hi_side] |= hit
    del dist_wm, nearest

    if not carve.any():
        return None

    # A sheet that does not reach the exterior would only hollow the cortex.
    # Border-connected background counts as exterior; enclosed CSF does not.
    outside = ~selected_mask
    comp, _ = nd_label(carve | outside)
    border = np.unique(np.concatenate([
        comp[0].ravel(), comp[-1].ravel(),
        comp[:, 0].ravel(), comp[:, -1].ravel(),
        comp[:, :, 0].ravel(), comp[:, :, -1].ravel(),
    ]))
    border = border[border > 0]
    carve &= np.isin(comp, border)
    if not carve.any():
        return None

    full = np.zeros(seg_arr.shape, dtype=bool)
    full[box] = carve
    return full


def load_nontissue_map(seg_path: str, shape) -> np.ndarray | None:
    """Load the partial-volume map beside ``seg_path`` as float32 in [0, 1]."""
    path = os.path.join(os.path.dirname(seg_path), NONTISSUE_MAP_NAME)
    if not os.path.isfile(path):
        return None
    import nibabel as nib

    arr = np.asarray(nib.load(path).dataobj)
    if tuple(arr.shape) != tuple(shape):
        return None
    return arr.astype(np.float32) / 255.0


def partial_volume_sulci(
    seg_arr: np.ndarray,
    affine: np.ndarray,
    selected_labels: Iterable[int],
    label_schema: str | None,
    nontissue_prob: np.ndarray,
) -> np.ndarray | None:
    """Return cortex voxels inside sulci that SynthSeg sees as partly CSF."""
    from scipy.ndimage import distance_transform_edt

    tissue = _SULCAL_TISSUE.get(normalize_label_schema(label_schema))
    if tissue is None or nontissue_prob is None:
        return None
    gm_labels = tissue[0]
    selected = {int(lbl) for lbl in selected_labels}
    if not selected & gm_labels:
        return None

    seg_arr = np.asarray(seg_arr)
    mask_full = np.isin(seg_arr, list(selected))
    if not mask_full.any():
        return None
    vox = np.abs(np.linalg.norm(np.asarray(affine, dtype=np.float64)[:3, :3], axis=0))
    vox = np.where(vox > 0, vox, 1.0)
    pad = int(np.ceil(_PV_HULL_RADIUS_MM / float(vox.min()))) + 2
    mask, origin = _crop_mask_bbox(mask_full, pad)
    lo = origin.astype(int)
    box = tuple(slice(int(a), int(a) + s) for a, s in zip(lo, mask.shape))

    dilated = distance_transform_edt(~mask, sampling=vox) <= _PV_HULL_RADIUS_MM
    hull = distance_transform_edt(dilated, sampling=vox) > _PV_HULL_RADIUS_MM
    depth = distance_transform_edt(hull, sampling=vox)
    del dilated, hull

    cortex = np.isin(seg_arr[box], list(gm_labels & selected))
    pv = cortex & (nontissue_prob[box] >= _PV_NONTISSUE_MIN) & (depth >= _PV_MIN_HULL_DEPTH_MM)
    if not pv.any():
        return None
    full = np.zeros(seg_arr.shape, dtype=bool)
    full[box] = pv
    return full


def _crop_mask_bbox(mask_bool: np.ndarray, margin: int):
    """Crop a boolean mask to its foreground bbox plus margin; return crop + origin."""
    coords = np.argwhere(mask_bool)
    if coords.size == 0:
        return mask_bool, np.zeros(3, dtype=np.float64)
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, mask_bool.shape)
    cropped = mask_bool[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return cropped, lo.astype(np.float64)


def _decimation_face_target(mesh, options: StlExportOptions) -> int:
    """Combine max_faces with a nozzle-aware target edge length (~0.5–1× nozzle)."""
    current = int(len(mesh.faces))
    if current <= 0:
        return options.max_faces

    cap = int(options.max_faces)
    face_target = min(current, cap)
    target_edge = max(1e-6, float(options.nozzle_mm) * 0.75)

    try:
        area = float(mesh.area)
        median_edge = float(np.median(np.asarray(mesh.edges_unique_length, dtype=np.float64)))
    except Exception:
        return max(100, face_target)

    if area <= 0:
        return max(100, face_target)

    nozzle_faces = max(100, int(area / (0.433 * target_edge * target_edge)))
    # Sub-nozzle ridges: pull toward nozzle resolution, but do not coarsen past
    # ~nozzle edge length when max_faces still allows that finer mesh.
    if median_edge < float(options.nozzle_mm) * 0.5 and nozzle_faces < face_target:
        if nozzle_faces <= cap:
            face_target = nozzle_faces
    return max(100, face_target)


def _apply_flat_base(mesh, affine: np.ndarray):
    """Cut along patient-inferior RAS and cap the opening."""
    if len(mesh.vertices) < 4:
        return mesh
    inferior = _inferior_world_direction(affine)
    superior = -inferior
    proj = mesh.vertices @ superior
    cut = float(np.percentile(proj, 5))
    plane_origin = superior * cut
    try:
        sliced = mesh.slice_plane(plane_origin, superior, cap=True)
        if sliced is not None and len(sliced.faces) > 0:
            return sliced
    except Exception:
        pass
    # Fallback: clamp along superior axis (may leave an open cut)
    verts = mesh.vertices.copy()
    below = proj < cut
    verts[below] = verts[below] + np.outer(cut - proj[below], superior)
    mesh.vertices = verts
    return mesh


def _repair_mesh(mesh) -> None:
    """Late repair pass: degenerate faces, normals, hole fill."""
    import trimesh as _trimesh

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


def generate_mesh_from_mask(
    binary_mask: np.ndarray,
    affine: np.ndarray,
    tissue_params: dict,
    options: StlExportOptions,
):
    """
    Yield ``(local_pct, message)`` progress tuples, then return a trimesh
    (via StopIteration.value) or ``None`` if the mask is too small.

    Raises ``MeshGenerationError`` when hollow is requested but the shell
    cannot be formed.
    """
    import trimesh as _trimesh
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    from scipy.ndimage import zoom as nd_zoom
    from skimage.measure import marching_cubes
    from trimesh import smoothing as tri_smooth

    name = tissue_params.get("name", "tissue")
    voxel_count = int(np.asarray(binary_mask).astype(bool).sum())
    if voxel_count < 100:
        return None

    vox2mm = affine[:3, :3]
    origin = affine[:3, 3]
    voxel_sizes = np.abs(np.linalg.norm(vox2mm, axis=0))

    mask = binary_mask.astype(np.float32)
    surface_mode = normalize_surface_mode(options.surface_mode)
    upsample = max(1, int(options.upsample))

    if upsample > 1:
        yield 5, f"{name} - upsampling x{upsample}..."
        mask = nd_zoom(mask, upsample, order=1)

    sampling_mm = [
        float(voxel_sizes[i]) / upsample if voxel_sizes[i] > 0 else 1.0 / upsample
        for i in range(3)
    ]
    mask_bool = mask >= 0.5

    # Hollow: anisotropic erosion after upsample, single shell SDF
    if options.hollow:
        yield 6, f"{name} - hollow shell..."
        if not mask_bool.any():
            raise MeshGenerationError(
                f"Hollow shell failed for {name}: empty mask after upsample"
            )
        inside_dist = distance_transform_edt(mask_bool, sampling=sampling_mm)
        eroded = inside_dist > float(options.wall_mm)
        if not eroded.any():
            raise MeshGenerationError(
                f"Hollow shell failed for {name}: wall_mm={options.wall_mm} "
                f"erodes through the entire structure"
            )
        shell = mask_bool & ~eroded
        if int(shell.sum()) < 100:
            raise MeshGenerationError(
                f"Hollow shell failed for {name}: shell too thin to mesh "
                f"({int(shell.sum())} voxels)"
            )
        mask_bool = shell
        mask = shell.astype(np.float32)

    raw_sigma = max(0.0, float(tissue_params.get("sigma", 0.8)))
    voxel_size_mm = [
        float(voxel_sizes[i]) if voxel_sizes[i] > 0 else 1.0 for i in range(3)
    ]
    fine_sigma, coarse_sigma = _surface_gaussian_sigmas(
        raw_sigma,
        sampling_mm,
        float(options.sulci_enhance),
        voxel_size_mm,
    )
    sigmas = fine_sigma if coarse_sigma is None else [*fine_sigma, *coarse_sigma]
    # Crop to foreground bbox. Margin must cover the Gaussian kernel
    # (truncate * sigma) so the surface is not distorted by the crop edge.
    margin = max(
        4,
        int(np.ceil(_GAUSSIAN_TRUNCATE * max(sigmas))) + 1,
        int(np.ceil(2.0 / max(min(sampling_mm), 0.01))),
    )
    mask_bool, crop_origin = _crop_mask_bbox(mask_bool, margin)
    mask = mask_bool.astype(np.float32)

    if surface_mode == "sdf":
        yield 8, f"{name} - signed distance field..."
        if not mask_bool.any() or mask_bool.all():
            return None
        inside = distance_transform_edt(mask_bool, sampling=sampling_mm)
        outside = distance_transform_edt(~mask_bool, sampling=sampling_mm)
        sdf = (inside - outside).astype(np.float32, copy=False)
        # Blur the raw field at both scales. A second blur of an already
        # blurred field would compound and pull sulci back out of band.
        fine_field = gaussian_filter(sdf, sigma=fine_sigma)
        if coarse_sigma is not None:
            coarse_field = gaussian_filter(sdf, sigma=coarse_sigma)
            surface_field = fine_field + float(options.sulci_enhance) * 0.35 * (
                fine_field - coarse_field
            )
        else:
            surface_field = fine_field
        surface_field = _support_thin_cortex(
            surface_field, mask_bool, inside, sampling_mm,
        )
        if surface_field.max() <= 0 or surface_field.min() >= 0:
            return None
        surface_level = 0.0
    else:
        yield 8, f"{name} - volume surface field..."
        fine_field = gaussian_filter(mask, sigma=fine_sigma)
        if float(fine_field.max()) < 0.3:
            return None
        if coarse_sigma is not None:
            coarse_field = gaussian_filter(mask, sigma=coarse_sigma)
            surface_field = np.clip(
                fine_field + float(options.sulci_enhance) * (fine_field - coarse_field),
                0.0,
                1.0,
            )
            if float(surface_field.max()) < 0.3:
                return None
        else:
            surface_field = fine_field
        surface_level = 0.5

    yield 30, f"{name} - marching cubes..."
    try:
        verts, faces_mc, _, _ = marching_cubes(surface_field, level=surface_level)
    except Exception as exc:
        return None

    verts = verts + crop_origin
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

    # 1) Smoothing (Taubin / HC)
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

    # 2) Decimation (nozzle-aware + max_faces)
    face_target = _decimation_face_target(mesh, options)
    yield 78, f"{name} - decimation ({len(mesh.faces):,} → {face_target:,} faces)..."
    if len(mesh.faces) > face_target:
        decimated = False
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=face_target)
            decimated = True
        except Exception:
            pass
        if not decimated:
            try:
                pitch = float(np.max(mesh.extents)) / ((face_target / 2) ** 0.5)
                mesh = mesh.simplify_vertex_clustering(pitch)
                if len(mesh.faces) > face_target * 1.5:
                    mesh = mesh.simplify_quadric_decimation(face_count=face_target)
            except Exception:
                pass

    # 3) Flat base along inferior RAS
    if options.flat_base and len(mesh.vertices) > 3:
        yield 85, f"{name} - flat base..."
        try:
            mesh = _apply_flat_base(mesh, affine)
        except Exception:
            pass

    # 4) Late repair — watertightness evaluated after this
    yield 90, f"{name} - mesh repair..."
    _repair_mesh(mesh)

    if abs(options.scale - 1.0) > 0.01:
        mesh.vertices *= options.scale

    yield 94, f"{name} - {len(mesh.faces):,} faces"
    return mesh


def _apply_sulcal_recovery(seg_arr, affine, selected_labels, options: StlExportOptions,
                           nontissue_prob=None):
    """Return ``seg_arr`` with fused sulci set to background (copy only if carved)."""
    if not options.sulcal_recovery:
        return seg_arr
    carve = recover_fused_sulci(seg_arr, affine, selected_labels, options.label_schema)
    pv = None
    if nontissue_prob is not None:
        pv = partial_volume_sulci(
            seg_arr, affine, selected_labels, options.label_schema, nontissue_prob,
        )
    if carve is None and pv is None:
        return seg_arr
    work = np.array(seg_arr, copy=True)
    if carve is not None:
        work[carve] = 0
    if pv is not None:
        work[pv] = 0
    return work


def build_tissue_meshes(
    seg_arr: np.ndarray,
    affine: np.ndarray,
    labels: dict,
    options: StlExportOptions | None = None,
    max_faces: int | None = None,
) -> list[tuple]:
    """
    Build per-label meshes with the same SDF pipeline as STL export.

    Returns a list of ``(trimesh, label_int)`` suitable for the 3D preview.
    """
    options = options or StlExportOptions()
    if max_faces is not None:
        options = replace(options, max_faces=max_faces)
    label_schema = normalize_label_schema(options.label_schema)
    unique_labels = sorted({int(lbl) for lbl in labels.values() if int(lbl) > 0})
    seg_arr = _apply_sulcal_recovery(seg_arr, affine, unique_labels, options)
    tissue_meshes = []
    for lbl in unique_labels:
        mask = seg_arr == lbl
        if int(mask.sum()) < 100:
            continue
        base_profile = resolve_mesh_profile(lbl, label_schema)
        sigma, taubin = effective_smoothing_params(options, base_profile)
        params = {
            "label": lbl,
            "name": f"label_{lbl}",
            "sigma": sigma,
            "taubin": taubin,
            "profile": base_profile.name,
        }
        gen = generate_mesh_from_mask(mask, affine, params, options)
        mesh = None
        try:
            while True:
                next(gen)
        except StopIteration as exc:
            mesh = exc.value
        except MeshGenerationError:
            continue
        if mesh is not None:
            tissue_meshes.append((mesh, lbl))
    return tissue_meshes


def iter_stl_export(
    seg_arr: np.ndarray,
    affine: np.ndarray,
    label_items: list[tuple[str, int]],
    out_dir: str,
    timestamp: str,
    excluded_components: set[int] | None = None,
    options: StlExportOptions | None = None,
    nontissue_prob: np.ndarray | None = None,
):
    """Yield progress dictionaries while generating STL meshes.

    ``nontissue_prob`` is the SynthSeg partial-volume map (see
    ``load_nontissue_map``); without it only the WM-core sulcus recovery runs.
    """
    try:
        import trimesh as _trimesh
        from scipy.ndimage import label as nd_label
    except ImportError as exc:
        yield {"status": "error", "error": f"Missing dependency: {exc}"}
        return

    options = options or StlExportOptions()
    excluded_components = excluded_components or set()
    label_schema = normalize_label_schema(options.label_schema)
    surface_mode = normalize_surface_mode(options.surface_mode)
    combined_mode = normalize_combined_mode(options.combined_mode)
    require_wt = bool(options.hollow or options.flat_base)

    yield {"pct": 3, "msg": "Preparing mesh pipeline..."}

    component_labels = None
    if excluded_components:
        selected_mask = np.zeros_like(seg_arr, dtype=np.uint8)
        for _, label in label_items:
            selected_mask[seg_arr == int(label)] = 1
        component_labels, _ = nd_label(selected_mask)
        sizes = np.bincount(component_labels.ravel())
        main_id = int(np.argmax(sizes[1:]) + 1) if sizes.size > 1 else 0
        # Preview component ids are not the same numbering when the preview
        # labeled a wider set of tissues. Never let that delete the brain.
        if main_id in excluded_components:
            excluded_components = {cid for cid in excluded_components if cid != main_id}

    # Component IDs above come from the uncarved mask, matching the preview.
    if options.sulcal_recovery:
        yield {"pct": 6, "msg": "Recovering fused sulci..."}
    seg_arr = _apply_sulcal_recovery(
        seg_arr, affine, [label for _, label in label_items], options, nontissue_prob,
    )

    def _mask_for_label(label: int):
        mask = seg_arr == int(label)
        if component_labels is not None:
            mask = mask.copy()
            for component_id in excluded_components:
                mask[component_labels == component_id] = False
        return mask

    def _effective_profile(label: int | None, name: str, base_profile: MeshProfile) -> dict:
        sigma, taubin = effective_smoothing_params(options, base_profile)
        return {
            "label": label,
            "name": name,
            "sigma": sigma,
            "taubin": taubin,
            "profile": base_profile.name,
            "smoothing": normalize_smoothing(options.smoothing),
        }

    def _drive(binary_mask, tissue_params, start_pct, end_pct):
        span = max(1, end_pct - start_pct)
        gen = generate_mesh_from_mask(binary_mask, affine, tissue_params, options)
        mesh = None
        try:
            while True:
                local_pct, msg = next(gen)
                yield start_pct + int(local_pct / 100 * span), msg
        except StopIteration as exc:
            mesh = exc.value
        except MeshGenerationError:
            raise
        return mesh

    saved = []
    reports = {}
    quality_errors = []
    generated_meshes = {}

    def _record_quality(mesh, out_path):
        try:
            report = mesh_quality_report(mesh, require_watertight=require_wt)
        except Exception as exc:
            report = {"error": str(exc), "watertight": None}
        reports[out_path] = report
        if report.get("error"):
            quality_errors.append(report["error"])

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
            except MeshGenerationError as exc:
                yield {"status": "error", "error": str(exc)}
                return

            if mesh is not None:
                out_path = os.path.join(out_dir, f"{timestamp}_{key}.stl")
                mesh.export(out_path)
                saved.append(out_path)
                _record_quality(mesh, out_path)
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
                except MeshGenerationError as exc:
                    yield {"status": "error", "error": str(exc)}
                    return
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
                _record_quality(mesh_combined, out_path)
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
            except MeshGenerationError as exc:
                yield {"status": "error", "error": str(exc)}
                return

            if mesh_combined is not None:
                yield {"pct": 88, "msg": "Saving STL..."}
                out_path = os.path.join(out_dir, f"{timestamp}_combined.stl")
                mesh_combined.export(out_path)
                saved.append(out_path)
                _record_quality(mesh_combined, out_path)

    yield {"pct": 95, "msg": f"{len(saved)} file(s) ready."}
    done = {
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
        "smoothing": normalize_smoothing(options.smoothing),
    }
    if quality_errors:
        done["quality_errors"] = quality_errors
    yield done
