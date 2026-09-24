"""Fast tests for schema-aware STL mesh export helpers."""

import json

import numpy as np

from neuroflux.mesh_export import (
    StlExportOptions,
    _inferior_world_direction,
    _surface_gaussian_sigmas,
    combine_mesh_profiles,
    effective_smoothing_params,
    iter_stl_export,
    json_ready,
    mesh_quality_report,
    normalize_combined_mode,
    normalize_label_schema,
    normalize_smoothing,
    normalize_surface_mode,
    recover_fused_sulci,
    resolve_mesh_profile,
    stl_options_from_body,
)


def _fused_gyri_seg(n=48, gap=6):
    """Two WM gyral cores under one grey-matter cap with no sulcus between.

    Axes are (z, y, x). Cores are slabs along y, separated by ``gap`` voxels
    of GM along x, with a 3-voxel GM cap above them (+z) and CSF outside.
    """
    seg = np.zeros((n, n, n), dtype=np.uint8)
    lo, hi = 10, n - 10
    seg[lo:hi, lo:hi, lo:hi] = 1  # CSF shell region
    seg[lo + 2:hi - 2, lo + 2:hi - 2, lo + 2:hi - 2] = 2  # GM block
    mid = n // 2
    core_w = 4
    left = slice(mid - gap // 2 - core_w, mid - gap // 2)
    right = slice(mid + gap // 2, mid + gap // 2 + core_w)
    base = slice(lo + 4, hi - 5)  # cores stop 3 voxels below the GM top
    seg[base, lo + 4:hi - 4, left] = 3
    seg[base, lo + 4:hi - 4, right] = 3
    # Shared WM floor under both cores (U-fibres at the sulcal fundus).
    seg[lo + 4:lo + 7, lo + 4:hi - 4, left.start:right.stop] = 3
    return seg, mid, hi


class TestFusedSulcusRecovery:
    def test_opens_sulcus_between_two_wm_cores(self):
        seg, mid, hi = _fused_gyri_seg()
        carve = recover_fused_sulci(seg, np.eye(4), [2, 3], "full")
        assert carve is not None
        assert np.all(seg[carve] == 2)
        # Midline between the cores, open to the top surface.
        cols = np.argwhere(carve)
        assert np.all(np.abs(cols[:, 2] - (mid - 0.5)) <= 1.5)
        assert cols[:, 0].max() >= hi - 4
        # A grey-matter layer stays on the WM at the fundus.
        from scipy.ndimage import distance_transform_edt

        dist_wm = distance_transform_edt(seg != 3)
        assert dist_wm[carve].min() >= 1.0

    def test_single_core_crown_is_untouched(self):
        seg = np.zeros((40, 40, 40), dtype=np.uint8)
        seg[8:32, 8:32, 8:32] = 1
        seg[10:30, 10:30, 10:30] = 2
        seg[13:27, 13:27, 13:27] = 3
        assert recover_fused_sulci(seg, np.eye(4), [2, 3], "full") is None

    def test_skips_ct_and_exports_without_gm(self):
        seg, _, _ = _fused_gyri_seg()
        assert recover_fused_sulci(seg, np.eye(4), [1], "ct") is None
        assert recover_fused_sulci(seg, np.eye(4), [3], "full") is None

    def test_export_option_is_parsed_and_defaults_on(self):
        assert stl_options_from_body({}).sulcal_recovery is True
        assert stl_options_from_body({"stl_sulcal_recovery": False}).sulcal_recovery is False

    def test_export_mesh_gets_the_groove(self, tmp_path):
        import trimesh

        seg, mid, hi = _fused_gyri_seg()

        def _export(recovery, name):
            opts = StlExportOptions(
                smoothing="custom", sigma=0.0, taubin=0, sulci_enhance=0.30,
                hc_iter=0, max_faces=200_000, combined_only=True,
                sulcal_recovery=recovery,
            )
            (tmp_path / name).mkdir()
            done = list(iter_stl_export(
                seg_arr=seg, affine=np.eye(4),
                label_items=[("gm", 2), ("wm", 3)],
                out_dir=str(tmp_path / name), timestamp=name, options=opts,
            ))[-1]
            assert done["status"] == "done", done
            return trimesh.load(done["saved"][0])

        def _top_on_midline(mesh):
            # Cut at y = mid (decimation thins flat caps, so use the section).
            # Lowest surface point near the sulcus midline, upper half only.
            section = mesh.section(plane_origin=[0, mid, 0], plane_normal=[0, 1, 0])
            v = np.asarray(section.vertices)
            near = (np.abs(v[:, 2] - (mid - 0.5)) < 1.5) & (v[:, 0] > mid)
            return float(v[near, 0].min())

        flat = _export(False, "off")
        grooved = _export(True, "on")
        # Without recovery the midline surface is the flat cap; with it the
        # surface drops into the sulcus.
        assert _top_on_midline(grooved) < _top_on_midline(flat) - 3.0


def _ellipsoid_mask(shape, radii, center=None):
    center = center or tuple(s / 2 for s in shape)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    ) <= 1.0
    return mask


class TestLabelSchemaResolution:
    def test_normalizes_aliases(self):
        assert normalize_label_schema("seg_full") == "full"
        assert normalize_label_schema("hemisphere") == "hemi"
        assert normalize_label_schema("freesurfer") == "fs"
        assert normalize_label_schema("ct_bone") == "ct"
        assert normalize_label_schema("surprise") == "full"

    def test_full_schema_profiles(self):
        assert resolve_mesh_profile(2, "full").name == "gm"
        assert resolve_mesh_profile(3, "full").name == "wm"
        assert resolve_mesh_profile(6, "full").name == "cerebellum"

    def test_hemi_schema_profiles_do_not_use_raw_label_meaning(self):
        assert resolve_mesh_profile(2, "hemi").name == "gm"
        assert resolve_mesh_profile(3, "hemi").name == "gm"
        assert resolve_mesh_profile(4, "hemi").name == "wm"
        assert resolve_mesh_profile(5, "hemi").name == "wm"
        assert resolve_mesh_profile(10, "hemi").name == "cerebellum"

    def test_hemi_label_6_is_not_wm_profile(self):
        profile = resolve_mesh_profile(6, "hemi")
        assert profile.name == "cc"
        assert profile.sigma < 0.75
        assert profile.sigma != resolve_mesh_profile(4, "hemi").sigma

    def test_freesurfer_cc_uses_cc_profile(self):
        for label in (251, 252, 253, 254, 255):
            assert resolve_mesh_profile(label, "fs").name == "cc"

    def test_freesurfer_schema_profiles(self):
        assert resolve_mesh_profile(3, "fs").name == "gm"
        assert resolve_mesh_profile(42, "fs").name == "gm"
        assert resolve_mesh_profile(2, "fs").name == "wm"
        assert resolve_mesh_profile(41, "fs").name == "wm"
        assert resolve_mesh_profile(8, "fs").name == "cerebellum"
        assert resolve_mesh_profile(47, "fs").name == "cerebellum"

    def test_ct_schema_uses_bone_profile(self):
        assert resolve_mesh_profile(1, "ct").name == "bone"


class TestCombinedProfiles:
    def test_combined_profile_preserves_most_detailed_selected_profile(self):
        gm = resolve_mesh_profile(2, "full")
        wm = resolve_mesh_profile(3, "full")
        combined = combine_mesh_profiles([wm, gm])

        assert combined.name == "combined"
        assert combined.sigma == gm.sigma
        assert combined.taubin == gm.taubin


class TestStlOptions:
    def test_normalizes_surface_and_combined_modes(self):
        assert normalize_surface_mode("signed_distance") == "sdf"
        assert normalize_surface_mode("legacy") == "volume"
        assert normalize_surface_mode("unexpected") == "sdf"
        assert normalize_combined_mode("multi_part") == "multipart"
        assert normalize_combined_mode("shell") == "external_shell"
        assert normalize_combined_mode("unexpected") == "external_shell"

    def test_parses_new_request_fields(self):
        options = stl_options_from_body({
            "label_schema": "freesurfer",
            "stl_surface_mode": "signed_distance",
            "stl_combined_mode": "multi",
            "stl_combined_only": True,
            "stl_nozzle_mm": 0.4,
            "stl_layer_height_mm": 0.12,
            "stl_scale": 1.5,
            "stl_smoothing": "custom",
        })

        assert options.label_schema == "fs"
        assert options.surface_mode == "sdf"
        assert options.combined_mode == "multipart"
        assert options.combined_only is True
        assert options.nozzle_mm == 0.4
        assert options.layer_height_mm == 0.12
        assert options.scale == 1.5
        assert options.smoothing == "custom"

    def test_smoothing_defaults_to_profile(self):
        assert stl_options_from_body({}).smoothing == "profile"
        assert normalize_smoothing("weird") == "profile"
        assert normalize_smoothing("CUSTOM") == "custom"

    def test_request_defaults_stay_combined_only(self):
        options = stl_options_from_body({})

        assert options.combined_only is True


class TestLatticeAntialias:
    def test_zero_sigma_still_blurs_one_acquired_voxel(self):
        fine, coarse = _surface_gaussian_sigmas(0.0, [1.0, 1.0, 1.0], 0.3, [1.0, 1.0, 1.0])
        assert min(fine) >= 0.85
        assert coarse is not None
        assert min(coarse) >= 2.6
        assert all(c > f for f, c in zip(fine, coarse))

    def test_upsample_keeps_blur_in_millimetres(self):
        # 2× grid: one acquired voxel is two samples. The lattice blur has
        # to follow the acquired voxel, not the finer sample.
        fine, coarse = _surface_gaussian_sigmas(0.0, [0.5, 0.5, 0.5], 0.0, [1.0, 1.0, 1.0])
        assert coarse is None
        assert all(abs(s - 1.7) < 1e-6 for s in fine)

    def test_requested_sigma_can_exceed_the_lattice_floor(self):
        fine, _ = _surface_gaussian_sigmas(2.0, [1.0, 1.0, 1.0], 0.0, [1.0, 1.0, 1.0])
        assert all(abs(s - 2.0) < 1e-6 for s in fine)

    def test_anatomical_sdf_drops_voxel_stairs(self, tmp_path):
        import trimesh

        mask = _ellipsoid_mask((40, 40, 40), (14, 13, 12))
        seg = mask.astype(np.int32) * 2
        options = StlExportOptions(
            label_schema="full",
            smoothing="custom",
            sigma=0.0,
            taubin=0,
            max_faces=200_000,
            sulci_enhance=0.30,
            hc_iter=0,
            combined_only=True,
        )
        done = list(iter_stl_export(
            seg_arr=seg,
            affine=np.eye(4),
            label_items=[("gm", 2)],
            out_dir=str(tmp_path),
            timestamp="stairs",
            options=options,
        ))[-1]
        assert done["status"] == "done", done
        mesh = trimesh.load(done["saved"][0])
        normals = np.abs(np.asarray(mesh.face_normals, dtype=np.float64))
        stair = float(np.mean(np.max(normals, axis=1) > 0.98))
        # Unsmoothed SDF marching cubes leaves about a quarter of the faces
        # on the voxel lattice. The lattice blur should take that well down.
        assert stair < 0.12

    def test_thin_cortex_keeps_sulci_and_stays_closed(self):
        from neuroflux.mesh_export import generate_mesh_from_mask

        n = 56
        center = (n - 1) / 2.0
        zz, yy, xx = np.ogrid[:n, :n, :n]
        radius = np.sqrt((zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2)
        # 1.5 mm cortical shell. The lattice blur used to pinch tunnels
        # through banks this thin (one mesh component, large negative Euler).
        shell = (radius <= 20.0) & (radius >= 18.5)
        # Solid hemisphere with a 3 mm sulcus, 5 mm deep. The thin-wall
        # clamp must not fill that groove.
        brain = radius <= 18.0
        groove = brain & ~(
            (np.abs(yy - center) <= 1.5)
            & (xx > center)
            & (radius >= 13.0)
            & (radius <= 18.6)
        )
        options = StlExportOptions(
            smoothing="custom",
            sigma=0.0,
            taubin=0,
            max_faces=500_000,
            sulci_enhance=0.30,
            hc_iter=3,
            combined_only=True,
        )

        def _mesh(mask):
            gen = generate_mesh_from_mask(
                mask, np.eye(4), {"name": "ctx", "sigma": 0.0, "taubin": 0}, options,
            )
            try:
                while True:
                    next(gen)
            except StopIteration as exc:
                return exc.value

        shell_mesh = _mesh(shell)
        assert shell_mesh is not None
        assert shell_mesh.is_watertight
        assert len(shell_mesh.split(only_watertight=False)) == 2
        assert shell_mesh.euler_number == 4

        brain_mesh = _mesh(groove)
        assert brain_mesh is not None and brain_mesh.is_watertight
        # Voxel axes are (z, y, x). Along the sulcus midline the surface
        # sits on the floor (~5 mm in), not back out on the crown.
        verts = np.asarray(brain_mesh.vertices)
        midline = (
            (np.abs(verts[:, 0] - center) < 1.2)
            & (np.abs(verts[:, 1] - center) < 1.2)
            & (verts[:, 2] > center)
        )
        floor_x = float(verts[midline, 2].min())
        assert floor_x < center + 14.5


class TestSmoothingModes:
    def test_custom_sigma_zero_does_not_use_profile_gaussian(self):
        base = resolve_mesh_profile(3, "full")  # wm sigma 0.75
        opts = StlExportOptions(smoothing="custom", sigma=0.0, taubin=0)
        sigma, taubin = effective_smoothing_params(opts, base)
        assert sigma == 0.0
        assert taubin == 0
        assert sigma != base.sigma

    def test_profile_mode_ignores_request_sigma(self):
        base = resolve_mesh_profile(2, "full")
        opts = StlExportOptions(smoothing="profile", sigma=9.0, taubin=99)
        sigma, taubin = effective_smoothing_params(opts, base)
        assert sigma == base.sigma
        assert taubin == base.taubin

    def test_custom_zero_sigma_mesh_differs_from_profile(self, tmp_path):
        mask = _ellipsoid_mask((28, 28, 28), (8, 7, 6))
        seg = mask.astype(np.int32) * 3
        affine = np.eye(4)
        common = dict(
            label_schema="full",
            max_faces=20_000,
            sulci_enhance=0.0,
            hc_iter=0,
            combined_only=True,
            taubin=0,
        )
        custom_opts = StlExportOptions(smoothing="custom", sigma=0.0, **common)
        profile_opts = StlExportOptions(smoothing="profile", sigma=0.0, **common)

        (tmp_path / "c").mkdir()
        (tmp_path / "p").mkdir()
        custom_events = list(iter_stl_export(
            seg_arr=seg, affine=affine, label_items=[("wm", 3)],
            out_dir=str(tmp_path / "c"), timestamp="c", options=custom_opts,
        ))
        profile_events = list(iter_stl_export(
            seg_arr=seg, affine=affine, label_items=[("wm", 3)],
            out_dir=str(tmp_path / "p"), timestamp="p", options=profile_opts,
        ))

        c_done = custom_events[-1]
        p_done = profile_events[-1]
        assert c_done["status"] == "done" and p_done["status"] == "done"
        assert c_done["smoothing"] == "custom"
        assert p_done["smoothing"] == "profile"
        c_rep = c_done["reports"][c_done["saved"][0]]
        p_rep = p_done["reports"][p_done["saved"][0]]
        # Zero-sigma custom surface should differ from profile-blurred WM
        assert c_rep["faces"] != p_rep["faces"] or c_rep["extents_mm"] != p_rep["extents_mm"]


class TestHollowShell:
    def test_hollow_ellipsoid_is_one_watertight_mesh(self, tmp_path):
        mask = _ellipsoid_mask((36, 36, 36), (12, 10, 9))
        seg = mask.astype(np.int32) * 2
        options = StlExportOptions(
            label_schema="full",
            smoothing="custom",
            sigma=0.2,
            taubin=0,
            max_faces=40_000,
            sulci_enhance=0.0,
            hc_iter=0,
            hollow=True,
            wall_mm=2.0,
            combined_only=True,
        )
        events = list(iter_stl_export(
            seg_arr=seg,
            affine=np.eye(4),
            label_items=[("gm", 2)],
            out_dir=str(tmp_path),
            timestamp="hollow",
            options=options,
        ))
        done = events[-1]
        assert done["status"] == "done", done
        report = done["reports"][done["saved"][0]]
        # Shell SDF yields the wall boundary (outer + cavity); may be 1–2 components
        # but must be a single exported mesh that is watertight after repair.
        assert report["watertight"] is True
        assert report["components"] in (1, 2)
        assert report.get("volume_mm3") is not None and report["volume_mm3"] > 0
        assert "error" not in report or report.get("error") is None


class TestBboxCropExtents:
    def test_padded_and_tight_masks_match_extents(self, tmp_path):
        tight = np.zeros((24, 24, 24), dtype=np.int32)
        tight_mask = _ellipsoid_mask((24, 24, 24), (7, 6, 5))
        tight[tight_mask] = 2

        padded = np.zeros((48, 48, 48), dtype=np.int32)
        # Place the same ellipsoid in the center of the padded volume
        padded[12:36, 12:36, 12:36] = tight
        # Affine so world coords of the ellipsoid region match
        aff_tight = np.eye(4)
        aff_padded = np.eye(4)
        aff_padded[:3, 3] = [-12.0, -12.0, -12.0]

        opts = StlExportOptions(
            label_schema="full",
            smoothing="custom",
            sigma=0.25,
            taubin=0,
            max_faces=20_000,
            sulci_enhance=0.0,
            hc_iter=0,
            combined_only=True,
        )
        (tmp_path / "t").mkdir()
        (tmp_path / "p").mkdir()
        t_done = list(iter_stl_export(
            seg_arr=tight, affine=aff_tight, label_items=[("gm", 2)],
            out_dir=str(tmp_path / "t"), timestamp="t", options=opts,
        ))[-1]
        p_done = list(iter_stl_export(
            seg_arr=padded, affine=aff_padded, label_items=[("gm", 2)],
            out_dir=str(tmp_path / "p"), timestamp="p", options=opts,
        ))[-1]
        assert t_done["status"] == "done" and p_done["status"] == "done"
        t_ext = np.array(t_done["reports"][t_done["saved"][0]]["extents_mm"], dtype=float)
        p_ext = np.array(p_done["reports"][p_done["saved"][0]]["extents_mm"], dtype=float)
        assert np.all(np.abs(t_ext - p_ext) < 1.0)


class TestFlatBaseDirection:
    def test_inferior_follows_affine_column_not_world_index(self):
        # Voxel axis 0 increases toward superior (world +Z). Inferior is -Z,
        # not world -X (the old axcode-index bug).
        affine = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        inferior = _inferior_world_direction(affine)
        assert np.allclose(inferior, [0.0, 0.0, -1.0], atol=1e-6)

        ras = np.eye(4)
        assert np.allclose(_inferior_world_direction(ras), [0.0, 0.0, -1.0], atol=1e-6)


class TestRepairAndQualityReport:
    def test_flat_base_report_states_watertight_honestly(self, tmp_path):
        mask = _ellipsoid_mask((30, 30, 30), (9, 8, 7))
        seg = mask.astype(np.int32) * 2
        options = StlExportOptions(
            label_schema="full",
            smoothing="custom",
            sigma=0.2,
            taubin=0,
            max_faces=8_000,
            sulci_enhance=0.0,
            hc_iter=0,
            flat_base=True,
            combined_only=True,
        )
        events = list(iter_stl_export(
            seg_arr=seg,
            affine=np.eye(4),
            label_items=[("gm", 2)],
            out_dir=str(tmp_path),
            timestamp="flat",
            options=options,
        ))
        done = events[-1]
        assert done["status"] == "done"
        report = done["reports"][done["saved"][0]]
        assert "watertight" in report
        assert isinstance(report["watertight"], bool)
        if not report["watertight"]:
            assert "error" in report
            assert done.get("quality_errors")

    def test_mesh_quality_report_require_watertight_flag(self):
        import trimesh

        # Open mesh: a single triangle
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            process=False,
        )
        report = mesh_quality_report(mesh, require_watertight=True)
        assert report["watertight"] is False
        assert "error" in report

        report_ok = mesh_quality_report(mesh, require_watertight=False)
        assert report_ok["watertight"] is False
        assert "error" not in report_ok

    def test_non_finite_metrics_stay_json_safe(self):
        class _NanMesh:
            vertices = (0, 1, 2)
            faces = (0,)
            is_watertight = True
            area = float("nan")
            volume = float("inf")
            bounds = ((0, 0, 0), (1, float("nan"), 1))
            extents = (1, 1, 1)

        report = mesh_quality_report(_NanMesh(), require_watertight=True)
        payload = json.dumps(json_ready({"status": "done", "reports": {"a": report}}), allow_nan=False)
        data = json.loads(payload)
        assert data["status"] == "done"
        assert data["reports"]["a"]["volume_mm3"] is None
        assert data["reports"]["a"]["surface_area_mm2"] is None
        assert "NaN" not in payload


class TestStlExportGenerator:
    def test_writes_combined_stl_and_report(self, tmp_path):
        seg_arr = np.zeros((18, 18, 18), dtype=np.int32)
        seg_arr[4:14, 4:14, 4:14] = 2
        affine = np.eye(4)
        options = StlExportOptions(
            label_schema="full",
            smoothing="custom",
            sigma=0.25,
            taubin=1,
            max_faces=10_000,
            sulci_enhance=0.0,
            hc_iter=0,
            combined_only=True,
        )

        events = list(iter_stl_export(
            seg_arr=seg_arr,
            affine=affine,
            label_items=[("gm", 2)],
            out_dir=str(tmp_path),
            timestamp="test",
            options=options,
        ))

        done = events[-1]
        assert done["status"] == "done"
        assert done["surface_mode"] == "sdf"
        assert any("signed distance field" in e.get("msg", "") for e in events)
        assert len(done["saved"]) == 1
        assert done["saved"][0].endswith("test_combined.stl")
        assert (tmp_path / "test_combined.stl").is_file()
        report = done["reports"][done["saved"][0]]
        assert report["faces"] > 0
        assert report["vertices"] > 0
        assert done["printer"]["nozzle_mm"] == 0.4

    def test_writes_multipart_combined_stl(self, tmp_path):
        seg_arr = np.zeros((24, 24, 24), dtype=np.int32)
        seg_arr[3:10, 3:10, 3:10] = 2
        seg_arr[14:21, 14:21, 14:21] = 3
        options = StlExportOptions(
            label_schema="full",
            max_faces=20_000,
            sulci_enhance=0.0,
            hc_iter=0,
            combined_only=True,
            combined_mode="multipart",
        )

        events = list(iter_stl_export(
            seg_arr=seg_arr,
            affine=np.eye(4),
            label_items=[("gm", 2), ("wm", 3)],
            out_dir=str(tmp_path),
            timestamp="multi",
            options=options,
        ))

        done = events[-1]
        assert done["status"] == "done"
        assert done["combined_mode"] == "multipart"
        assert len(done["saved"]) == 1
        report = done["reports"][done["saved"][0]]
        assert report["faces"] > 0
        assert report["components"] >= 2

    def test_preserves_disconnected_components_for_one_label(self, tmp_path):
        seg_arr = np.zeros((28, 28, 28), dtype=np.int32)
        seg_arr[3:10, 3:10, 3:10] = 2
        seg_arr[18:25, 18:25, 18:25] = 2
        options = StlExportOptions(
            label_schema="full",
            max_faces=20_000,
            sulci_enhance=0.0,
            hc_iter=0,
            combined_only=True,
        )

        events = list(iter_stl_export(
            seg_arr=seg_arr,
            affine=np.eye(4),
            label_items=[("gm", 2)],
            out_dir=str(tmp_path),
            timestamp="bilateral",
            options=options,
        ))

        done = events[-1]
        report = done["reports"][done["saved"][0]]
        assert report["components"] >= 2
