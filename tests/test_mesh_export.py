"""Fast tests for schema-aware STL mesh export helpers."""

import numpy as np

from neuroflux.mesh_export import (
    StlExportOptions,
    combine_mesh_profiles,
    iter_stl_export,
    normalize_combined_mode,
    normalize_label_schema,
    normalize_surface_mode,
    resolve_mesh_profile,
    stl_options_from_body,
)


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
        })

        assert options.label_schema == "fs"
        assert options.surface_mode == "sdf"
        assert options.combined_mode == "multipart"
        assert options.combined_only is True
        assert options.nozzle_mm == 0.4
        assert options.layer_height_mm == 0.12
        assert options.scale == 1.5

    def test_request_defaults_stay_combined_only(self):
        options = stl_options_from_body({})

        assert options.combined_only is True


class TestStlExportGenerator:
    def test_writes_combined_stl_and_report(self, tmp_path):
        seg_arr = np.zeros((18, 18, 18), dtype=np.int32)
        seg_arr[4:14, 4:14, 4:14] = 2
        affine = np.eye(4)
        options = StlExportOptions(
            label_schema="full",
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
