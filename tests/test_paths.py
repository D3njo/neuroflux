"""Unit tests for neuroflux.paths — model-directory resolution."""
import os

from neuroflux.paths import resolve_models_dir, setup_models_dir


class TestResolveModelsDir:
    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NEUROFLUX_MODELS_DIR", str(tmp_path))
        assert resolve_models_dir() == tmp_path
        assert setup_models_dir() == tmp_path

    def test_setup_and_require_weights_share_env_when_valid(self, monkeypatch, tmp_path):
        (tmp_path / "synthseg_2.0.h5").write_bytes(b"x" * 1_000_001)
        monkeypatch.setenv("NEUROFLUX_MODELS_DIR", str(tmp_path))
        assert resolve_models_dir(require_weights=True) == tmp_path
        assert setup_models_dir() == tmp_path

    def test_require_weights_honours_env_even_if_tiny(self, monkeypatch, tmp_path):
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        (env_dir / "synthseg_2.0.h5").write_bytes(b"tiny")
        monkeypatch.setenv("NEUROFLUX_MODELS_DIR", str(env_dir))
        result = resolve_models_dir(require_weights=True)
        assert result == env_dir

    def test_env_unset_does_not_crash(self, monkeypatch):
        monkeypatch.delenv("NEUROFLUX_MODELS_DIR", raising=False)
        path = resolve_models_dir()
        assert path.name == "models" or "neuroflux" in str(path).replace("\\", "/")
        assert os.path.isabs(str(path))
