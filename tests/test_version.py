"""Project invariant: package version matches pyproject.toml."""
import pathlib
import re

import neuroflux

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_version_matches_pyproject():
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version = "([^"]+)"', text)
    assert match, "version field missing from pyproject.toml"
    assert neuroflux.__version__ == match.group(1)
