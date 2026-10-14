"""STL viewer kick timers must not paint a disposed WebGLRenderer."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve().parent / "js" / "stl_viewer_kick.test.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required")


def test_wayland_kick_does_not_paint_disposed_renderer():
    proc = subprocess.run(
        ["node", str(HARNESS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
