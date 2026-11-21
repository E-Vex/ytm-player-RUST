"""Safeguards for the binary-wheel build configuration.

The published wheels are built by maturin (see ``.github/workflows/publish.yml``)
while hatchling keeps building the sdist, the pure-Python fallback wheel and
source installs. That split hides several contracts that only surface at
release time, which these tests pin down:

- the ``[tool.maturin]`` table places the compiled extension at the exact
  import path the package's ``from .utils._ytm_player_native import ...``
  sites expect,
- the PyO3 ``abi3-py310`` feature pins the stable-ABI floor to the
  ``requires-python`` floor, which is what makes the ``cp310-abi3`` wheel tag
  honest (one binary loads on every supported Python),
- the version maturin stamps into the wheels comes from
  ``native/Cargo.toml`` — it must never drift from ``__version__``, the
  single source of truth the release tag guard checks,
- the ``#[pymodule]`` function name matches the module name maturin
  exports (``PyInit__ytm_player_native``), otherwise the wheel builds but
  the extension cannot be imported,
- the ci.yml ``wheels`` job mirrors publish.yml's build matrix exactly
  (same per-leg platform tags), and the macOS leg expects the compressed
  platform tag set maturin actually emits for a universal2 wheel — a
  single ``macosx_11_0_universal2`` expectation rejects the real wheel,
  which is how the first TestPyPI dry-run failed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from ytm_player import __version__

REPO_ROOT = Path(__file__).resolve().parents[2]

# .github/workflows is not part of the sdist allowlist — skip cleanly when
# running the suite from an sdist.
_WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _cargo() -> dict:
    return tomllib.loads((REPO_ROOT / "native" / "Cargo.toml").read_text(encoding="utf-8"))


def test_maturin_config_places_extension_where_the_package_imports_it():
    maturin = _pyproject()["tool"]["maturin"]
    assert maturin["python-source"] == "src"
    assert maturin["module-name"] == "ytm_player.utils._ytm_player_native"


def test_pyo3_abi3_floor_matches_requires_python():
    pyproject = _pyproject()
    assert pyproject["project"]["requires-python"] == ">=3.10"
    features = _cargo()["dependencies"]["pyo3"]["features"]
    assert "abi3" in features
    assert "abi3-py310" in features, (
        "the abi3 floor must stay pinned to requires-python (3.10) — "
        "it is what makes the cp310-abi3 wheel tag truthful"
    )
    assert "extension-module" in features


def test_cargo_version_matches_package_version():
    cargo_version = _cargo()["package"]["version"]
    assert cargo_version == __version__, (
        f"native/Cargo.toml says {cargo_version!r} but __version__ is {__version__!r}: "
        "maturin stamps the wheels with the Cargo version, so the binary wheels "
        "would carry a different version than the sdist and the pure wheel"
    )


def test_pymodule_name_matches_maturin_module_name():
    module_name = _pyproject()["tool"]["maturin"]["module-name"].rsplit(".", 1)[-1]
    lib_rs = (REPO_ROOT / "native" / "src" / "lib.rs").read_text(encoding="utf-8")
    match = re.search(r"^#\[pymodule\]\s*\n\s*fn\s+(\w+)\s*\(", lib_rs, re.MULTILINE)
    assert match is not None, "no #[pymodule] function found in native/src/lib.rs"
    assert match.group(1) == module_name, (
        f"the #[pymodule] function is {match.group(1)!r} but maturin exports "
        f"PyInit_{module_name} — the built wheel would not be importable"
    )


def _wheel_matrix(workflow: str, job: str) -> dict[str, str]:
    """The build matrix's per-leg platform tags from a workflow file."""
    path = _WORKFLOWS_DIR / workflow
    if not path.is_file():
        pytest.skip(f"{workflow} not present in this checkout")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    include = doc["jobs"][job]["strategy"]["matrix"]["include"]
    return {leg["name"]: leg["platform-tag"] for leg in include if "platform-tag" in leg}


def test_ci_wheels_matrix_mirrors_publish_build_matrix():
    """The ci.yml wheels job exists to prove the publish matrix on every push.

    It can only do that while the two carry identical per-leg expectations —
    a tag fixed in one workflow but not the other makes CI prove a wheel the
    release never builds (or vice versa).
    """
    ci = _wheel_matrix("ci.yml", "wheels")
    publish = _wheel_matrix("publish.yml", "build")
    assert ci == publish, (
        "the ci.yml wheels matrix and the publish.yml build matrix have "
        "drifted apart: fix whichever is wrong so they carry identical "
        f"platform tags per leg (ci: {ci}, publish: {publish})"
    )


def test_macos_platform_tag_is_the_full_compressed_tag_set():
    """maturin tags the universal2 fat wheel with every platform it satisfies.

    cargo-zigbuild lipo's both slices into one binary, so the wheel name and
    the WHEEL ``Tag:`` headers carry the compressed
    arm64/universal2/x86_64 tag set. A single ``macosx_11_0_universal2``
    expectation rejects the real wheel at check_native_wheel.py — exactly how
    the first TestPyPI dry-run failed the macOS leg.
    """
    for workflow, job in (("ci.yml", "wheels"), ("publish.yml", "build")):
        tags = _wheel_matrix(workflow, job)
        assert tags.get("macos-universal2") == (
            "macosx_11_0_arm64.macosx_11_0_universal2.macosx_11_0_x86_64"
        ), f"{workflow}'s macOS platform tag must be the full compressed tag set"
