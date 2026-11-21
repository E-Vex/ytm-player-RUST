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
  the extension cannot be imported.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from ytm_player import __version__

REPO_ROOT = Path(__file__).resolve().parents[2]


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
