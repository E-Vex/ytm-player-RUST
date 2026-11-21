"""Safeguards for the AUR package's native (Rust/PyO3) build.

The AUR ``ytm-player-git`` package compiles the Rust extension from source
instead of installing the pure-Python fallback (``aur/PKGBUILD``). That build
has several contracts this file pins down:

- the package ships a compiled binary, so ``arch`` must not be ``any`` and
  must match the platforms the PyPI binary wheels cover,
- ``cargo`` must be in ``makedepends`` — it is the whole toolchain the
  extension build needs (Arch's ``python`` package already ships the headers
  pyo3's build script reads),
- ``build()`` must run ``native/build.sh``, the same upstream build path the
  README documents, so artifact/module naming changes stay picked up,
- ``package()`` must install the compiled module at
  ``ytm_player/utils/_ytm_player_native.abi3.so`` — the exact import path
  every ``from .utils._ytm_player_native import ...`` site resolves through,
  and the exact file name ``native/build.sh`` produces on Linux. The module
  cannot ride along in the wheel: hatchling honours ``.gitignore`` (which
  excludes ``*.so``), so it is installed beside the wheel's files instead,
- ``source`` must point at the repository that actually carries ``native/``
  (the Rust-transition repo). The upstream ``peternaame-boop/ytm-player``
  master does not contain the Rust extension, so a PKGBUILD sourcing it
  would fail in ``build()`` the moment a user builds the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKGBUILD = (REPO_ROOT / "aur" / "PKGBUILD").read_text(encoding="utf-8")

# Make scripts/regenerate_srcinfo.py importable as a module (same pattern as
# tests/test_scripts/test_regenerate_srcinfo.py).
SCRIPT_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from regenerate_srcinfo import parse_pkgbuild  # noqa: E402

# The repository that carries native/ — the Rust transition's home. See the
# module docstring: the upstream peternaame-boop/ytm-player master has no
# native/ directory, so only this source URL can build the extension.
NATIVE_SOURCE_URL = "git+https://github.com/E-Vex/ytm-player-RUST.git"


def test_arch_matches_the_binary_wheel_platforms():
    parsed = parse_pkgbuild(PKGBUILD)
    arch = parsed["arch"]
    assert isinstance(arch, list)
    assert "any" not in arch, (
        "the package ships the compiled Rust extension, so it is not "
        "architecture-independent — arch=('any') would let makepkg build it "
        "on platforms the extension was never compiled for"
    )
    assert {"x86_64", "aarch64"} <= set(arch), (
        "arch should cover at least the platforms the PyPI binary wheels "
        f"ship (x86_64, aarch64); got {arch}"
    )


def test_cargo_is_a_makedepend():
    parsed = parse_pkgbuild(PKGBUILD)
    makedepends = parsed["makedepends"]
    assert isinstance(makedepends, list)
    assert "cargo" in makedepends, (
        "building the Rust extension needs the Rust toolchain; without "
        "'cargo' in makedepends the build fails on a clean chroot"
    )


def test_build_runs_the_upstream_native_build_script():
    assert "native/build.sh" in PKGBUILD, (
        "build() must compile the extension via native/build.sh (the "
        "documented upstream build path) rather than a cargo invocation "
        "that could drift from it"
    )


def test_package_installs_the_extension_at_the_import_path():
    # The module name must match native/build.sh's Linux output exactly.
    build_sh = (REPO_ROOT / "native" / "build.sh").read_text(encoding="utf-8")
    assert "module=_ytm_player_native.abi3.so" in build_sh
    module = "_ytm_player_native.abi3.so"

    assert "ytm_player/utils" in PKGBUILD, (
        "package() must install the compiled module inside the wheel's "
        "ytm_player/utils/ directory — the import path the package resolves "
        "`from .utils._ytm_player_native import ...` through"
    )
    assert module in PKGBUILD, (
        f"package() must install {module} (the file native/build.sh "
        "produces on Linux) — any other name would not be importable"
    )
    assert "purelib" in PKGBUILD, (
        "package() must derive the site-packages directory from the "
        "interpreter (sysconfig purelib), not hard-code a versioned path"
    )


def test_source_points_at_the_repository_that_carries_native():
    parsed = parse_pkgbuild(PKGBUILD)
    source = parsed["source"]
    assert isinstance(source, list)
    assert NATIVE_SOURCE_URL in source, (
        f"source must fetch from the Rust-transition repository "
        f"({NATIVE_SOURCE_URL}); the upstream peternaame-boop/ytm-player "
        "master does not contain native/, so its build() would fail. If the "
        "Rust work has been merged upstream, update this pin along with the "
        "source URL."
    )
