"""Safeguards for the Nix flake's native (Rust/PyO3) build.

The flake compiles the Rust extension from source (``flake.nix``) instead of
installing the pure-Python fallback. That build has several contracts this
file pins down:

- a ``ytm-player-native`` derivation built with ``rustPlatform.buildRustPackage``
  compiles the cdylib; the Python application installs it beside the wheel's
  files (hatchling honours ``.gitignore`` — which excludes ``*.so`` — even in
  the git-less flake source, so the module cannot ride along in the wheel),
- the derivation vendors its dependencies through
  ``cargoLock.lockFile = ./native/Cargo.lock`` (importCargoLock): no
  ``cargoHash`` to keep in sync, and the lock is guaranteed consistent with
  the crate because cargoSetupHook byte-compares the two,
- ``postInstall`` must place the module at
  ``ytm_player/utils/_ytm_player_native.abi3.so`` — the exact import path
  every ``from .utils._ytm_player_native import ...`` site resolves through,
  and the exact file name ``native/build.sh`` produces on Linux,
- ``pythonImportsCheck`` must import the extension: that is the build-time
  proof the installed module actually loads,
- the nixpkgs lock pin must be new enough to ship a Rust toolchain at or
  above the crate's ``rust-version`` floor (the pre-2026-11 pin shipped
  rust 1.97, older than the crate's 1.98 floor, and the build failed).

Parsing Nix from Python is regex-only here: the assertions target stable,
intentional markers of the flake's structure, not incidental formatting.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]

# flake.nix / flake.lock live in the repository but are not part of the sdist
# allowlist — skip cleanly when running the suite from an sdist (same pattern
# as tests/test_scripts/test_regenerate_srcinfo.py).
if not (REPO_ROOT / "flake.nix").is_file():
    pytest.skip("flake.nix not present in this checkout", allow_module_level=True)

FLAKE = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
FLAKE_LOCK = (REPO_ROOT / "flake.lock").read_text(encoding="utf-8")

# The module name must match native/build.sh's Linux output exactly — the
# same name the AUR package installs and the PyPI wheels carry.
NATIVE_MODULE = "_ytm_player_native.abi3.so"

# The nixpkgs pin that shipped rust 1.97 — older than the crate's
# rust-version floor, so the native build failed on it. Recorded by its
# lastModified (2026-09-02); any pin at or before this timestamp is too old.
TOO_OLD_NIXPKGS_LAST_MODIFIED = 1788372231


def _cargo() -> dict:
    return tomllib.loads((REPO_ROOT / "native" / "Cargo.toml").read_text(encoding="utf-8"))


def test_flake_builds_the_extension_with_build_rust_package():
    match = re.search(r"ytm-player-native\s*=\s*pkgs\.rustPlatform\.buildRustPackage", FLAKE)
    assert match is not None, (
        "flake.nix must build the Rust extension in a ytm-player-native "
        "rustPlatform.buildRustPackage derivation"
    )


def test_extension_deps_vendored_from_the_repo_lockfile():
    match = re.search(r"cargoLock\.lockFile\s*=\s*\./native/Cargo\.lock", FLAKE)
    assert match is not None, (
        "the native derivation must vendor dependencies via "
        "cargoLock.lockFile = ./native/Cargo.lock (importCargoLock). A "
        "cargoHash would go stale on every Cargo.lock change; the lockfile "
        "path keeps the flake build reproducible with zero hash maintenance"
    )


def test_app_installs_the_extension_at_the_import_path():
    assert "ytm_player/utils" in FLAKE, (
        "the application derivation must install the compiled module inside "
        "ytm_player/utils/ — the import path the package resolves "
        "`from .utils._ytm_player_native import ...` through"
    )
    assert NATIVE_MODULE in FLAKE, (
        f"postInstall must place {NATIVE_MODULE} (the file native/build.sh "
        "produces on Linux) — any other name would not be importable"
    )
    assert "python.sitePackages" in FLAKE, (
        "postInstall must derive the site-packages directory from "
        "python.sitePackages, not hard-code a versioned path"
    )


def test_imports_check_proves_the_extension_loads():
    matches = re.findall(r"pythonImportsCheck\s*=\s*\[(.*?)\]", FLAKE, re.DOTALL)
    assert matches, "no pythonImportsCheck list found in flake.nix"
    assert any("ytm_player.utils._ytm_player_native" in m for m in matches), (
        "the application's pythonImportsCheck must import the compiled "
        "extension: that import is the build-time proof the postInstall "
        "copy actually loads on the interpreter the application runs on"
    )


def test_dev_shell_carries_the_rust_toolchain():
    dev_shell = FLAKE.split("devShells.default", 1)[-1]
    for tool in ("cargo", "rustc"):
        assert re.search(rf"^\s+{tool}\s*$", dev_shell, re.MULTILINE), (
            f"the dev shell must provide {tool} so `nix develop` users can "
            "run native/build.sh without installing a toolchain themselves"
        )


def test_nixpkgs_pin_is_new_enough_for_the_crate_rust_version():
    """The lock pin must ship rust >= native/Cargo.toml's rust-version.

    nixpkgs only ever moves its default rust forward, so a pin strictly newer
    than the last known rust-1.97 pin satisfies the crate's 1.98 floor. This
    catches a reverted lock bump (e.g. a merge conflict resolved to the old
    lock), which would fail the build with cargo's "requires rustc X or
    newer".
    """
    crate_rust_version = _cargo()["package"]["rust-version"]
    assert crate_rust_version, "native/Cargo.toml lost its rust-version floor"

    match = re.search(r'"nixpkgs":\s*\{.*?"lastModified":\s*(\d+)', FLAKE_LOCK, re.DOTALL)
    assert match is not None, "no nixpkgs lastModified found in flake.lock"
    last_modified = int(match.group(1))
    assert last_modified > TOO_OLD_NIXPKGS_LAST_MODIFIED, (
        "the flake.lock nixpkgs pin is at or older than the last pin known "
        "to ship rust 1.97, but native/Cargo.toml requires rust "
        f"{crate_rust_version} — `nix build` of the native extension would "
        "fail. Bump the pin: nix flake update nixpkgs"
    )
