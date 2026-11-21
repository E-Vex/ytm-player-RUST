"""Tests for the publish-workflow binary-wheel verifier.

The script lives outside the package (under ``.github/scripts/``) so it gets
imported by path here. The wheels are synthetic: only the structure the
script reads (dist-info headers, entry points, the compiled extension's
binary header) is materialised, which keeps every failure mode of the
checker testable without a Rust toolchain.
"""

from __future__ import annotations

import importlib.util
import struct
import zipfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "check_native_wheel.py"
_spec = importlib.util.spec_from_file_location("check_native_wheel", _SCRIPT)
assert _spec is not None and _spec.loader is not None
check_native_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_native_wheel)

ENTRY_POINT = "ytm=ytm_player.cli:main"
CPU_TYPE_X86_64 = 0x01000007
CPU_TYPE_ARM64 = 0x0100000C


def _elf(machine: int) -> bytes:
    data = bytearray(b"\x7fELF" + b"\x00" * 36)
    struct.pack_into("<H", data, 18, machine)
    return bytes(data)


def _macho_universal2() -> bytes:
    entry = lambda cpu: struct.pack(">i", cpu) + b"\x00" * 16  # noqa: E731
    return (
        b"\xca\xfe\xba\xbe" + struct.pack(">I", 2) + entry(CPU_TYPE_X86_64) + entry(CPU_TYPE_ARM64)
    )


def _pe(machine: int) -> bytes:
    pe_offset = 0x40
    data = bytearray(b"MZ" + b"\x00" * (0x3C - 2))
    data += struct.pack("<I", pe_offset)  # e_lfanew at 0x3C
    data += b"PE\x00\x00" + struct.pack("<H", machine)  # signature at pe_offset
    return bytes(data)


def _build_wheel(
    tmp_path: Path,
    *,
    platform: str,
    payload: bytes,
    version: str = "2.1.0",
    python_tag: str = "cp310",
    abi_tag: str = "abi3",
    tag_lines: list[str] | None = None,
    root_is_purelib: str = "false",
    entry_point: str = ENTRY_POINT,
    include_extension: bool = True,
    stray_module: str | None = None,
    filename: str | None = None,
) -> Path:
    """Write a synthetic wheel with the given properties and return its path."""
    extension_name = (
        "ytm_player/utils/_ytm_player_native.pyd"
        if payload[:2] == b"MZ"
        else "ytm_player/utils/_ytm_player_native.abi3.so"
    )
    if tag_lines is None:
        tag_lines = [f"{python_tag}-{abi_tag}-{part}" for part in platform.split(".")]
    wheel_metadata = "".join(
        f"{line}\n"
        for line in [
            "Wheel-Version: 1.0",
            f"Root-Is-Purelib: {root_is_purelib}",
            *(f"Tag: {t}" for t in tag_lines),
        ]
    )
    dist_info = f"ytm_player-{version}.dist-info"
    contents = {
        "ytm_player/__init__.py": f'__version__ = "{version}"\n',
        "ytm_player/cli.py": "# entry point target\n",
        "ytm_player/py.typed": "",
        "ytm_player/utils/_ytm_player_native.pyi": "# stubs\n",
        f"{dist_info}/METADATA": f"Metadata-Version: 2.4\nName: ytm-player\nVersion: {version}\n\nDescription.\n",
        f"{dist_info}/WHEEL": wheel_metadata,
        f"{dist_info}/entry_points.txt": f"[console_scripts]\n{entry_point}\n",
        f"{dist_info}/RECORD": "",
    }
    if include_extension:
        contents[extension_name] = "<binary>"
    if stray_module is not None:
        contents[stray_module] = "<binary>"

    path = tmp_path / (filename or f"ytm_player-{version}-{python_tag}-{abi_tag}-{platform}.whl")
    with zipfile.ZipFile(path, "w") as wheel:
        for name, text in contents.items():
            if name == extension_name and include_extension:
                continue  # written once, with the real payload, below
            wheel.writestr(name, text)
        if include_extension:
            wheel.writestr(extension_name, payload)
    return path


def test_accepts_manylinux_x86_64_wheel(tmp_path):
    wheel = _build_wheel(
        tmp_path, platform="manylinux2014_x86_64.manylinux_2_17_x86_64", payload=_elf(62)
    )
    check_native_wheel.check_wheel(
        str(wheel), "cp310", "abi3", "manylinux2014_x86_64.manylinux_2_17_x86_64", elf_machine=62
    )


def test_accepts_manylinux_aarch64_wheel(tmp_path):
    wheel = _build_wheel(
        tmp_path, platform="manylinux2014_aarch64.manylinux_2_17_aarch64", payload=_elf(183)
    )
    check_native_wheel.check_wheel(
        str(wheel), "cp310", "abi3", "manylinux2014_aarch64.manylinux_2_17_aarch64", elf_machine=183
    )


def test_accepts_macos_universal2_wheel(tmp_path):
    # The real maturin 1.16 output: the fat binary satisfies every macOS
    # platform, so the wheel name and the WHEEL Tag: headers carry the
    # compressed arm64/universal2/x86_64 tag set (one Tag: line per part —
    # exactly what a wheel built with --target universal2-apple-darwin
    # carries; a single macosx_11_0_universal2 expectation rejects it, which
    # is how the first TestPyPI dry-run failed).
    platform = "macosx_11_0_arm64.macosx_11_0_universal2.macosx_11_0_x86_64"
    wheel = _build_wheel(tmp_path, platform=platform, payload=_macho_universal2())
    check_native_wheel.check_wheel(str(wheel), "cp310", "abi3", platform, macho_universal2=True)


def test_accepts_windows_wheel(tmp_path):
    wheel = _build_wheel(tmp_path, platform="win_amd64", payload=_pe(0x8664))
    check_native_wheel.check_wheel(str(wheel), "cp310", "abi3", "win_amd64", pe_machine=0x8664)


def test_rejects_wrong_elf_machine(tmp_path):
    wheel = _build_wheel(tmp_path, platform="manylinux_2_17_x86_64", payload=_elf(183))
    with pytest.raises(ValueError, match="e_machine is 183, expected 62"):
        check_native_wheel.check_wheel(
            str(wheel), "cp310", "abi3", "manylinux_2_17_x86_64", elf_machine=62
        )


def test_rejects_wrong_pe_machine(tmp_path):
    wheel = _build_wheel(tmp_path, platform="win_amd64", payload=_pe(0xAA64))
    with pytest.raises(ValueError, match="PE machine"):
        check_native_wheel.check_wheel(str(wheel), "cp310", "abi3", "win_amd64", pe_machine=0x8664)


def test_rejects_universal2_wheel_missing_arm64_slice(tmp_path):
    single = (
        b"\xca\xfe\xba\xbe"
        + struct.pack(">I", 1)
        + struct.pack(">i", CPU_TYPE_X86_64)
        + b"\x00" * 16
    )
    wheel = _build_wheel(tmp_path, platform="macosx_11_0_universal2", payload=single)
    with pytest.raises(ValueError, match="universal binary"):
        check_native_wheel.check_wheel(
            str(wheel), "cp310", "abi3", "macosx_11_0_universal2", macho_universal2=True
        )


def test_rejects_mismatched_filename_tag(tmp_path):
    wheel = _build_wheel(tmp_path, platform="manylinux_2_17_x86_64", payload=_elf(62))
    with pytest.raises(ValueError, match="wheel is named"):
        check_native_wheel.check_wheel(
            str(wheel),
            "cp310",
            "abi3",
            "manylinux2014_x86_64.manylinux_2_17_x86_64",
            elf_machine=62,
        )


def test_rejects_tag_header_mismatch(tmp_path):
    # The WHEEL metadata must list every tag the compressed file name claims.
    wheel = _build_wheel(
        tmp_path,
        platform="manylinux2014_x86_64.manylinux_2_17_x86_64",
        payload=_elf(62),
        tag_lines=["cp310-abi3-manylinux2014_x86_64"],
    )
    with pytest.raises(ValueError, match="WHEEL Tag headers"):
        check_native_wheel.check_wheel(
            str(wheel),
            "cp310",
            "abi3",
            "manylinux2014_x86_64.manylinux_2_17_x86_64",
            elf_machine=62,
        )


def test_rejects_tag_header_with_extra_tag(tmp_path):
    wheel = _build_wheel(
        tmp_path,
        platform="manylinux_2_17_x86_64",
        payload=_elf(62),
        tag_lines=["cp310-abi3-manylinux_2_17_x86_64", "cp310-abi3-manylinux_2_28_x86_64"],
    )
    with pytest.raises(ValueError, match="WHEEL Tag headers"):
        check_native_wheel.check_wheel(
            str(wheel), "cp310", "abi3", "manylinux_2_17_x86_64", elf_machine=62
        )


def test_rejects_purelib_root(tmp_path):
    wheel = _build_wheel(
        tmp_path, platform="win_amd64", payload=_pe(0x8664), root_is_purelib="true"
    )
    with pytest.raises(ValueError, match="Root-Is-Purelib"):
        check_native_wheel.check_wheel(str(wheel), "cp310", "abi3", "win_amd64", pe_machine=0x8664)


def test_rejects_version_drift(tmp_path):
    wheel = _build_wheel(
        tmp_path, platform="manylinux_2_17_x86_64", payload=_elf(62), version="2.1.0"
    )
    with pytest.raises(ValueError, match="drifted apart"):
        check_native_wheel.check_wheel(
            str(wheel), "cp310", "abi3", "manylinux_2_17_x86_64", elf_machine=62, version="2.0.9"
        )


def test_rejects_missing_extension(tmp_path):
    wheel = _build_wheel(
        tmp_path, platform="manylinux_2_17_x86_64", payload=_elf(62), include_extension=False
    )
    with pytest.raises(ValueError, match="missing required entries"):
        check_native_wheel.check_wheel(
            str(wheel), "cp310", "abi3", "manylinux_2_17_x86_64", elf_machine=62
        )


def test_rejects_missing_entry_point(tmp_path):
    wheel = _build_wheel(
        tmp_path, platform="manylinux_2_17_x86_64", payload=_elf(62), entry_point="other=other:main"
    )
    with pytest.raises(ValueError, match="entry_points.txt"):
        check_native_wheel.check_wheel(
            str(wheel), "cp310", "abi3", "manylinux_2_17_x86_64", elf_machine=62
        )


def test_rejects_stray_compiled_module(tmp_path):
    wheel = _build_wheel(
        tmp_path,
        platform="manylinux_2_17_x86_64",
        payload=_elf(62),
        stray_module="ytm_player/helpers.abi3.so",
    )
    with pytest.raises(ValueError, match="unexpected compiled modules"):
        check_native_wheel.check_wheel(
            str(wheel), "cp310", "abi3", "manylinux_2_17_x86_64", elf_machine=62
        )


def test_requires_exactly_one_binary_check(tmp_path):
    wheel = _build_wheel(tmp_path, platform="manylinux_2_17_x86_64", payload=_elf(62))
    with pytest.raises(ValueError, match="exactly one"):
        check_native_wheel.check_wheel(str(wheel), "cp310", "abi3", "manylinux_2_17_x86_64")
    with pytest.raises(ValueError, match="exactly one"):
        check_native_wheel.check_wheel(
            str(wheel), "cp310", "abi3", "manylinux_2_17_x86_64", elf_machine=62, pe_machine=0x8664
        )


def test_main_accepts_valid_wheel(tmp_path, capsys):
    wheel = _build_wheel(
        tmp_path, platform="manylinux2014_x86_64.manylinux_2_17_x86_64", payload=_elf(62)
    )
    rc = check_native_wheel.main(
        [
            "script",
            str(wheel),
            "--python-tag",
            "cp310",
            "--abi-tag",
            "abi3",
            "--platform-tag",
            "manylinux2014_x86_64.manylinux_2_17_x86_64",
            "--elf-machine",
            "62",
            "--version",
            "2.1.0",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "OK" in captured.out


def test_main_reads_expected_version_from_checkout(tmp_path, monkeypatch, capsys):
    """Without --version the expected version comes from the src/ checkout."""
    checkout = tmp_path / "checkout"
    (checkout / "src" / "ytm_player").mkdir(parents=True)
    (checkout / "src" / "ytm_player" / "__init__.py").write_text(
        '__version__ = "2.1.0"\n', encoding="utf-8"
    )
    wheel = _build_wheel(tmp_path, platform="manylinux_2_17_x86_64", payload=_elf(62))
    monkeypatch.chdir(checkout)

    rc = check_native_wheel.main(
        [
            "script",
            str(wheel),
            "--python-tag",
            "cp310",
            "--abi-tag",
            "abi3",
            "--platform-tag",
            "manylinux_2_17_x86_64",
            "--elf-machine",
            "62",
        ]
    )
    assert rc == 0
    assert "OK" in capsys.readouterr().out

    drift = _build_wheel(
        tmp_path, platform="manylinux_2_17_x86_64", payload=_elf(62), version="9.9.9"
    )
    rc = check_native_wheel.main(
        [
            "script",
            str(drift),
            "--python-tag",
            "cp310",
            "--abi-tag",
            "abi3",
            "--platform-tag",
            "manylinux_2_17_x86_64",
            "--elf-machine",
            "62",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "drifted apart" in captured.err


def test_main_errors_without_version_or_checkout(tmp_path, monkeypatch, capsys):
    wheel = _build_wheel(tmp_path, platform="manylinux_2_17_x86_64", payload=_elf(62))
    monkeypatch.chdir(tmp_path)  # no src/ytm_player/__init__.py here

    rc = check_native_wheel.main(
        [
            "script",
            str(wheel),
            "--python-tag",
            "cp310",
            "--abi-tag",
            "abi3",
            "--platform-tag",
            "manylinux_2_17_x86_64",
            "--elf-machine",
            "62",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "no --version" in captured.err


def test_main_reports_errors_on_stderr(tmp_path, capsys):
    wheel = _build_wheel(tmp_path, platform="manylinux_2_17_x86_64", payload=_elf(183))
    rc = check_native_wheel.main(
        [
            "script",
            str(wheel),
            "--python-tag",
            "cp310",
            "--abi-tag",
            "abi3",
            "--platform-tag",
            "manylinux_2_17_x86_64",
            "--elf-machine",
            "62",
            "--version",
            "2.1.0",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "::error::" in captured.err
