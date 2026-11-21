#!/usr/bin/env python3
"""Verify that a binary wheel of ytm-player carries the correct tags and payload.

Used by the Publish and CI workflows after every ``maturin build``. Wheel
tagging is the one step of binary packaging that fails silently: a wheel
renamed to claim a platform it was not built for installs without complaint
and crashes at import time on the user's machine — the exact failure mode
this project's Rust transition could not afford after moving the published
wheels from pure-Python to per-platform binaries. Errors loudly (exit 1) on
any mismatch.

Checks performed on the wheel at PATH:
  - the file name is ``ytm_player-<version>-<python>-<abi>-<platform>.whl``,
    with <version> read from the wheel's own METADATA (and compared to
    --version when given: the maturin wheels take their version from
    native/Cargo.toml, so drift from the release tag must fail here, before
    upload),
  - the WHEEL metadata's ``Tag:`` header equals the file name's tag and
    ``Root-Is-Purelib`` is false,
  - the compiled extension is present under ``ytm_player/utils/`` with the
    file name CPython requires on the platform (``.abi3.so`` on Linux and
    macOS, ``.pyd`` on Windows — the same names native/build.sh produces),
  - the extension's binary header proves the machine type: ELF e_machine
    (--elf-machine), a two-slice Mach-O fat binary with the x86_64 and
    arm64 slices (--macho-universal2) or a PE machine type (--pe-machine),
  - the ``ytm`` console script, ``py.typed`` and the extension's ``.pyi``
    stub are all packaged.

Usage:
    check_native_wheel.py WHEEL --python-tag cp310 --abi-tag abi3 \
        --platform-tag manylinux2014_x86_64.manylinux_2_17_x86_64 \
        --elf-machine 62 [--version 2.1.0]
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import zipfile
from pathlib import Path

DIST_NAME = "ytm-player"
PACKAGE = "ytm_player"
MODULE_BASE = "ytm_player/utils/_ytm_player_native"
ENTRY_POINT = "ytm=ytm_player.cli:main"

# Mach-O fat-header CPU types (-big-endian- in the fat header).
CPU_TYPE_X86_64 = 0x01000007
CPU_TYPE_ARM64 = 0x0100000C


def _parse_metadata(raw: str) -> dict[str, str]:
    """The single-value METADATA headers (Name, Version, ...)."""
    headers: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            break
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip()] = value.strip()
    return headers


def _parse_wheel_metadata(raw: str) -> dict[str, list[str]]:
    """The WHEEL file's headers; repeated headers (``Tag``) collect their values."""
    headers: dict[str, list[str]] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            headers.setdefault(key.strip(), []).append(value.strip())
    return headers


def _elf_machine(data: bytes) -> int:
    if data[:4] != b"\x7fELF":
        raise ValueError("extension is not an ELF binary")
    return struct.unpack("<H", data[18:20])[0]


def _macho_fat_cputypes(data: bytes) -> set[int]:
    if data[:4] != b"\xca\xfe\xba\xbe":
        raise ValueError("extension is not a Mach-O fat binary")
    (count,) = struct.unpack(">I", data[4:8])
    types = set()
    for i in range(count):
        entry = data[8 + 20 * i : 8 + 20 * (i + 1)]
        if len(entry) < 8:
            raise ValueError("truncated Mach-O fat header")
        (cputype,) = struct.unpack(">i", entry[:4])
        types.add(cputype)
    return types


def _pe_machine(data: bytes) -> int:
    if data[:2] != b"MZ":
        raise ValueError("extension is not a PE binary")
    (pe_offset,) = struct.unpack("<I", data[0x3C:0x40])
    if data[pe_offset : pe_offset + 4] != b"PE\x00\x00":
        raise ValueError("extension has no PE signature")
    return struct.unpack("<H", data[pe_offset + 4 : pe_offset + 6])[0]


def check_wheel(
    path: str,
    python_tag: str,
    abi_tag: str,
    platform_tag: str,
    *,
    elf_machine: int | None = None,
    macho_universal2: bool = False,
    pe_machine: int | None = None,
    version: str | None = None,
) -> None:
    """Raise ``ValueError`` describing the first failed check."""
    binary_checks = [elf_machine is not None, macho_universal2, pe_machine is not None]
    if sum(binary_checks) != 1:
        raise ValueError(
            "exactly one of --elf-machine, --macho-universal2, --pe-machine is required"
        )

    tag = f"{python_tag}-{abi_tag}-{platform_tag}"
    extension_name = f"{MODULE_BASE}.pyd" if pe_machine is not None else f"{MODULE_BASE}.abi3.so"

    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())

        try:
            metadata = _parse_metadata(
                wheel.read(f"{PACKAGE}-{_dist_info_version(wheel)}.dist-info/METADATA").decode(
                    "utf-8"
                )
            )
        except KeyError as exc:
            raise ValueError("wheel has no METADATA") from exc

        wheel_version = metadata.get("Version", "")
        if metadata.get("Name") != DIST_NAME:
            raise ValueError(f"METADATA Name is {metadata.get('Name')!r}, expected {DIST_NAME!r}")
        if version is not None and wheel_version != version:
            raise ValueError(
                f"wheel version {wheel_version!r} does not match the expected {version!r} "
                "(native/Cargo.toml and the release tag have drifted apart)"
            )

        expected_name = f"{PACKAGE}-{wheel_version}-{tag}.whl"
        base_name = path.replace("\\", "/").rsplit("/", 1)[-1]
        if base_name != expected_name:
            raise ValueError(f"wheel is named {base_name!r}, expected {expected_name!r}")

        dist_info = f"{PACKAGE}-{wheel_version}.dist-info"
        wheel_headers = _parse_wheel_metadata(wheel.read(f"{dist_info}/WHEEL").decode("utf-8"))
        # A compressed platform tag (``manylinux2014_x86_64.manylinux_2_17_x86_64``)
        # names several equivalent tags; the WHEEL metadata must list each of
        # them on its own ``Tag:`` line and nothing else.
        expected_tags = {f"{python_tag}-{abi_tag}-{part}" for part in platform_tag.split(".")}
        actual_tags = set(wheel_headers.get("Tag", []))
        if actual_tags != expected_tags:
            raise ValueError(
                f"WHEEL Tag headers are {sorted(actual_tags)}, expected {sorted(expected_tags)}"
            )
        if wheel_headers.get("Root-Is-Purelib") != ["false"]:
            raise ValueError(
                "WHEEL Root-Is-Purelib is not 'false' — a binary wheel must not be purelib"
            )

        missing = [
            name
            for name in (
                f"{PACKAGE}/__init__.py",
                f"{PACKAGE}/cli.py",
                f"{PACKAGE}/py.typed",
                f"{MODULE_BASE}.pyi",
                extension_name,
                f"{dist_info}/entry_points.txt",
                f"{dist_info}/RECORD",
            )
            if name not in names
        ]
        if missing:
            raise ValueError(f"wheel is missing required entries: {', '.join(missing)}")

        if ENTRY_POINT not in wheel.read(f"{dist_info}/entry_points.txt").decode("utf-8"):
            raise ValueError(f"entry_points.txt does not register {ENTRY_POINT!r}")

        binaries = {name for name in names if name.endswith((".so", ".pyd"))}
        if binaries != {extension_name}:
            raise ValueError(f"unexpected compiled modules in the wheel: {sorted(binaries)}")

        payload = wheel.read(extension_name)
        if elf_machine is not None:
            actual = _elf_machine(payload)
            if actual != elf_machine:
                raise ValueError(f"extension ELF e_machine is {actual}, expected {elf_machine}")
        elif macho_universal2:
            cputypes = _macho_fat_cputypes(payload)
            expected_cpus = {CPU_TYPE_X86_64, CPU_TYPE_ARM64}
            if cputypes != expected_cpus:
                raise ValueError(
                    "extension is not a two-slice x86_64+arm64 universal binary "
                    f"(found CPU types {sorted(hex(c) for c in cputypes)})"
                )
        else:
            assert pe_machine is not None
            actual = _pe_machine(payload)
            if actual != pe_machine:
                raise ValueError(f"extension PE machine is {actual:#x}, expected {pe_machine:#x}")


def _dist_info_version(wheel: zipfile.ZipFile) -> str:
    """The version embedded in the wheel's dist-info directory name."""
    suffix = ".dist-info/METADATA"
    for name in wheel.namelist():
        if name.startswith(PACKAGE + "-") and name.endswith(suffix):
            return name[len(PACKAGE) + 1 : -len(suffix)]
    raise ValueError("wheel has no dist-info/METADATA")


def _checkout_version() -> str | None:
    """__version__ from the checkout the script runs in, if there is one."""
    marker = Path("src/ytm_player/__init__.py")
    if not marker.is_file():
        return None
    match = re.search(r'__version__\s*=\s*"([^"]+)"', marker.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", help="path to the wheel to verify")
    parser.add_argument("--python-tag", required=True, help="e.g. cp310")
    parser.add_argument("--abi-tag", required=True, help="e.g. abi3")
    parser.add_argument(
        "--platform-tag", required=True, help="e.g. manylinux2014_x86_64.manylinux_2_17_x86_64"
    )
    parser.add_argument(
        "--elf-machine", type=int, help="expected ELF e_machine (62 = x86_64, 183 = aarch64)"
    )
    parser.add_argument(
        "--macho-universal2",
        action="store_true",
        help="expect a two-slice x86_64+arm64 Mach-O universal binary",
    )
    parser.add_argument(
        "--pe-machine",
        type=lambda value: int(value, 0),
        help="expected PE machine (0x8664 = amd64)",
    )
    parser.add_argument(
        "--version",
        help="the release version the wheel must carry; defaults to __version__ "
        "from the src/ checkout the script runs in",
    )
    args = parser.parse_args(argv[1:])

    version = args.version
    if version is None:
        version = _checkout_version()
        if version is None:
            print(
                "::error::no --version given and no src/ytm_player/__init__.py "
                "found to read __version__ from",
                file=sys.stderr,
            )
            return 1

    try:
        check_wheel(
            args.wheel,
            args.python_tag,
            args.abi_tag,
            args.platform_tag,
            elf_machine=args.elf_machine,
            macho_universal2=args.macho_universal2,
            pe_machine=args.pe_machine,
            version=version,
        )
    except (ValueError, zipfile.BadZipFile) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    print(f"OK: {args.wheel.rsplit('/', 1)[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
