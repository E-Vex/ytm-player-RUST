#!/usr/bin/env bash
# Build the Rust/PyO3 native extension and copy it into the Python package.
#
# Cargo names a cdylib artifact per platform (lib{name}.so on Linux,
# lib{name}.dylib on macOS, {name}.dll on Windows), and CPython's import
# system accepts different module suffixes per platform too: ".abi3.so"
# is an EXTENSION_SUFFIX on Linux and macOS, but Windows' suffix list is
# ['.cp3NN-win_amd64.pyd', '.pyd'] — there is no ".abi3.pyd" — so the
# Windows module must be a plain ".pyd". Resolve artifact and destination
# per platform so the README's build instructions work on every supported
# OS, not just Linux.
set -euo pipefail

cd "$(dirname "$0")"

export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cargo build --release --locked

case "$(uname -s)" in
    Linux*)
        artifact=libytm_player_native.so
        module=_ytm_player_native.abi3.so
        ;;
    Darwin*)
        artifact=libytm_player_native.dylib
        module=_ytm_player_native.abi3.so
        ;;
    MINGW* | MSYS* | CYGWIN*)
        # Git Bash / MSYS (GitHub's Windows runners and the standard
        # Rust-on-Windows dev setups all provide it).
        artifact=ytm_player_native.dll
        module=_ytm_player_native.pyd
        ;;
    *)
        echo "Unsupported platform for native/build.sh: $(uname -s)" >&2
        exit 1
        ;;
esac

cp "target/release/$artifact" "../src/ytm_player/utils/$module"
echo "Native extension built at src/ytm_player/utils/$module"
