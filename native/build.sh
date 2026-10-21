#!/usr/bin/env bash
# Build the Rust/PyO3 native extension and copy it into the Python package.
set -euo pipefail

cd "$(dirname "$0")"

export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cargo build --release --locked

cp target/release/libformatting_native.so ../src/ytm_player/utils/_formatting_native.abi3.so
echo "Native extension built at src/ytm_player/utils/_formatting_native.abi3.so"
