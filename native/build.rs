//! Build script for the PyO3 extension.
//!
//! `pyo3`'s own build script emits its cdylib link args for *pyo3's
//! package compilation*, which never reach the final `cdylib` link of
//! THIS crate — the extension module itself. On macOS that link needs
//! `-undefined dynamic_lookup` (the Py* symbols stay undefined and are
//! resolved by the host interpreter at load time, the same way Python
//! extension modules have always linked there), and without it a plain
//! `cargo build` of the extension fails with "Undefined symbols for
//! architecture arm64: _PyBaseObject_Type, ..." — every Py* symbol
//! unresolved.
//!
//! maturin passes the flag itself, which is why the `wheels`/`publish`
//! workflows and `maturin build` never hit this; the bare-cargo build
//! paths — `native/build.sh` (CI's `native` job, the AUR PKGBUILD),
//! Nix's `buildRustPackage` and a developer's local `cargo build` —
//! do not. `pyo3_build_config::add_extension_module_link_args()` emits
//! the `cargo:rustc-cdylib-link-arg` directives for exactly this
//! (Darwin only; a no-op on every other platform), and must run from
//! THIS crate's build script so the args apply to our cdylib link.
//!
//! The build-dependency version must track `pyo3`'s (0.23) — bump them
//! together.
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
