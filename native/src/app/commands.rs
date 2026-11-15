//! Native cores for `src/ytm_player/app/_commands.py` — the IPC command
//! catalog.

use pyo3::prelude::*;

/// The command palette catalog behind `YTMCommandProvider` from
/// `app/_commands.py` — the pure core of the command provider.
///
/// Mirrors the pure-Python reference (`_command_catalog`): the
/// `(name, help)` pair for every ytm-player command, in dispatch
/// order — the data both the discovery list and the query search
/// walk. The names are the palette's dispatch keys (the Python side
/// resolves each to its host callback through `_command_callback`),
/// so they must stay unique and in sync with the resolvers. A fresh
/// list of fresh tuples comes back on every call — nothing is cached,
/// so no shared mutable state can ever leak between calls.
#[pyfunction]
pub(crate) fn command_catalog() -> Vec<(String, String)> {
    vec![(
        "Theme: Set Current as Default".to_string(),
        "Save the active theme to config.toml".to_string(),
    )]
}
