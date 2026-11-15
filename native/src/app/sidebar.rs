//! Native cores for `src/ytm_player/app/_sidebar.py` — pane/sidebar
//! visibility state and pane-focus cycling.

use crate::ui::pages::recently_played::shallow_copy_dict;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Sidebar pane-routing state machine (mirrors ytm_player/app/_sidebar.py)
// ---------------------------------------------------------------------------

/// The pane identifiers, in the layout's left-to-right order
/// `[playlists | content | lyrics]` — the Python module's `PANE_*`
/// constants.
pub(crate) const PANE_PLAYLISTS: &str = "playlists";

const PANE_CONTENT: &str = "content";

pub(crate) const PANE_LYRICS: &str = "lyrics";

/// The DOM ids of the three pane containers; a focused widget's ancestor
/// chain is scanned for these, exactly like the fallback's `node_id ==`
/// comparisons.
const ID_PLAYLIST_SIDEBAR: &str = "playlist-sidebar";

const ID_LYRICS_SIDEBAR: &str = "lyrics-sidebar";

const ID_MAIN_CONTENT: &str = "main-content";

/// The page key a sidebar-visibility read/toggle/ensure resolves to: an
/// empty page (no page mounted yet) means Library, exactly like the
/// fallback's `page or "library"`.
fn sidebar_page_key(page: &str) -> &str {
    if page.is_empty() {
        "library"
    } else {
        page
    }
}

/// `per_page.get(key, default)` gated by Python truthiness — the current
/// visibility flag for the read/toggle/ensure decisions. A stored value
/// keeps its own truthiness (a hand-edited session.json may hold `1` or
/// `""`), and a missing entry reads the default's.
fn sidebar_page_visible(
    per_page: &Bound<'_, PyDict>,
    key: &str,
    default: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    match per_page.get_item(key)? {
        Some(value) => value.is_truthy(),
        None => default.is_truthy(),
    }
}

/// The playlist-sidebar visibility stored for a page — the read behind
/// `SidebarMixin._is_playlist_sidebar_visible` (and the toggle/ensure
/// bookkeeping) from `app/_sidebar.py`.
///
/// Mirrors the pure-Python reference (`sidebar_visible_for_page`): an
/// empty *page* (no page mounted yet) reads Library. The stored value —
/// or *default* when the page has no entry — is returned raw, so a truthy
/// non-bool stored value (a hand-edited session.json) flows on with its
/// own truthiness, like the reference's `.get`.
#[pyfunction]
pub(crate) fn sidebar_visible_for_page<'py>(
    per_page: &Bound<'py, PyDict>,
    page: &str,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match per_page.get_item(sidebar_page_key(page))? {
        Some(value) => Ok(value),
        None => Ok(default.clone()),
    }
}

/// Flip the playlist-sidebar visibility stored for a page — the pure half
/// of `SidebarMixin._toggle_playlist_sidebar` from `app/_sidebar.py`.
///
/// Mirrors the pure-Python reference (`toggle_sidebar_state`): an empty
/// *page* means Library; the new state is the negation of the current one
/// (*default* when the page has no entry, Python truthiness deciding) and
/// lands in a fresh copy of *per_page* — inputs are never mutated.
/// Returns `{"visible": bool, "sidebar_per_page": dict}`.
#[pyfunction]
pub(crate) fn toggle_sidebar_state<'py>(
    py: Python<'py>,
    per_page: &Bound<'py, PyDict>,
    page: &str,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let key = sidebar_page_key(page);
    let new_state = !sidebar_page_visible(per_page, key, default)?;
    let fresh = shallow_copy_dict(py, per_page)?;
    fresh.set_item(key, new_state)?;
    let out = PyDict::new(py);
    out.set_item("visible", new_state)?;
    out.set_item("sidebar_per_page", fresh)?;
    Ok(out)
}

/// Make the playlist-sidebar entry for a page visible, if it isn't — the
/// pure half of `SidebarMixin._ensure_playlist_sidebar_visible` from
/// `app/_sidebar.py`.
///
/// Mirrors the pure-Python reference (`ensure_sidebar_visible`): when the
/// current value (*default* for a missing entry, Python truthiness
/// deciding) is falsy, a fresh copy of *per_page* gets `True` written for
/// the page; otherwise the map is returned unchanged (still a fresh copy,
/// like every engine outcome). Returns `{"changed": bool,
/// "sidebar_per_page": dict}`.
#[pyfunction]
pub(crate) fn ensure_sidebar_visible<'py>(
    py: Python<'py>,
    per_page: &Bound<'py, PyDict>,
    page: &str,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let key = sidebar_page_key(page);
    let changed = !sidebar_page_visible(per_page, key, default)?;
    let fresh = shallow_copy_dict(py, per_page)?;
    if changed {
        fresh.set_item(key, true)?;
    }
    let out = PyDict::new(py);
    out.set_item("changed", changed)?;
    out.set_item("sidebar_per_page", fresh)?;
    Ok(out)
}

/// The visible panes in left-to-right order — the list behind
/// `SidebarMixin._visible_panes` (and the cycle movement) from
/// `app/_sidebar.py`. Content is always present; the Playlists sidebar
/// leads when its per-page visibility says so and the lyrics pane closes
/// the row while it is open. Both flags are read with Python truthiness
/// (`if flag:`), so a truthy non-bool visibility — the raw stored value
/// from a hand-edited session.json — includes its pane, like the
/// fallback.
#[pyfunction]
pub(crate) fn visible_panes<'py>(
    playlist_sidebar_visible: &Bound<'py, PyAny>,
    lyrics_open: &Bound<'py, PyAny>,
) -> PyResult<Vec<String>> {
    Ok(visible_panes_list(
        playlist_sidebar_visible.is_truthy()?,
        lyrics_open.is_truthy()?,
    )
    .into_iter()
    .map(String::from)
    .collect())
}

/// The shared body of `visible_panes` — panes as static strs so the
/// movement decisions can index into them without reallocation.
fn visible_panes_list(playlist_sidebar_visible: bool, lyrics_open: bool) -> Vec<&'static str> {
    let mut panes: Vec<&'static str> = Vec::with_capacity(3);
    if playlist_sidebar_visible {
        panes.push(PANE_PLAYLISTS);
    }
    panes.push(PANE_CONTENT);
    if lyrics_open {
        panes.push(PANE_LYRICS);
    }
    panes
}

/// `getattr(node, name, None)` — None for a missing attribute or an
/// explicit None value, so a widget without the attribute reads as
/// id-less / parent-less. Any other error propagates, exactly like the
/// fallback's three-argument `getattr`.
fn attr_or_none<'py>(node: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    match node.getattr(name) {
        Ok(value) => Ok(if value.is_none() { None } else { Some(value) }),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(node.py()) => Ok(None),
        Err(err) => Err(err),
    }
}

/// The pane the focused widget lives in — the pure half of
/// `SidebarMixin._current_pane` from `app/_sidebar.py`.
///
/// Mirrors the pure-Python reference (`pane_of_focused`): walk the
/// focused widget's parent chain (duck-typed `.id` / `.parent` attribute
/// reads, matching the fallback's `getattr(node, ..., None)`) looking for
/// a known pane container (`playlist-sidebar` / `lyrics-sidebar` /
/// `main-content`) and return its pane. The id comparisons go through
/// Python `==`, so a node whose id object defines its own equality
/// resolves exactly like it does on the fallback. A chain with no known
/// container means content — the `_active_pane` hint fallback only
/// applies when nothing is focused, which the mixin handles before
/// calling.
#[pyfunction]
pub(crate) fn pane_of_focused(focused: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut node: Option<Bound<'_, PyAny>> = Some(focused.clone());
    while let Some(current) = node {
        if let Some(node_id) = attr_or_none(&current, "id")? {
            if node_id.eq(ID_PLAYLIST_SIDEBAR)? {
                return Ok(PANE_PLAYLISTS.to_string());
            }
            if node_id.eq(ID_LYRICS_SIDEBAR)? {
                return Ok(PANE_LYRICS.to_string());
            }
            if node_id.eq(ID_MAIN_CONTENT)? {
                return Ok(PANE_CONTENT.to_string());
            }
        }
        node = attr_or_none(&current, "parent")?;
    }
    Ok(PANE_CONTENT.to_string())
}

/// Whether a pane can take keyboard focus, and what to focus — the gate
/// behind `SidebarMixin._focus_pane` from `app/_sidebar.py`.
///
/// Mirrors the pure-Python reference (`pane_focus_target`): the Playlists
/// pane is always focusable (focusing it auto-shows the sidebar), the
/// lyrics pane only while it is open (*lyrics_open* read with Python
/// truthiness), and anything else — the content pane or an unknown pane
/// name — normalises to content. Returns None when the pane cannot take
/// focus (lyrics while closed), meaning "no move".
#[pyfunction]
pub(crate) fn pane_focus_target<'py>(
    pane: &str,
    lyrics_open: &Bound<'py, PyAny>,
) -> PyResult<Option<String>> {
    if pane == PANE_PLAYLISTS {
        Ok(Some(PANE_PLAYLISTS.to_string()))
    } else if pane == PANE_LYRICS {
        if lyrics_open.is_truthy()? {
            Ok(Some(PANE_LYRICS.to_string()))
        } else {
            Ok(None)
        }
    } else {
        Ok(Some(PANE_CONTENT.to_string()))
    }
}

/// Ctrl+w h — the pane left of the current one. Reaching the Playlists
/// sidebar from the content pane auto-shows it (via the focus gate), so
/// left never fails; lyrics steps to content. Any other pane also goes to
/// Playlists, like the reference's else.
///
/// Mirrors `next_pane_left` from `app/_sidebar.py`.
#[pyfunction]
pub(crate) fn next_pane_left(current_pane: &str) -> String {
    if current_pane == PANE_LYRICS {
        PANE_CONTENT.to_string()
    } else {
        PANE_PLAYLISTS.to_string()
    }
}

/// Ctrl+w l — the pane right of the current one, or None at the edge.
/// Playlists steps to content; content enters the lyrics pane only while
/// it is open (*lyrics_open* read with Python truthiness); lyrics (or an
/// unknown pane) is already at the rightmost reachable position — None
/// means "stay put", and the caller skips the focus move entirely.
///
/// Mirrors `next_pane_right` from `app/_sidebar.py`.
#[pyfunction]
pub(crate) fn next_pane_right<'py>(
    current_pane: &str,
    lyrics_open: &Bound<'py, PyAny>,
) -> PyResult<Option<String>> {
    if current_pane == PANE_PLAYLISTS {
        Ok(Some(PANE_CONTENT.to_string()))
    } else if current_pane == PANE_CONTENT && lyrics_open.is_truthy()? {
        Ok(Some(PANE_LYRICS.to_string()))
    } else {
        Ok(None)
    }
}

/// Ctrl+w w — the next visible pane after the current one, wrapping. The
/// cycle runs through `visible_panes` left-to-right; a current pane that
/// isn't visible (hidden lyrics, an unknown name) is treated as content,
/// so the cycle always has a starting point — content is always in the
/// list. Both visibility flags are read with Python truthiness, like
/// `visible_panes`.
///
/// Mirrors `next_pane_cycle` from `app/_sidebar.py`.
#[pyfunction]
pub(crate) fn next_pane_cycle<'py>(
    current_pane: &str,
    playlist_sidebar_visible: &Bound<'py, PyAny>,
    lyrics_open: &Bound<'py, PyAny>,
) -> PyResult<String> {
    let panes = visible_panes_list(
        playlist_sidebar_visible.is_truthy()?,
        lyrics_open.is_truthy()?,
    );
    let current = if panes.contains(&current_pane) {
        current_pane
    } else {
        PANE_CONTENT
    };
    // Content is always in the list, so the fallback above always finds
    // a starting point.
    let idx = panes
        .iter()
        .position(|pane| *pane == current)
        .expect("content pane is always visible");
    Ok(panes[(idx + 1) % panes.len()].to_string())
}
