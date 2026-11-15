//! Native cores for `src/ytm_player/app/_navigation.py` — the nav stack
//! resolvers and dead-playlist-reference purges.

use crate::ui::pages::recently_played::shallow_copy_dict;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet};

// ---------------------------------------------------------------------------
// Navigation stack engine (mirrors ytm_player/app/_navigation.py)
// ---------------------------------------------------------------------------

/// One back/forward stack entry: `(page_name, kwargs)`. The kwargs dict is
/// an opaque payload carried by reference — a popped entry's dict flows to
/// the mounted page exactly like the fallback's, while the snapshots built
/// along the way are fresh dicts.
type NavEntry = (String, Py<PyDict>);

/// Keep the last *max* entries, mirroring Python's `stack[-max:]`. A *max*
/// of 0 keeps everything, exactly like `x[-0:]` (== `x[0:]`).
fn cap_stack<T>(stack: &mut Vec<T>, max: usize) {
    if max > 0 && stack.len() > max {
        stack.drain(..stack.len() - max);
    }
}

/// `dict(current_kwargs)` + `.update(cache.get(current_page, {}))` — the
/// snapshot of the page being left that gets pushed onto a stack: a fresh
/// dict of the constructor kwargs with the page's cached nav state applied
/// over them. The `update` call goes through the method (not a Rust-side
/// item copy) so any mapping value behaves exactly like the fallback's
/// `.update` — cached nav states are always dicts.
fn leaving_snapshot<'py>(
    py: Python<'py>,
    current_kwargs: &Bound<'py, PyDict>,
    cache: &Bound<'py, PyDict>,
    current_page: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let snap = shallow_copy_dict(py, current_kwargs)?;
    if let Some(state) = cache.get_item(current_page)? {
        snap.call_method1("update", (state,))?;
    }
    Ok(snap)
}

/// Assemble the engine outcome dict. The new state is included on every
/// action — a no-op still returns the (cache-refreshed, possibly popped)
/// stacks, because the cache refresh and any stack pop happen before the
/// request is validated and the caller must always apply them.
#[allow(clippy::too_many_arguments)]
fn nav_outcome<'py>(
    py: Python<'py>,
    action: &str,
    reason: Option<&str>,
    page: &str,
    kwargs: Py<PyDict>,
    nav: Vec<NavEntry>,
    forward: Vec<NavEntry>,
    cache: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("action", action)?;
    match reason {
        Some(reason) => {
            out.set_item("reason", reason)?;
        }
        None => {
            out.set_item("reason", pyo3::types::PyNone::get(py))?;
        }
    }
    out.set_item("page", page)?;
    out.set_item("kwargs", kwargs)?;
    out.set_item("nav_stack", PyList::new(py, nav)?)?;
    out.set_item("forward_stack", PyList::new(py, forward)?)?;
    out.set_item("page_state_cache", cache)?;
    Ok(out)
}

/// Resolve one navigation request against the back/forward stacks — the
/// pure stack machine behind `NavigationMixin.navigate_to` from
/// `app/_navigation.py`, transcribed verbatim from its bookkeeping.
///
/// Mirrors the pure-Python reference (`resolve_navigation` in
/// `_navigation.py`) case for case:
///
/// - The leaving page's fresh *current_page_state* (Python `None` = no
///   mounted page / no `get_nav_state`: the cache is left alone) is
///   folded into a copy of *page_state_cache* first — truthy state
///   replaces the cached entry, empty state drops it — because the
///   stack snapshots below read that cache back.
/// - `back` pops the nav stack (pushing the current page's snapshot onto
///   the forward stack); `forward` does the reverse. `back` with an empty
///   stack goes to `library`; `forward` with an empty stack is a silent
///   no-op.
/// - Clicking the page you're already on (empty *kwargs*) pops back
///   instead; with nothing to pop it's a silent no-op.
/// - Any non-back/forward navigation pushes the current page's snapshot
///   onto the nav stack and invalidates the forward history — you can't
///   redo a future you didn't take.
/// - A switch with empty kwargs restores the target's cached state.
/// - A page not in *page_names* is a no-op (after the back/forward
///   handling, so a dead name popped from a stack is caught here too).
///
/// Inputs are never mutated; the returned `nav_stack` / `forward_stack` /
/// `page_state_cache` are fresh containers whose payload dicts are the
/// input objects themselves (snapshots and restored kwargs are new dicts,
/// exactly like the fallback). The caller passes `page_names` /
/// `max_stack` so the Python module's constants stay the single source
/// of truth.
///
/// Returns `{action, reason, page, kwargs, nav_stack, forward_stack,
/// page_state_cache}`: `action` is `"switch"` or `"noop"`, `reason` is
/// `"unknown_page"` / `"no_forward_history"` / `"same_page_no_history"`
/// for a no-op and None for a switch.
#[pyfunction(signature = (
    current_page,
    current_kwargs,
    nav_stack,
    forward_stack,
    page_state_cache,
    current_page_state,
    page_name,
    kwargs,
    page_names,
    max_stack,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_navigation<'py>(
    py: Python<'py>,
    current_page: &str,
    current_kwargs: &Bound<'py, PyDict>,
    nav_stack: Vec<NavEntry>,
    forward_stack: Vec<NavEntry>,
    page_state_cache: &Bound<'py, PyDict>,
    current_page_state: &Bound<'py, PyAny>,
    page_name: &str,
    kwargs: &Bound<'py, PyDict>,
    page_names: Vec<String>,
    max_stack: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mut page_name = page_name.to_string();
    let mut kwargs = kwargs.clone().unbind();

    // -- refresh the leaving page's cached nav state ------------------------
    // A truthy fresh state replaces the cached entry; an empty one means
    // the user left whatever the cache remembers — drop it rather than
    // restore it later. None (no mounted page / no get_nav_state) leaves
    // the cache alone.
    let cache = shallow_copy_dict(py, page_state_cache)?;
    if !current_page.is_empty() && !current_page_state.is_none() {
        if current_page_state.is_truthy()? {
            cache.set_item(current_page, current_page_state)?;
        } else if cache.contains(current_page)? {
            cache.del_item(current_page)?;
        }
    }

    let mut nav = nav_stack;
    let mut forward = forward_stack;

    // -- handle "back" / "forward" navigation via stacks --------------------
    let is_back = page_name == "back";
    let is_forward = page_name == "forward";
    if is_back {
        if let Some((prev_page, prev_kwargs)) = nav.pop() {
            // Push the page we're leaving onto the forward stack so the
            // user can come back to it via forward.
            if !current_page.is_empty() {
                let snap = leaving_snapshot(py, current_kwargs, &cache, current_page)?;
                forward.push((current_page.to_string(), snap.unbind()));
                cap_stack(&mut forward, max_stack);
            }
            page_name = prev_page;
            kwargs = prev_kwargs;
        } else {
            page_name = "library".to_string();
        }
    } else if is_forward {
        if let Some((next_page, next_kwargs)) = forward.pop() {
            // Push the page we're leaving back onto the nav stack so the
            // user can return to it via back.
            if !current_page.is_empty() {
                let snap = leaving_snapshot(py, current_kwargs, &cache, current_page)?;
                nav.push((current_page.to_string(), snap.unbind()));
                cap_stack(&mut nav, max_stack);
            }
            page_name = next_page;
            kwargs = next_kwargs;
        } else {
            // No forward history — silently no-op.
            return nav_outcome(
                py,
                "noop",
                Some("no_forward_history"),
                &page_name,
                kwargs,
                nav,
                forward,
                cache,
            );
        }
    }

    if !page_names.contains(&page_name) {
        return nav_outcome(
            py,
            "noop",
            Some("unknown_page"),
            &page_name,
            kwargs,
            nav,
            forward,
            cache,
        );
    }

    // -- same-page click pops back instead ----------------------------------
    let mut same_page_back = false;
    if page_name == current_page && !kwargs.bind(py).is_truthy()? {
        if let Some((prev_page, prev_kwargs)) = nav.pop() {
            page_name = prev_page;
            kwargs = prev_kwargs;
            same_page_back = true;
        } else {
            return nav_outcome(
                py,
                "noop",
                Some("same_page_no_history"),
                &page_name,
                kwargs,
                nav,
                forward,
                cache,
            );
        }
    }

    // -- push current page onto the nav stack before switching --------------
    // Skip for back / forward / same-page-back — those already managed the
    // stacks above (or popped, in the same-page case).
    if !is_back
        && !is_forward
        && !same_page_back
        && !current_page.is_empty()
        && current_page != page_name
    {
        let snap = leaving_snapshot(py, current_kwargs, &cache, current_page)?;
        nav.push((current_page.to_string(), snap.unbind()));
        cap_stack(&mut nav, max_stack);
        // Browser semantics: any non-back/forward navigation invalidates
        // the forward history (you can't redo a future you didn't take).
        forward.clear();
    }

    // -- restore cached state when no explicit kwargs were given ------------
    if !kwargs.bind(py).is_truthy()? {
        if let Some(state) = cache.get_item(page_name.as_str())? {
            // `dict(cache[page_name])` — a fresh shallow copy; the cached
            // entry itself stays in the cache.
            let restored = PyDict::new(py);
            restored.call_method1("update", (state,))?;
            kwargs = restored.unbind();
        }
    }

    nav_outcome(py, "switch", None, &page_name, kwargs, nav, forward, cache)
}

/// The dead-playlist predicate of the purge: a Library entry carries
/// `playlist_id`, a playlist context page (opened from search/browse)
/// carries `context_type` + `context_id`. Set membership goes through the
/// Python set object (hash/eq semantics), so `None` and missing values
/// match nothing, exactly like the fallback's `in dead_ids`.
fn nav_entry_is_dead(
    py: Python<'_>,
    dead_ids: &Bound<'_, PySet>,
    kw: &Bound<'_, PyDict>,
) -> PyResult<bool> {
    let _ = py;
    if let Some(playlist_id) = kw.get_item("playlist_id")? {
        if dead_ids.contains(playlist_id)? {
            return Ok(true);
        }
    }
    if let Some(context_type) = kw.get_item("context_type")? {
        if context_type.eq("playlist")? {
            if let Some(context_id) = kw.get_item("context_id")? {
                if dead_ids.contains(context_id)? {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Drop every navigation reference to the playlists in *dead_ids* — the
/// pure half of `NavigationMixin._purge_playlist_nav_state` from
/// `app/_navigation.py`. The `library` cache entry and the stack entries
/// naming a deleted playlist (via `playlist_id` — Library pages — or
/// `context_type` + `context_id` — playlist context pages) plus the
/// bare-Library fallback id must all go, or a bare footer-nav to Library
/// would reopen the deleted playlist.
///
/// Mirrors the pure-Python reference (`purge_dead_playlist_refs` in
/// `_navigation.py`) case for case; inputs are never mutated. Returns
/// `{active_library_playlist_id, nav_stack, forward_stack,
/// page_state_cache}`.
#[pyfunction]
pub(crate) fn purge_dead_playlist_refs<'py>(
    py: Python<'py>,
    dead_ids: &Bound<'py, PySet>,
    active_library_playlist_id: &Bound<'py, PyAny>,
    nav_stack: Vec<NavEntry>,
    forward_stack: Vec<NavEntry>,
    page_state_cache: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    // LibraryPage.on_mount falls back to this id when mounted bare. The
    // membership test goes through the Python set (hash/eq semantics), so
    // `None in dead_ids` matches nothing and a non-dead id is carried
    // through unchanged — the original object, like the fallback.
    let active: Option<Py<PyAny>> = if dead_ids.contains(active_library_playlist_id)? {
        None
    } else {
        Some(active_library_playlist_id.clone().unbind())
    };

    let cache = shallow_copy_dict(py, page_state_cache)?;
    if let Some(lib_state) = cache.get_item("library")? {
        if lib_state.is_truthy()? {
            // `lib_state.get("playlist_id") in dead_ids` — the get() call
            // goes through the attribute, like the fallback.
            let playlist_id = lib_state.call_method1("get", ("playlist_id",))?;
            if dead_ids.contains(playlist_id)? {
                cache.del_item("library")?;
            }
        }
    }

    let mut nav: Vec<NavEntry> = Vec::with_capacity(nav_stack.len());
    for (name, kw) in &nav_stack {
        if !nav_entry_is_dead(py, dead_ids, kw.bind(py))? {
            nav.push((name.clone(), kw.clone_ref(py)));
        }
    }
    let mut forward: Vec<NavEntry> = Vec::with_capacity(forward_stack.len());
    for (name, kw) in &forward_stack {
        if !nav_entry_is_dead(py, dead_ids, kw.bind(py))? {
            forward.push((name.clone(), kw.clone_ref(py)));
        }
    }

    let out = PyDict::new(py);
    match active {
        Some(id) => {
            out.set_item("active_library_playlist_id", id)?;
        }
        None => {
            out.set_item("active_library_playlist_id", pyo3::types::PyNone::get(py))?;
        }
    }
    out.set_item("nav_stack", PyList::new(py, nav)?)?;
    out.set_item("forward_stack", PyList::new(py, forward)?)?;
    out.set_item("page_state_cache", cache)?;
    Ok(out)
}
