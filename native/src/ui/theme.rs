//! Native cores for `src/ytm_player/ui/theme.py` (plus the CSS variable
//! assembly in `app/_app.py`) — theme field extraction from CSS
//! variables, with rich-safe colour parsing.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

// ---------------------------------------------------------------------------
// Theme resolution engines (mirror ytm_player/ui/theme.py and app/_app.py)
// ---------------------------------------------------------------------------

/// Translate a Textual-only ANSI color token into a Rich-parseable name.
/// Mirrors `ytm_player.ui.theme.rich_safe_color`: the `ansi_` prefix is
/// stripped (five ASCII bytes), anything else passes through unchanged.
#[pyfunction]
pub(crate) fn rich_safe_color(color: &str) -> String {
    color.strip_prefix("ansi_").unwrap_or(color).to_string()
}

/// `variables.get(key, fallback)` with the fallback as a fresh Python
/// string — a present key yields the original object, not a copy, exactly
/// like the reference's dict lookups.
fn css_get_or<'py>(
    variables: &Bound<'py, PyDict>,
    key: &str,
    fallback: &str,
) -> PyResult<Bound<'py, PyAny>> {
    match variables.get_item(key)? {
        Some(v) => Ok(v),
        None => Ok(PyString::new(variables.py(), fallback).into_any()),
    }
}

/// `variables.get(key, default)` with the default already built.
fn get_or<'py>(
    variables: &Bound<'py, PyDict>,
    key: &str,
    default: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match variables.get_item(key)? {
        Some(v) => Ok(v),
        None => Ok(default),
    }
}

/// The 21 ThemeColors field values resolved from a CSS-variables dict.
/// Mirrors the pure core of `ThemeColors.from_css_variables`
/// (`ytm_player.ui.theme`): each of the twelve base fields is its CSS
/// variable (note `muted_text` reads `text-muted`) or the dataclass
/// default; the nine app-specific fields fall back one level deeper —
/// the derived-from variable first, that field's own dataclass default
/// last, so the shared keys (`surface`, `text`, `text-muted`, `primary`)
/// carry each field's own innermost constant rather than the base
/// field's; and `lyrics_current` runs the documented three-level chain
/// lyrics-current → accent → primary → DEFAULT_LYRIC_CURRENT. A present
/// value passes through as the object it is — the ANSI-token
/// normalisation is NOT applied here, exactly as the reference's mapping
/// doesn't apply it: `ThemeColors.__post_init__` runs it after
/// construction, on whichever backend built the fields.
#[pyfunction]
pub(crate) fn theme_fields_from_css<'py>(
    py: Python<'py>,
    variables: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    // The twelve base colours: (field, CSS variable, dataclass default).
    const BASE: [(&str, &str, &str); 12] = [
        ("background", "background", "#0f0f0f"),
        ("foreground", "foreground", "#ffffff"),
        ("primary", "primary", "#ff0000"),
        ("secondary", "secondary", "#aaaaaa"),
        ("accent", "accent", "#ff4e45"),
        ("success", "success", "#2ecc71"),
        ("warning", "warning", "#f39c12"),
        ("error", "error", "#e74c3c"),
        ("surface", "surface", "#1a1a1a"),
        ("border", "border", "#333333"),
        ("muted_text", "text-muted", "#999999"),
        ("text", "text", "#ffffff"),
    ];
    let fields = PyDict::new(py);
    for (name, css_var, default) in BASE {
        fields.set_item(name, css_get_or(variables, css_var, default)?)?;
    }

    // The nine app-specific fields, each its own nested chain — the
    // derived-from key's value first (or that field's own default), the
    // field's own innermost constant at the bottom.
    fields.set_item(
        "playback_bar_bg",
        get_or(
            variables,
            "playback-bar-bg",
            css_get_or(variables, "surface", "#1a1a1a")?,
        )?,
    )?;
    fields.set_item(
        "active_tab",
        get_or(
            variables,
            "active-tab",
            css_get_or(variables, "text", "#ffffff")?,
        )?,
    )?;
    fields.set_item(
        "inactive_tab",
        get_or(
            variables,
            "inactive-tab",
            css_get_or(variables, "text-muted", "#999999")?,
        )?,
    )?;
    fields.set_item(
        "selected_item",
        get_or(
            variables,
            "selected-item",
            css_get_or(variables, "surface", "#2a2a2a")?,
        )?,
    )?;
    fields.set_item(
        "progress_filled",
        get_or(
            variables,
            "progress-filled",
            css_get_or(variables, "primary", "#ff0000")?,
        )?,
    )?;
    fields.set_item(
        "progress_empty",
        get_or(
            variables,
            "progress-empty",
            css_get_or(variables, "surface", "#555555")?,
        )?,
    )?;
    fields.set_item(
        "lyrics_played",
        get_or(
            variables,
            "lyrics-played",
            css_get_or(variables, "text-muted", "#999999")?,
        )?,
    )?;
    // The three-level chain: lyrics-current → accent → primary →
    // DEFAULT_LYRIC_CURRENT (the dataclass default of the field itself).
    fields.set_item(
        "lyrics_current",
        get_or(
            variables,
            "lyrics-current",
            get_or(
                variables,
                "accent",
                css_get_or(variables, "primary", "#ff4e45")?,
            )?,
        )?,
    )?;
    fields.set_item(
        "lyrics_upcoming",
        get_or(
            variables,
            "lyrics-upcoming",
            css_get_or(variables, "text", "#aaaaaa")?,
        )?,
    )?;

    Ok(fields)
}

/// The app's CSS-variables overlay on Textual's theme variables. Mirrors
/// the pure core of `YTMPlayerApp.get_css_variables`
/// (`ytm_player.app._app`): the nine app-specific variables default to
/// values derived from the base palette — all evaluated from the
/// pre-patch state, exactly as the reference's dict literal evaluates
/// them before any insertion, each carrying its own innermost constant —
/// and each is inserted only when the theme hasn't set it; then the
/// theme.toml overrides layer on top — every ThemeColors field name
/// mapped to its CSS dash-case variable, in the reference's literal
/// order, user values winning over everything. The `variables` dict is
/// patched in place and returned — the same object the reference mutates
/// — so existing keys keep their position and new ones append in
/// app-default order.
#[pyfunction]
pub(crate) fn app_css_variables<'py>(
    variables: &Bound<'py, PyDict>,
    colors: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let defaults: [(&str, Bound<'py, PyAny>); 9] = [
        (
            "playback-bar-bg",
            css_get_or(variables, "surface", "#1a1a1a")?,
        ),
        ("active-tab", css_get_or(variables, "text", "#ffffff")?),
        (
            "inactive-tab",
            css_get_or(variables, "text-muted", "#999999")?,
        ),
        (
            "selected-item",
            css_get_or(variables, "surface", "#2a2a2a")?,
        ),
        (
            "progress-filled",
            css_get_or(variables, "primary", "#ff0000")?,
        ),
        (
            "progress-empty",
            css_get_or(variables, "surface", "#555555")?,
        ),
        (
            "lyrics-played",
            css_get_or(variables, "text-muted", "#999999")?,
        ),
        (
            "lyrics-current",
            get_or(
                variables,
                "accent",
                css_get_or(variables, "primary", "#ff4e45")?,
            )?,
        ),
        ("lyrics-upcoming", css_get_or(variables, "text", "#aaaaaa")?),
    ];
    for (key, default) in &defaults {
        // `key not in variables` / `variables[key] = default` — the
        // protocol operations the reference spells, so a dict subclass
        // steering __contains__/__setitem__ behaves exactly as it would
        // there (the default computations above stay non-dispatching
        // `.get`s, like the reference's).
        if !variables.as_any().contains(*key)? {
            variables.as_any().set_item(*key, default.clone())?;
        }
    }

    // theme.toml overrides — `if colors:` gates on the object's
    // truthiness (an empty table is no overrides), and each of the 21
    // field names maps to its CSS variable in the reference's literal
    // order. The membership test and the item read are the Python
    // protocol operations the reference spells (`in`, `[]`), so a
    // non-dict mapping would behave exactly as it would there.
    if colors.is_truthy()? {
        let field_to_css: [(&str, &str); 21] = [
            ("background", "background"),
            ("foreground", "foreground"),
            ("primary", "primary"),
            ("secondary", "secondary"),
            ("accent", "accent"),
            ("success", "success"),
            ("warning", "warning"),
            ("error", "error"),
            ("surface", "surface"),
            ("border", "border"),
            ("text", "text"),
            ("muted_text", "text-muted"),
            ("playback_bar_bg", "playback-bar-bg"),
            ("active_tab", "active-tab"),
            ("inactive_tab", "inactive-tab"),
            ("selected_item", "selected-item"),
            ("progress_filled", "progress-filled"),
            ("progress_empty", "progress-empty"),
            ("lyrics_played", "lyrics-played"),
            ("lyrics_current", "lyrics-current"),
            ("lyrics_upcoming", "lyrics-upcoming"),
        ];
        for (field_name, css_name) in &field_to_css {
            if colors.contains(*field_name)? {
                let value = colors.get_item(*field_name)?;
                variables.as_any().set_item(*css_name, value)?;
            }
        }
    }

    Ok(variables.clone())
}

/// The 21 ThemeColors field values from a Textual Theme's raw colours.
/// Mirrors the pure core of `YTMPlayerApp.watch_theme`'s ThemeColors
/// rebuild (`ytm_player.app._app`): the nine raw Theme attributes arrive
/// as plain values (the attribute reads stay in Python, in the
/// reference's first-read order), and every field applies the
/// reference's own `or` chain — a falsy raw value (None or empty) falls
/// back to its per-site constant, which differs between sites sharing a
/// source (`foreground` yields `#ffffff` for itself but `#aaaaaa` for
/// `lyrics_upcoming`; `surface` yields `#1a1a1a`, `#333333` and
/// `#555555` at its three sites); `progress_filled` keeps the raw
/// `primary` as its default even when falsy; and `lyrics_current`'s
/// default side runs the raw accent → primary → DEFAULT_LYRIC_CURRENT
/// chain. The `variables` dict is the Theme's own; a present
/// app-specific key wins over every fallback. As in
/// `theme_fields_from_css`, the values pass through unnormalised —
/// `__post_init__` runs after construction.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the Theme's nine raw colours — one parameter per attribute the reference reads
#[pyo3(signature = (primary, background, foreground, secondary, accent, success, warning, error, surface, variables))]
pub(crate) fn theme_fields_from_theme<'py>(
    py: Python<'py>,
    primary: &Bound<'py, PyAny>,
    background: &Bound<'py, PyAny>,
    foreground: &Bound<'py, PyAny>,
    secondary: &Bound<'py, PyAny>,
    accent: &Bound<'py, PyAny>,
    success: &Bound<'py, PyAny>,
    warning: &Bound<'py, PyAny>,
    error: &Bound<'py, PyAny>,
    surface: &Bound<'py, PyAny>,
    variables: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    /// `value or fallback` — CPython truthiness, the original object when
    /// truthy, a fresh Python string otherwise.
    fn or_fallback<'py>(value: &Bound<'py, PyAny>, fallback: &str) -> PyResult<Bound<'py, PyAny>> {
        if value.is_truthy()? {
            Ok(value.clone())
        } else {
            Ok(PyString::new(value.py(), fallback).into_any())
        }
    }

    let fields = PyDict::new(py);
    fields.set_item("primary", primary.clone())?;
    fields.set_item("background", or_fallback(background, "#0f0f0f")?)?;
    fields.set_item("foreground", or_fallback(foreground, "#ffffff")?)?;
    fields.set_item("secondary", or_fallback(secondary, "#aaaaaa")?)?;
    fields.set_item("accent", or_fallback(accent, "#ff4e45")?)?;
    fields.set_item("success", or_fallback(success, "#2ecc71")?)?;
    fields.set_item("warning", or_fallback(warning, "#f39c12")?)?;
    fields.set_item("error", or_fallback(error, "#e74c3c")?)?;
    fields.set_item("surface", or_fallback(surface, "#1a1a1a")?)?;
    fields.set_item("text", or_fallback(foreground, "#ffffff")?)?;
    fields.set_item("muted_text", or_fallback(secondary, "#999999")?)?;
    fields.set_item("border", or_fallback(surface, "#333333")?)?;
    fields.set_item(
        "playback_bar_bg",
        get_or(
            variables,
            "playback-bar-bg",
            or_fallback(surface, "#1a1a1a")?,
        )?,
    )?;
    fields.set_item(
        "active_tab",
        get_or(variables, "active-tab", or_fallback(foreground, "#ffffff")?)?,
    )?;
    fields.set_item(
        "inactive_tab",
        get_or(
            variables,
            "inactive-tab",
            or_fallback(secondary, "#999999")?,
        )?,
    )?;
    fields.set_item(
        "selected_item",
        get_or(variables, "selected-item", or_fallback(surface, "#2a2a2a")?)?,
    )?;
    fields.set_item(
        "progress_filled",
        get_or(variables, "progress-filled", primary.clone())?,
    )?;
    fields.set_item(
        "progress_empty",
        get_or(
            variables,
            "progress-empty",
            or_fallback(surface, "#555555")?,
        )?,
    )?;
    fields.set_item(
        "lyrics_played",
        get_or(
            variables,
            "lyrics-played",
            or_fallback(secondary, "#999999")?,
        )?,
    )?;
    // The default side runs the raw chain — `t.accent or t.primary or
    // DEFAULT_LYRIC_CURRENT` — note it is the raw accent and primary
    // here, not the `#ff4e45`/or-fallback values the fields above took.
    fields.set_item(
        "lyrics_current",
        get_or(
            variables,
            "lyrics-current",
            if accent.is_truthy()? {
                accent.clone()
            } else if primary.is_truthy()? {
                primary.clone()
            } else {
                PyString::new(py, "#ff4e45").into_any()
            },
        )?,
    )?;
    fields.set_item(
        "lyrics_upcoming",
        get_or(
            variables,
            "lyrics-upcoming",
            or_fallback(foreground, "#aaaaaa")?,
        )?,
    )?;

    Ok(fields)
}
