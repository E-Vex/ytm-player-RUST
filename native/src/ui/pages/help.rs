//! Native cores for `src/ytm_player/ui/pages/help.py` — help page rows,
//! incl. CPython `str.title`/`str.capitalize` semantics for keybinding
//! labels.

use crate::ui::sidebars::python_strip;
use crate::utils::pycompat::python_lower;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Help-page rows engine (mirrors ui/pages/help.py)
// ---------------------------------------------------------------------------

/// The titlecase-letter category (Lt) — the 31 characters whose UnicodeData
/// titlecase field is the mixed-case digraph form (ǅ-style). CPython's
/// `ISTITLE` tests exactly this category, and `str.title()`'s
/// previous-is-cased check admits them (see `py_is_cased`); Rust's
/// `char::is_uppercase`/`is_lowercase` (the Uppercase/Lowercase derived
/// properties: Lu+Other_Uppercase / Ll+Other_Lowercase) do not, so the
/// category is spelled out. The set has been stable across every Unicode
/// version CPython 3.10–3.14 ships.
fn is_titlecase_letter(c: char) -> bool {
    matches!(
        c,
        '\u{01C5}'
            | '\u{01C8}'
            | '\u{01CB}'
            | '\u{01F2}'
            | '\u{1F88}'..='\u{1F8F}'
            | '\u{1F98}'..='\u{1F9F}'
            | '\u{1FA8}'..='\u{1FAF}'
            | '\u{1FBC}'
            | '\u{1FCC}'
            | '\u{1FFC}'
    )
}

/// CPython's "cased" test as `str.title()` applies it: the Uppercase
/// derived property (Lu + Other_Uppercase — Rust's `char::is_uppercase`,
/// which is why Roman numerals and circled letters count) ∪ the Lowercase
/// derived property (Ll + Other_Lowercase — `char::is_lowercase`, which
/// is why modifier letters like `ʰ` and ordinal indicators like `ª`
/// count) ∪ the titlecase letters (Lt — see `is_titlecase_letter`).
fn py_is_cased(c: char) -> bool {
    c.is_uppercase() || c.is_lowercase() || is_titlecase_letter(c)
}

/// The Case_Ignorable property: characters a final-sigma scan steps over.
/// Case_Ignorable = Word_Break ∈ {MidLetter, MidNumLet, Single_Quote} ∪
/// general categories {Mn, Me, Cf, Lm}. The general categories come from
/// the regex crate's Unicode tables (`unicode-gencat`); the word-break
/// characters and the modifier symbols the property also claims are the
/// explicit list (verified against CPython's own tables char by char —
/// every member below is Case_Ignorable on every Unicode release CPython
/// 3.10–3.14 ships). A newly-scripted combining mark the regex tables
/// know but an older CPython does not is the one place the two sides can
/// disagree, and only for characters younger than the interpreter's
/// Unicode — never for the ASCII identifiers the help rows carry.
static CASE_IGNORABLE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^[\u{0027}\u{002e}\u{003a}\u{005e}\u{0060}\u{00a8}\u{00af}\u{00b4}",
        r"\u{00b7}\u{00b8}\u{02c2}\u{02c3}\u{02c4}\u{02c5}\u{02d2}\u{02d3}",
        r"\u{02d4}\u{02d5}\u{02d6}\u{02d7}\u{02d8}\u{02d9}\u{02da}\u{02db}",
        r"\u{02dc}\u{02dd}\u{02de}\u{02df}\u{02e5}\u{02e6}\u{02e7}\u{02e8}",
        r"\u{02e9}\u{02ea}\u{02eb}\u{02ed}\u{02ef}\u{02f0}\u{02f1}\u{02f2}",
        r"\u{02f3}\u{02f4}\u{02f5}\u{02f6}\u{02f7}\u{02f8}\u{02f9}\u{02fa}",
        r"\u{02fb}\u{02fc}\u{02fd}\u{02fe}\u{02ff}\u{0375}\u{0384}\u{0385}",
        r"\u{0387}\u{055f}\u{05f4}\u{0888}\u{1fbd}\u{1fbf}\u{1fc0}\u{1fc1}",
        r"\u{1fcd}\u{1fce}\u{1fcf}\u{1fdd}\u{1fde}\u{1fdf}\u{1fed}\u{1fee}",
        r"\u{1fef}\u{1ffd}\u{1ffe}\u{2018}\u{2019}\u{2024}\u{2027}\u{309b}",
        r"\u{309c}\u{a700}\u{a701}\u{a702}\u{a703}\u{a704}\u{a705}\u{a706}",
        r"\u{a707}\u{a708}\u{a709}\u{a70a}\u{a70b}\u{a70c}\u{a70d}\u{a70e}",
        r"\u{a70f}\u{a710}\u{a711}\u{a712}\u{a713}\u{a714}\u{a715}\u{a716}",
        r"\u{a720}\u{a721}\u{a789}\u{a78a}\u{ab5b}\u{ab6a}\u{ab6b}\u{fbb2}",
        r"\u{fbb3}\u{fbb4}\u{fbb5}\u{fbb6}\u{fbb7}\u{fbb8}\u{fbb9}\u{fbba}",
        r"\u{fbbb}\u{fbbc}\u{fbbd}\u{fbbe}\u{fbbf}\u{fbc0}\u{fbc1}\u{fbc2}",
        r"\u{fe13}\u{fe52}\u{fe55}\u{ff07}\u{ff0e}\u{ff1a}\u{ff3e}\u{ff40}",
        r"\u{ffe3}\u{1f3fb}\u{1f3fc}\u{1f3fd}\u{1f3fe}\u{1f3ff}",
        r"\p{Mn}\p{Me}\p{Cf}\p{Lm}]$",
    ))
    .expect("valid case-ignorable class")
});

/// Case_Ignorable for a single character (see `CASE_IGNORABLE_RE`).
fn case_ignorable(c: char) -> bool {
    let mut buf = [0u8; 4];
    CASE_IGNORABLE_RE.is_match(c.encode_utf8(&mut buf))
}

/// Every code point whose titlecase mapping differs from its uppercase
/// mapping — the table `py_title` needs, because Rust's stable `std` has
/// no `to_titlecase`. For every character NOT in this table the titlecase
/// mapping equals the uppercase mapping (verified exhaustively against
/// CPython: 135 divergences), which Rust's `char::to_uppercase` — the
/// full Unicode uppercase mapping, multi-character expansions included —
/// implements with the same mappings CPython does.
///
/// The members, all stable since the Unicode versions CPython 3.10–3.14
/// ship: the sharp s (`ß` → `Ss`, the SpecialCasing title entry); the
/// Latin digraph families (`Ǆ`/`ǅ`/`ǆ` and their LJ/NJ/DZ partners —
/// every spelling titles onto the mixed-case form); the Armenian
/// ligatures (`ﬗ` → `Եւ`, `ﬓ`–`ﬗ` → mixed-case pairs); the Latin
/// ligatures (`ﬀ` → `Ff` … `ﬆ` → `St`); the Georgian mkhedruli letters
/// (title is the character itself while uppercase is the Unicode 11
/// Mtavruli capital — `ა` stays `ა`, unlike its `.upper()`); and the
/// Greek ypogegrammeni family (`ᾀ` → `ᾈ`-shaped titles, the ᾳ/ῃ/ῳ
/// trio → `ᾼ`-shaped single-char titles, and the circumflex-perispomeni
/// trio → two- and three-character titles).
static TITLE_DIVERGENCES: &[(char, &str)] = &[
    ('\u{00DF}', "Ss"),
    ('\u{01C4}', "\u{01C5}"),
    ('\u{01C5}', "\u{01C5}"),
    ('\u{01C6}', "\u{01C5}"),
    ('\u{01C7}', "\u{01C8}"),
    ('\u{01C8}', "\u{01C8}"),
    ('\u{01C9}', "\u{01C8}"),
    ('\u{01CA}', "\u{01CB}"),
    ('\u{01CB}', "\u{01CB}"),
    ('\u{01CC}', "\u{01CB}"),
    ('\u{01F1}', "\u{01F2}"),
    ('\u{01F2}', "\u{01F2}"),
    ('\u{01F3}', "\u{01F2}"),
    ('\u{0587}', "\u{0535}\u{0582}"),
    ('\u{10D0}', "\u{10D0}"),
    ('\u{10D1}', "\u{10D1}"),
    ('\u{10D2}', "\u{10D2}"),
    ('\u{10D3}', "\u{10D3}"),
    ('\u{10D4}', "\u{10D4}"),
    ('\u{10D5}', "\u{10D5}"),
    ('\u{10D6}', "\u{10D6}"),
    ('\u{10D7}', "\u{10D7}"),
    ('\u{10D8}', "\u{10D8}"),
    ('\u{10D9}', "\u{10D9}"),
    ('\u{10DA}', "\u{10DA}"),
    ('\u{10DB}', "\u{10DB}"),
    ('\u{10DC}', "\u{10DC}"),
    ('\u{10DD}', "\u{10DD}"),
    ('\u{10DE}', "\u{10DE}"),
    ('\u{10DF}', "\u{10DF}"),
    ('\u{10E0}', "\u{10E0}"),
    ('\u{10E1}', "\u{10E1}"),
    ('\u{10E2}', "\u{10E2}"),
    ('\u{10E3}', "\u{10E3}"),
    ('\u{10E4}', "\u{10E4}"),
    ('\u{10E5}', "\u{10E5}"),
    ('\u{10E6}', "\u{10E6}"),
    ('\u{10E7}', "\u{10E7}"),
    ('\u{10E8}', "\u{10E8}"),
    ('\u{10E9}', "\u{10E9}"),
    ('\u{10EA}', "\u{10EA}"),
    ('\u{10EB}', "\u{10EB}"),
    ('\u{10EC}', "\u{10EC}"),
    ('\u{10ED}', "\u{10ED}"),
    ('\u{10EE}', "\u{10EE}"),
    ('\u{10EF}', "\u{10EF}"),
    ('\u{10F0}', "\u{10F0}"),
    ('\u{10F1}', "\u{10F1}"),
    ('\u{10F2}', "\u{10F2}"),
    ('\u{10F3}', "\u{10F3}"),
    ('\u{10F4}', "\u{10F4}"),
    ('\u{10F5}', "\u{10F5}"),
    ('\u{10F6}', "\u{10F6}"),
    ('\u{10F7}', "\u{10F7}"),
    ('\u{10F8}', "\u{10F8}"),
    ('\u{10F9}', "\u{10F9}"),
    ('\u{10FA}', "\u{10FA}"),
    ('\u{10FD}', "\u{10FD}"),
    ('\u{10FE}', "\u{10FE}"),
    ('\u{10FF}', "\u{10FF}"),
    ('\u{1F80}', "\u{1F88}"),
    ('\u{1F81}', "\u{1F89}"),
    ('\u{1F82}', "\u{1F8A}"),
    ('\u{1F83}', "\u{1F8B}"),
    ('\u{1F84}', "\u{1F8C}"),
    ('\u{1F85}', "\u{1F8D}"),
    ('\u{1F86}', "\u{1F8E}"),
    ('\u{1F87}', "\u{1F8F}"),
    ('\u{1F88}', "\u{1F88}"),
    ('\u{1F89}', "\u{1F89}"),
    ('\u{1F8A}', "\u{1F8A}"),
    ('\u{1F8B}', "\u{1F8B}"),
    ('\u{1F8C}', "\u{1F8C}"),
    ('\u{1F8D}', "\u{1F8D}"),
    ('\u{1F8E}', "\u{1F8E}"),
    ('\u{1F8F}', "\u{1F8F}"),
    ('\u{1F90}', "\u{1F98}"),
    ('\u{1F91}', "\u{1F99}"),
    ('\u{1F92}', "\u{1F9A}"),
    ('\u{1F93}', "\u{1F9B}"),
    ('\u{1F94}', "\u{1F9C}"),
    ('\u{1F95}', "\u{1F9D}"),
    ('\u{1F96}', "\u{1F9E}"),
    ('\u{1F97}', "\u{1F9F}"),
    ('\u{1F98}', "\u{1F98}"),
    ('\u{1F99}', "\u{1F99}"),
    ('\u{1F9A}', "\u{1F9A}"),
    ('\u{1F9B}', "\u{1F9B}"),
    ('\u{1F9C}', "\u{1F9C}"),
    ('\u{1F9D}', "\u{1F9D}"),
    ('\u{1F9E}', "\u{1F9E}"),
    ('\u{1F9F}', "\u{1F9F}"),
    ('\u{1FA0}', "\u{1FA8}"),
    ('\u{1FA1}', "\u{1FA9}"),
    ('\u{1FA2}', "\u{1FAA}"),
    ('\u{1FA3}', "\u{1FAB}"),
    ('\u{1FA4}', "\u{1FAC}"),
    ('\u{1FA5}', "\u{1FAD}"),
    ('\u{1FA6}', "\u{1FAE}"),
    ('\u{1FA7}', "\u{1FAF}"),
    ('\u{1FA8}', "\u{1FA8}"),
    ('\u{1FA9}', "\u{1FA9}"),
    ('\u{1FAA}', "\u{1FAA}"),
    ('\u{1FAB}', "\u{1FAB}"),
    ('\u{1FAC}', "\u{1FAC}"),
    ('\u{1FAD}', "\u{1FAD}"),
    ('\u{1FAE}', "\u{1FAE}"),
    ('\u{1FAF}', "\u{1FAF}"),
    ('\u{1FB2}', "\u{1FBA}\u{0345}"),
    ('\u{1FB3}', "\u{1FBC}"),
    ('\u{1FB4}', "\u{0386}\u{0345}"),
    ('\u{1FB7}', "\u{0391}\u{0342}\u{0345}"),
    ('\u{1FBC}', "\u{1FBC}"),
    ('\u{1FC2}', "\u{1FCA}\u{0345}"),
    ('\u{1FC3}', "\u{1FCC}"),
    ('\u{1FC4}', "\u{0389}\u{0345}"),
    ('\u{1FC7}', "\u{0397}\u{0342}\u{0345}"),
    ('\u{1FCC}', "\u{1FCC}"),
    ('\u{1FF2}', "\u{1FFA}\u{0345}"),
    ('\u{1FF3}', "\u{1FFC}"),
    ('\u{1FF4}', "\u{038F}\u{0345}"),
    ('\u{1FF7}', "\u{03A9}\u{0342}\u{0345}"),
    ('\u{1FFC}', "\u{1FFC}"),
    ('\u{FB00}', "Ff"),
    ('\u{FB01}', "Fi"),
    ('\u{FB02}', "Fl"),
    ('\u{FB03}', "Ffi"),
    ('\u{FB04}', "Ffl"),
    ('\u{FB05}', "St"),
    ('\u{FB06}', "St"),
    ('\u{FB13}', "\u{0544}\u{0576}"),
    ('\u{FB14}', "\u{0544}\u{0565}"),
    ('\u{FB15}', "\u{0544}\u{056B}"),
    ('\u{FB16}', "\u{054E}\u{0576}"),
    ('\u{FB17}', "\u{0544}\u{056D}"),
];

/// The Final_Sigma verdict for the capital sigma at *i* in *chars* —
/// the condition Unicode's SpecialCasing spells and CPython's
/// `str.lower()`/`str.capitalize()` apply: the sigma lowers to `ς`
/// exactly when the run before it is zero or more case-ignorable
/// characters ending on a cased one, AND the run after it is zero or
/// more case-ignorable characters with no cased one beyond. A sigma
/// with no cased character behind it (or only an uncased one, once the
/// ignorables are stepped over) is never final — `1Σ`.lower() keeps
/// its `σ` — and one with a cased character ahead keeps it too
/// (`aΣb` → `aσb`, the mark between skipped: `aΣ◌́b` → `aσ◌́b`).
/// `str.title()`'s lowercase path reaches this with the preceded side
/// already guaranteed (its `prev` is the immediately preceding
/// character's casedness, no skipping — see `py_title`).
fn sigma_is_final(chars: &[char], i: usize) -> bool {
    // Preceded by: zero or more case-ignorables, then a cased one.
    let mut j = i;
    while j > 0 && case_ignorable(chars[j - 1]) {
        j -= 1;
    }
    if j == 0 || !py_is_cased(chars[j - 1]) {
        return false;
    }
    // Not followed by: zero or more case-ignorables, then a cased one.
    let mut k = i + 1;
    while k < chars.len() {
        if case_ignorable(chars[k]) {
            k += 1;
            continue;
        }
        return !py_is_cased(chars[k]);
    }
    true
}

/// CPython's `str.title()`: the titlecase mapping of a character that
/// follows an uncased one, the full lowercase mapping of one that follows
/// a cased one — "cased" being the property `py_is_cased` tests on the
/// ORIGINAL character, the immediately preceding one (no case-ignorable
/// skipping: a combining mark between two letters breaks the run just
/// like a digit does, so `a◌́Σ` titles the sigma, never lowers it).
///
/// Two properties CPython's tables carry that per-character Rust helpers
/// alone do not, both reproduced exactly:
///
/// - the titlecase mapping is the TITLECASE mapping — `ß` titles to
///   `Ss` (not the `SS` its uppercase gives), `ǳ` to `ǲ`, `ﬀ` to `Ff`,
///   Georgian mkhedruli to itself (not its Mtavruli uppercase). The
///   divergences from uppercase are `TITLE_DIVERGENCES`; everywhere else
///   title equals uppercase, which `char::to_uppercase` (the full
///   mapping, multi-character expansions like `ǰ` → `J̌` included)
///   supplies;
/// - the lowercase path applies the final-sigma rule: a capital sigma
///   lowered with no cased character after it (skipping the
///   Case_Ignorable set — see `case_ignorable`) becomes `ς`, one with a
///   cased character ahead stays `σ` — `ΑΣΣ`.title() is `Ασς` exactly.
///   Sigma is the only character whose lowercase mapping is contextual,
///   so every other character goes through `char::to_lowercase` (the
///   full mapping — `İ` lowers to `i` + combining dot above).
///
/// The production input is the help page's action values —
/// `_`-separated ASCII identifiers (`play_pause`.title() →
/// `Play_Pause`... after the caller's `_`→space replacement,
/// `Play Pause`) — but the engine is a faithful `str.title()` for any
/// string, pinned by the parity zoo.
fn py_title(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut prev_cased = false;
    for (i, &ch) in chars.iter().enumerate() {
        if prev_cased {
            if ch == '\u{03A3}' {
                // Final sigma: preceded by a cased character (the
                // lowercase path guarantees it), ς unless a cased
                // character follows once the ignorable set is skipped.
                out.push(if sigma_is_final(&chars, i) {
                    '\u{03C2}'
                } else {
                    '\u{03C3}'
                });
            } else {
                // The full lowercase mapping (İ → i + U+0307 included).
                out.extend(ch.to_lowercase());
            }
        } else {
            // The titlecase mapping: the divergence table first, the
            // uppercase mapping (identical to title everywhere else)
            // otherwise.
            match TITLE_DIVERGENCES.binary_search_by(|probe| probe.0.cmp(&ch)) {
                Ok(idx) => out.push_str(TITLE_DIVERGENCES[idx].1),
                Err(_) => out.extend(ch.to_uppercase()),
            }
        }
        // Casedness of the ORIGINAL character decides the next step —
        // not of anything the mapping emitted.
        prev_cased = py_is_cased(ch);
    }
    out
}

/// CPython's `str.capitalize()`: the TITLECASE mapping of the first
/// character (not its uppercase — `ß` capitalizes to `Ss`, exactly its
/// title, not the `SS` its uppercase gives; `ǳ` to `ǲ`; `ﬀ` to `Ff`;
/// everywhere else title equals uppercase, which `char::to_uppercase`
/// supplies — see `TITLE_DIVERGENCES`), the full lowercase mapping of
/// every character after it, with the tail's capital sigmas read
/// through the Final_Sigma condition exactly as `str.lower()` reads
/// them (`sigma_is_final`, scanning past the first character: `ΑΣΣ`
/// capitalizes to `Ασς`, `ABCΣ` to `Abcς`, while `1Σ` keeps its `σ`).
/// The production inputs are ASCII (`item_type`'s `track`/`album`/
/// `artist`/`playlist`, the subtitle's `PRIVATE`/`PUBLIC`/`UNLISTED`),
/// where the mapping is the trivial one; the case machinery exists for
/// the parity zoo, and tracks the Rust toolchain's Unicode tables
/// exactly like `py_title` does (the help-page caveat).
pub(crate) fn py_capitalize(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &ch) in chars.iter().enumerate() {
        if i == 0 {
            // The titlecase mapping: the divergence table first, the
            // uppercase mapping (identical to title everywhere else)
            // otherwise.
            match TITLE_DIVERGENCES.binary_search_by(|probe| probe.0.cmp(&ch)) {
                Ok(idx) => out.push_str(TITLE_DIVERGENCES[idx].1),
                Err(_) => out.extend(ch.to_uppercase()),
            }
        } else if ch == '\u{03A3}' && sigma_is_final(&chars, i) {
            out.push('\u{03C2}');
        } else {
            // The full lowercase mapping (İ → i + U+0307 included).
            out.extend(ch.to_lowercase());
        }
    }
    out
}

/// The per-row formatting core shared by the help-page row engines —
/// the `help_page_rows` body, factored out so the spec engine and the
/// fused keymap-walk engine below cannot drift.
///
/// The name through the `action_value.replace("_", " ").title()`
/// pipeline (`py_title`, the `_format_action_name` body), the keys as
/// the `" / "` join of the formatted key strings, or `"(unbound)"` when
/// the action has no binding.
fn help_row_from_parts(
    category: &str,
    action_value: &str,
    description: &str,
    key_strs: &[String],
) -> (String, String, String, String) {
    let name = py_title(&action_value.replace('_', " "));
    let keys_str = if key_strs.is_empty() {
        "(unbound)".to_string()
    } else {
        key_strs.join(" / ")
    };
    (
        category.to_string(),
        name,
        keys_str,
        description.to_string(),
    )
}

/// The all-rows list — the row-formatting half of the help page's row
/// assembly, the `HelpPage._build_rows` spec form.
///
/// Mirrors the pure-Python reference: each spec entry
/// `(category, action_value, description, key_strs)` becomes the row
/// `(category, name, keys_str, description)` through the shared
/// formatting core (`help_row_from_parts`). The description arrives
/// already defaulted (`""` for an undocumented action), exactly as the
/// reference's `ACTION_DESCRIPTIONS.get(action, "")` produced it; the
/// key strings arrive already formatted, exactly as the reference's
/// `format_key` calls produced them.
///
/// The keymap walk that produces the spec — the
/// `get_keys_for_action`/`format_key` calls over `ACTION_CATEGORIES`
/// and `ACTION_DESCRIPTIONS` — used to stay on the Python side next to
/// the objects it reads; it now lives in the fused engine
/// (`help_page_rows_for_keymap` below). This engine remains the
/// composable primitive: the form the fused engine's pure-Python
/// fallback composes, and the spec shape the row contract tests pin.
#[pyfunction]
pub(crate) fn help_page_rows(
    spec: &Bound<'_, PyList>,
) -> PyResult<Vec<(String, String, String, String)>> {
    let mut rows: Vec<(String, String, String, String)> = Vec::with_capacity(spec.len());
    for entry in spec.iter() {
        let (category, action_value, description, key_strs): (String, String, String, Vec<String>) =
            entry.extract()?;
        rows.push(help_row_from_parts(
            &category,
            &action_value,
            &description,
            &key_strs,
        ));
    }
    Ok(rows)
}

/// The fused all-rows engine behind `HelpPage._build_rows` — the whole
/// row assembly in one call: the keymap walk that used to stay on the
/// Python side next to the objects it reads now runs here.
///
/// Mirrors the pure-Python reference walk over the help page's own
/// objects: the category walk over *categories* in dict insertion
/// order, each action's description from *descriptions* through the
/// interpreter's own `.get(action, "")` (`""` for an undocumented
/// action), and its bound sequences from *bindings* — the
/// `get_keys_for_action` scan: one pass over the binding table here,
/// each bound sequence grouped under its action's value string so a
/// per-action lookup answers exactly the sequences the scan would, in
/// the binding table's own order — each formatted through
/// `format_key`'s `" ".join(seq)`. Every row then goes through the
/// same formatting core as the spec engine (`help_row_from_parts`),
/// so the fused engine and the composable primitive cannot drift.
///
/// *bindings* is the `KeyMap.bindings` dict (sequence tuples → `Action`
/// members — the str-enum values extract as their value strings, the
/// `KeyMatcher` convention); *categories* is `ACTION_CATEGORIES`
/// (category str → list of `Action` members — a non-list value raises
/// the downcast's own TypeError, a shape the literal dict of lists
/// never produces); *descriptions* is `ACTION_DESCRIPTIONS` (`Action`
/// member → description str). The keymap acquisition (`get_keymap()`)
/// stays at the Python call site; the walk, the keymap reads and the
/// row assembly all happen here.
#[pyfunction]
#[pyo3(signature = (bindings, categories, descriptions))]
pub(crate) fn help_page_rows_for_keymap(
    bindings: &Bound<'_, PyDict>,
    categories: &Bound<'_, PyDict>,
    descriptions: &Bound<'_, PyDict>,
) -> PyResult<Vec<(String, String, String, String)>> {
    // One pass over the binding table: each sequence grouped under its
    // action's value string, the groups keeping the dict's own order —
    // `get_keys_for_action`'s per-action scan answered in advance.
    let mut by_action: HashMap<String, Vec<Vec<String>>> = HashMap::new();
    for (seq_obj, action_obj) in bindings.iter() {
        let seq: Vec<String> = seq_obj.extract()?;
        let action: String = action_obj.extract()?;
        by_action.entry(action).or_default().push(seq);
    }
    let mut rows: Vec<(String, String, String, String)> = Vec::new();
    for (category_obj, actions_obj) in categories.iter() {
        let category: String = category_obj.extract()?;
        let actions = actions_obj.downcast::<PyList>()?;
        for action_obj in actions.iter() {
            // The str-enum member extracts as its value string —
            // `action.value` in the reference's spec.
            let action_value: String = action_obj.extract()?;
            // `ACTION_DESCRIPTIONS.get(action, "")` — the interpreter's
            // own dict lookup, `""` when the action is undocumented.
            let description = match descriptions.get_item(action_obj)? {
                Some(desc) => desc.extract()?,
                None => String::new(),
            };
            // `get_keys_for_action(action)` + `format_key(seq)` — the
            // grouped scan in binding-table order, each sequence
            // formatted `" ".join(seq)`.
            let key_strs: Vec<String> = by_action
                .get(&action_value)
                .map(|seqs| seqs.iter().map(|seq| seq.join(" ")).collect())
                .unwrap_or_default();
            rows.push(help_row_from_parts(
                &category,
                &action_value,
                &description,
                &key_strs,
            ));
        }
    }
    Ok(rows)
}

/// One visible-row emission of the help walk: `(is_header, category,
/// name, keys, description)` — headers carry `(True, category, "", "",
/// "")`, data rows the row itself.
type HelpVisibleRow = (bool, String, String, String, String);

/// The visible-row walk behind `HelpPage._populate_table` — the pure
/// core of the help page's filtering.
///
/// Mirrors the pure-Python reference: the needle is
/// `filter_text.lower().strip()` (lowercased FIRST, then stripped — the
/// opposite order of the playlist picker's query), an empty needle
/// filters nothing, and a non-empty one matches a row only when it is a
/// substring of the lowercased `"{category} {name} {keys} {desc}"`
/// haystack. A category header is emitted whenever the category changes
/// — starting from `""` (so the first category always earns one, and an
/// empty-string category never does), and re-earned if a category
/// recurs after another (a row list the reference would walk the same
/// way, whatever its caller produced). Headers come back as
/// `(True, category, "", "", "")` and data rows as
/// `(False, category, name, keys, description)`; the Textual markup
/// (`[b]{category}[/b]`) and the DataTable calls stay on the Python side.
#[pyfunction]
#[pyo3(signature = (all_rows, filter_text))]
pub(crate) fn help_page_visible_rows(
    all_rows: &Bound<'_, PyList>,
    filter_text: &str,
) -> PyResult<Vec<HelpVisibleRow>> {
    // filter_text.lower().strip() — lower first, then strip.
    let lowered = python_lower(filter_text);
    let needle = python_strip(&lowered);
    let mut out: Vec<HelpVisibleRow> = Vec::new();
    let mut current_category = String::new();
    for row in all_rows.iter() {
        let (category, name, keys, desc): (String, String, String, String) = row.extract()?;
        // `if needle:` — an empty (or whitespace-only) needle filters
        // nothing; every row passes.
        if !needle.is_empty() {
            let haystack = python_lower(&format!("{category} {name} {keys} {desc}"));
            if !haystack.contains(needle) {
                continue;
            }
        }
        if category != current_category {
            current_category.clone_from(&category);
            out.push((
                true,
                category.clone(),
                String::new(),
                String::new(),
                String::new(),
            ));
        }
        out.push((false, category, name, keys, desc));
    }
    Ok(out)
}
