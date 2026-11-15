//! Native cores for `src/ytm_player/services/update_check.py` — PEP 440
//! version ordering (BigNum comparison) and the newer check.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;
use std::sync::LazyLock;

/// The PEP 440 version grammar from `packaging`, in the two shapes its
/// releases have shipped, selected by the `admit_unicode_folds` probe
/// the Python side passes (see `crash_staleness_note`): every word
/// spelling is per-character cased (`AlPhA` is a valid pre-release
/// marker), and the wrapping whitespace is CPython's `\s` (see
/// `CRASH_META_RE`). Anchored at both ends like the reference's
/// `fullmatch`, with group names mirroring packaging's own.
///
/// The two shapes differ only in the Unicode case folds:
///
/// - `PEP440_FOLD_RE` matches the grammar `packaging` compiled through
///   26.0 — plain `re.IGNORECASE` on a Unicode pattern, no ASCII
///   scoping — which admits the simple case folds wherever a letter
///   can appear: the long s `ſ` (U+017F) wherever an s/S can, the
///   dotted and dotless İ/ı (U+0130/U+0131) wherever an i/I can, and
///   the Kelvin sign `K` (U+212A) wherever a k/K can. `1.0+ſ`,
///   `1.0+K`, `1.0prevıew` and `1.0poſt` are valid versions on those
///   releases.
/// - `PEP440_ASCII_RE` matches packaging 26.1+ — the groups are
///   `(?a:...)`-scoped and the local alphabet ASCII — so the same
///   fold spellings are InvalidVersion, exactly the verdicts the
///   pure-Python fallback's `Version(...)` returns on those releases.
///
/// Only the `preview` word's i, the `post` word's s and the local
/// alphabet can ever see a fold (no other grammar position spells an
/// i, s or k), which is why the two grammars differ in exactly those
/// three places. The comparison key the parse builds never changes:
/// on the ASCII grammar a fold spelling cannot appear, and on the
/// fold grammar it ranks by its raw characters through the same slot
/// tuples both grammars share.
static PEP440_FOLD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^[\s\x{1c}-\x{1f}]*[vV]?",
        r"(?:",
        r"(?:(?P<epoch>[0-9]+)!)?",
        r"(?P<release>[0-9]+(?:\.[0-9]+)*)",
        r"(?:[._-]?(?P<pre_l>[aA][lL][pP][hH][aA]|[aA]|[bB][eE][tT][aA]|[bB]",
        r"|[pP][rR][eE][vV][iI\x{130}\x{131}][eE][wW]|[pP][rR][eE]|[cC]|[rR][cC])",
        r"[._-]?(?P<pre_n>[0-9]+)?)?",
        r"(?:-(?P<post_n1>[0-9]+)|[._-]?(?P<post_l>[pP][oO][sS\x{17f}][tT]|[rR][eE][vV]|[rR])",
        r"[._-]?(?P<post_n2>[0-9]+)?)?",
        r"(?:[._-]?(?P<dev_l>[dD][eE][vV])[._-]?(?P<dev_n>[0-9]+)?)?",
        r")",
        r"(?:\+(?P<local>[a-zA-Z0-9\x{130}\x{131}\x{17f}\x{212a}]+(?:[._-][a-zA-Z0-9\x{130}\x{131}\x{17f}\x{212a}]+)*))?",
        r"[\s\x{1c}-\x{1f}]*$",
    ))
    .expect("valid pep440 regex")
});

/// The ASCII-scoped shape of the same grammar — packaging 26.1+ (see
/// `PEP440_FOLD_RE` for the shared structure and the fold inventory).
static PEP440_ASCII_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^[\s\x{1c}-\x{1f}]*[vV]?",
        r"(?:",
        r"(?:(?P<epoch>[0-9]+)!)?",
        r"(?P<release>[0-9]+(?:\.[0-9]+)*)",
        r"(?:[._-]?(?P<pre_l>[aA][lL][pP][hH][aA]|[aA]|[bB][eE][tT][aA]|[bB]",
        r"|[pP][rR][eE][vV][iI][eE][wW]|[pP][rR][eE]|[cC]|[rR][cC])",
        r"[._-]?(?P<pre_n>[0-9]+)?)?",
        r"(?:-(?P<post_n1>[0-9]+)|[._-]?(?P<post_l>[pP][oO][sS][tT]|[rR][eE][vV]|[rR])",
        r"[._-]?(?P<post_n2>[0-9]+)?)?",
        r"(?:[._-]?(?P<dev_l>[dD][eE][vV])[._-]?(?P<dev_n>[0-9]+)?)?",
        r")",
        r"(?:\+(?P<local>[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*))?",
        r"[\s\x{1c}-\x{1f}]*$",
    ))
    .expect("valid pep440 ascii regex")
});

/// CPython's default `sys.int_max_str_digits`: the length above which the
/// int(str) conversion raises a plain ValueError. `packaging` parses
/// version components through int(), so a numeric component longer than
/// this is a version the reference cannot compare — the engine enforces
/// the same default limit so both backends agree on the no-verdict
/// outcome for such strings.
const INT_MAX_STR_DIGITS: usize = 4300;

/// A non-negative integer of unbounded width, held as its normalized
/// decimal digits (leading zeros stripped; the empty string is zero).
/// PEP 440 epochs and release segments can exceed every fixed-width
/// integer, so the comparison is numeric on the digit run itself: the
/// longer run is the larger number, equal lengths compare digit by digit.
#[derive(PartialEq, Eq)]
struct BigNum(String);

impl BigNum {
    fn zero() -> Self {
        BigNum(String::new())
    }

    /// Normalize a matched digit run, or None when it exceeds CPython's
    /// int-conversion digit limit (see `INT_MAX_STR_DIGITS`).
    fn from_digits(digits: &str) -> Option<Self> {
        if digits.len() > INT_MAX_STR_DIGITS {
            return None;
        }
        Some(BigNum(digits.trim_start_matches('0').to_string()))
    }

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }
}

impl Ord for BigNum {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.0.is_empty(), other.0.is_empty()) {
            // Both zero.
            (true, true) => std::cmp::Ordering::Equal,
            // Zero is smaller than any other number.
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => self
                .0
                .len()
                .cmp(&other.0.len())
                .then_with(|| self.0.cmp(&other.0)),
        }
    }
}

impl PartialOrd for BigNum {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// packaging's `_LETTER_NORMALIZATION`, applied after `str.lower()`
/// exactly like `_parse_letter_version`: the alternate spellings collapse
/// onto the canonical marker letters (alpha→a, beta→b, c/pre/preview→rc,
/// rev/r→post), and a spelling the lower() left untouched — an exotic
/// Unicode fold like `prevıew` or `poſt`, which no map key equals — stays
/// raw and compares as its own string, exactly the tuple the reference
/// builds.
fn pep440_letter(letter: &str) -> String {
    match letter.to_lowercase().as_str() {
        "alpha" => "a".to_string(),
        "beta" => "b".to_string(),
        "c" | "pre" | "preview" => "rc".to_string(),
        "rev" | "r" => "post".to_string(),
        raw => raw.to_string(),
    }
}

/// One pre/post/dev slot of the comparison key. packaging's `_cmpkey`
/// fills each slot with `NegativeInfinity`, the `(letter, number)` tuple
/// or `Infinity`, and the slots compare in exactly that order — the
/// enum's derived ordering reproduces it: absent-before-everything, the
/// `(letter, number)` pairs comparing as tuples (the letter as its
/// post-lower() post-map string, so an exotic fold spelling like
/// `prevıew` ranks between `b` and `rc` by its raw characters, and
/// `poſt` sorts above `post`), absent-after-everything.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Pep440Slot {
    NegInf,
    Seg(String, BigNum),
    Inf,
}

/// The pre/post/dev comparison suffix in packaging's `_cmpkey` order,
/// each slot one `Pep440Slot`.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Pep440Suffix {
    pre: Pep440Slot,
    post: Pep440Slot,
    dev: Pep440Slot,
}

/// One local-version segment. Mirrors packaging's `(n, "")` /
/// `(NegativeInfinity, s)` segment pairs: a string segment sorts before
/// every numeric segment, strings compare lexicographically and numbers
/// numerically.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Pep440LocalSeg {
    Str(String),
    Num(BigNum),
}

/// The PEP 440 comparison key, in packaging's `_cmpkey` order: epoch,
/// release (trailing zeros stripped, so `1.0.0 == 1`), the pre/post/dev
/// suffix, then the local segments — absent sorting before present, and
/// the segment list comparing as the tuple it is.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Pep440Key {
    epoch: BigNum,
    release: Vec<BigNum>,
    suffix: Pep440Suffix,
    local: Option<Vec<Pep440LocalSeg>>,
}

/// How a PEP 440 parse failed. The distinction matters to callers that
/// catch only `InvalidVersion` (an `InvalidVersion` subclass): the
/// grammar's rejection is that exception, while a numeric component past
/// CPython's int-conversion digit limit is the plain `ValueError` the
/// constructor's own `int()` raises — which those callers let propagate.
pub(crate) enum Pep440Err {
    /// The grammar rejected the string — packaging raises `InvalidVersion`.
    Invalid,
    /// A numeric component exceeds CPython's int-conversion digit limit
    /// (see `INT_MAX_STR_DIGITS`) — the constructor's `int()` raises a
    /// plain `ValueError`, not an `InvalidVersion`.
    IntLimit,
}

/// `BigNum::from_digits` with the failure reason kept: `None` (the digit
/// run exceeds CPython's int-conversion limit) becomes `Pep440Err::IntLimit`.
fn big_num(digits: &str) -> Result<BigNum, Pep440Err> {
    BigNum::from_digits(digits).ok_or(Pep440Err::IntLimit)
}

/// Parse a version string into its PEP 440 comparison key, keeping the
/// failure reason: `Err(Pep440Err::Invalid)` when the grammar rejects the
/// string, `Err(Pep440Err::IntLimit)` when a numeric component exceeds
/// CPython's int-conversion digit limit (see `INT_MAX_STR_DIGITS`).
/// *admit_unicode_folds* selects the grammar shape (see `PEP440_FOLD_RE`).
pub(crate) fn parse_pep440_err(
    text: &str,
    admit_unicode_folds: bool,
) -> Result<Pep440Key, Pep440Err> {
    let caps = if admit_unicode_folds {
        PEP440_FOLD_RE.captures(text)
    } else {
        PEP440_ASCII_RE.captures(text)
    };
    let caps = caps.ok_or(Pep440Err::Invalid)?;

    let epoch = big_num(caps.name("epoch").map(|m| m.as_str()).unwrap_or("0"))?;
    let mut release: Vec<BigNum> = caps
        .name("release")
        .ok_or(Pep440Err::Invalid)?
        .as_str()
        .split('.')
        .map(big_num)
        .collect::<Result<Vec<_>, _>>()?;
    // Strip trailing zeros: 1.0.0 compares equal to 1.
    while release.last().is_some_and(BigNum::is_zero) {
        release.pop();
    }

    // Each numeric group is Option<Result<BigNum, Pep440Err>>: the outer
    // Option is "did the group match", the inner is the digit-limit-checked
    // number. A matched group whose digits exceed the limit invalidates the
    // whole version (the reference's int() raises through), an unmatched one
    // simply carries no number.
    let pre_l = caps.name("pre_l").map(|m| m.as_str());
    let pre_n = caps.name("pre_n").map(|m| big_num(m.as_str()));
    let post_n1 = caps.name("post_n1").map(|m| big_num(m.as_str()));
    let post_l = caps.name("post_l").map(|m| m.as_str());
    let post_n2 = caps.name("post_n2").map(|m| big_num(m.as_str()));
    let dev_l = caps.name("dev_l").map(|m| m.as_str());
    let dev_n = caps.name("dev_n").map(|m| big_num(m.as_str()));

    let post_matched = post_n1.is_some() || post_l.is_some();
    let dev_matched = dev_l.is_some();

    // The pre slot: a dev-only release (no pre, no post, dev present)
    // sorts before every alpha; a release without a pre sorts after every
    // rc; otherwise the marker's (letter, number) pair — the letter through
    // packaging's lower+normalise map, an implicit 0 when the digits are
    // absent — comparing as the tuple it is, so an exotic fold spelling
    // like `prevıew` ranks by its raw characters, between `b` and `rc`.
    let pre = if let Some(letter) = pre_l {
        let number = match pre_n {
            Some(number) => number?,
            None => BigNum::zero(),
        };
        Pep440Slot::Seg(pep440_letter(letter), number)
    } else if !post_matched && dev_matched {
        Pep440Slot::NegInf
    } else {
        Pep440Slot::Inf
    };

    // post: `-N` (post_n1) and `postN`/`post`/`rev`/`r` (post_l+post_n2)
    // both normalise to the ("post", N) pair — the `-N` shape's letter is
    // exactly "post" — again with the implicit 0; no post segment sorts
    // before every one that has it.
    let post = if post_matched {
        let number = match post_n1.or(post_n2) {
            Some(number) => number?,
            None => BigNum::zero(),
        };
        let letter = match post_l {
            Some(letter) => pep440_letter(letter),
            None => "post".to_string(),
        };
        Pep440Slot::Seg(letter, number)
    } else {
        Pep440Slot::NegInf
    };

    // dev: the ("dev", N) pair with the implicit 0; no dev segment sorts
    // after every one that has it.
    let dev = match dev_l {
        Some(letter) => {
            let number = match dev_n {
                Some(number) => number?,
                None => BigNum::zero(),
            };
            Pep440Slot::Seg(pep440_letter(letter), number)
        }
        None => Pep440Slot::Inf,
    };

    let suffix = Pep440Suffix { pre, post, dev };

    let local = match caps.name("local") {
        None => None,
        Some(m) => Some(
            m.as_str()
                .split(['.', '_', '-'])
                .map(|part| {
                    if part.bytes().all(|b| b.is_ascii_digit()) {
                        // int(part): a digit run beyond the int-conversion
                        // limit invalidates the version.
                        big_num(part).map(Pep440LocalSeg::Num)
                    } else {
                        Ok(Pep440LocalSeg::Str(part.to_lowercase()))
                    }
                })
                .collect::<Result<Vec<Pep440LocalSeg>, _>>()?,
        ),
    };

    Ok(Pep440Key {
        epoch,
        release,
        suffix,
        local,
    })
}

// ---------------------------------------------------------------------------
// Update-check version comparison (mirrors services/update_check.py)
// ---------------------------------------------------------------------------

/// The strictly-newer PEP 440 verdict behind `check_for_update` from
/// `services/update_check.py` — the pure core of `_is_newer`.
///
/// Mirrors the pure-Python reference: `Version(latest) > Version(current)`
/// with only `InvalidVersion` caught (returning `False`). The parse and
/// the comparison are the shared PEP 440 machinery the crash-staleness
/// engine already uses (`parse_pep440_err` → the `Pep440Key` ordering),
/// so every accept/reject and every ordering decision the reference makes
/// comes free — epochs, trailing-zero releases, the pre/post/dev slot
/// tuples, local segments.
///
/// The two failure kinds differ exactly as the reference's single
/// `except InvalidVersion` makes them:
///
/// - a grammar rejection is `InvalidVersion` — the reference catches it
///   and returns `False`, so the engine answers `Ok(false)` directly;
/// - a numeric component past CPython's int-conversion digit limit is
///   the plain `ValueError` the constructor's own `int()` raises (NOT an
///   `InvalidVersion`) — the reference lets it propagate out of
///   `_is_newer` into `check_for_update`'s caller, so the engine raises
///   `PyValueError` with CPython's own message shape and the binding
///   passes it through untouched.
///
/// `Version(latest)` evaluates before `Version(current)` in the
/// reference's expression, so a bad *latest* never even parses *current*:
/// the engine mirrors the order (an over-limit latest raises before the
/// current string is looked at, an invalid latest short-circuits to
/// `Ok(false)` the same way).
///
/// *admit_unicode_folds* steers the PEP 440 grammar to the shape the
/// installed `packaging` compiled — the Python side probes it once
/// (`Version("1.0+ſ")`: parseable through packaging 26.0, InvalidVersion
/// from 26.1's `(?a:...)` scoping on) and passes the verdict here, so
/// the engine tracks the reference the fallback itself calls, on every
/// `packaging` the dependency range (`>=21.0`, uncapped) admits.
#[pyfunction]
#[pyo3(signature = (latest, current, admit_unicode_folds))]
pub(crate) fn pep440_is_newer(
    latest: &str,
    current: &str,
    admit_unicode_folds: bool,
) -> PyResult<bool> {
    let latest_key = match parse_pep440_err(latest, admit_unicode_folds) {
        Ok(key) => key,
        // except InvalidVersion: return False — the grammar's rejection
        // never reaches the comparison.
        Err(Pep440Err::Invalid) => return Ok(false),
        // The plain ValueError CPython's int() raises for an over-limit
        // component — not an InvalidVersion, so it propagates.
        Err(Pep440Err::IntLimit) => return Err(int_limit_error(latest)),
    };
    let current_key = match parse_pep440_err(current, admit_unicode_folds) {
        Ok(key) => key,
        Err(Pep440Err::Invalid) => return Ok(false),
        Err(Pep440Err::IntLimit) => return Err(int_limit_error(current)),
    };
    Ok(latest_key > current_key)
}

/// The `ValueError` CPython's own `int()` raises for a digit run past the
/// conversion limit, in CPython's message shape — the digit count the
/// reference's message reports is the offending operand's longest digit
/// run (the component the constructor was converting when it raised).
fn int_limit_error(version: &str) -> PyErr {
    let digits = version
        .split(|c: char| !c.is_ascii_digit())
        .map(str::len)
        .max()
        .unwrap_or(0);
    PyValueError::new_err(format!(
        "Exceeds the limit ({INT_MAX_STR_DIGITS} digits) for integer string conversion: value \
         has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
    ))
}
