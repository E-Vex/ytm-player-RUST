//! Native cores for `src/ytm_player/utils/doctor.py` — diagnostics
//! redaction, running-status texts, crash traces and mpv/libmpv/mpris
//! status probes.

use crate::app::keys::unpack_too_many_message;
use crate::json::unpack_two;
use crate::services::auth::py_strip;
use crate::services::history::builtin_int;
use crate::services::update_check::parse_pep440_err;
use crate::services::update_check::Pep440Key;
use crate::utils::pycompat::py_isspace;
use crate::utils::pycompat::py_str;
use pyo3::exceptions::{PyImportError, PyIndexError, PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyList, PySlice, PyString, PyTuple};
use regex::Regex;
use std::sync::LazyLock;

/// Stream URLs in diagnostics may contain the public IP and signed playback
/// parameters; the whole URL is removed rather than guessing which query
/// fields are sensitive. Mirrors `doctor._URL_PATTERN`.
static DIAG_URL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\b(?:https?|tcp)://[^\s<>"']+"#).expect("valid diagnostic url regex")
});

/// Header-value pairs redacted from diagnostics before the report is shown.
/// Each keeps its prefix (group 1) and replaces the value with `[REDACTED]`.
/// Mirrors `doctor._REDACT_PATTERNS` (same order, same grouping).
static DIAG_REDACT_RES: LazyLock<[Regex; 6]> = LazyLock::new(|| {
    [
        Regex::new(r"(?i)(authorization\s*[:=]\s*)(.+)").expect("valid redact regex"),
        Regex::new(r"(?i)(cookie\s*[:=]\s*)(.+)").expect("valid redact regex"),
        Regex::new(r"(?i)(bearer\s+)(\S+)").expect("valid redact regex"),
        Regex::new(r"(?i)(token\s*[:=]\s*)(\S+)").expect("valid redact regex"),
        Regex::new(r"(?i)(x-goog-pageid\s*[:=]\s*)(\S+)").expect("valid redact regex"),
        Regex::new(r"(SAPISID\s*=\s*)(\S+)").expect("valid redact regex"),
    ]
});

// ---------------------------------------------------------------------------
// Diagnostics (mirrors ytm_player/utils/doctor.py and logging.py)
// ---------------------------------------------------------------------------

/// Redact recognized auth headers/tokens and HTTP(S)/TCP URLs from
/// diagnostic text. Mirrors `ytm_player.utils.doctor._redact` exactly:
/// each header pattern keeps its prefix and masks the rest of the line,
/// then whole URLs are replaced with `[URL REDACTED]`.
#[pyfunction]
pub(crate) fn redact(text: &str) -> String {
    let mut out = text.to_string();
    for re in DIAG_REDACT_RES.iter() {
        out = re.replace_all(&out, "${1}[REDACTED]").into_owned();
    }
    DIAG_URL_RE.replace_all(&out, "[URL REDACTED]").into_owned()
}

/// Mirror `pathlib.PurePath.name` for the platform's path syntax: split on
/// separators, drop empty and `.` components, and return the last remaining
/// component ("" when none). `..` is kept, matching pathlib.
fn py_path_name(path: &str) -> String {
    let separators: &[char] = if cfg!(windows) { &['/', '\\'] } else { &['/'] };
    let mut name = String::new();
    for component in path.split(separators) {
        if component.is_empty() || component == "." {
            continue;
        }
        name = component.to_string();
    }
    name
}

/// True when *argv* is a real ytm-player entry point. Mirrors
/// `ytm_player.utils.doctor._argv_is_ytm`: the `ytm` console script
/// (interpreter in argv[0], script path in argv[1]) or `python -m
/// ytm_player`; never an unrelated process whose cmdline merely mentions
/// a ytm-player path.
#[pyfunction]
pub(crate) fn argv_is_ytm(argv: Vec<String>) -> bool {
    let head_names: Vec<String> = argv.iter().take(2).map(|a| py_path_name(a)).collect();
    head_names.iter().any(|name| name == "ytm") || argv.iter().any(|a| a == "ytm_player")
}

/// The start index of Python's `seq[-n:]` tail slice: a positive *n*
/// keeps the last n, zero keeps everything (`seq[-0:]` is `seq[0:]`),
/// and a negative n drops the leading -n entries (empty when that
/// exceeds the length).
pub(crate) fn py_tail_start(len: usize, n: i64) -> usize {
    if n > 0 {
        len.saturating_sub(n as usize)
    } else if n == 0 {
        0
    } else {
        // unsigned_abs rather than -n: negating i64::MIN overflows.
        (n.unsigned_abs() as i128).min(len as i128) as usize
    }
}

// ---------------------------------------------------------------------------
// Doctor report assembly (utils/doctor.py)
// ---------------------------------------------------------------------------

/// The report assembly behind `gather_diagnostics` from
/// `utils/doctor.py` — the `ytm doctor` text users paste into issues.
///
/// Mirrors the pure-Python reference (`_doctor_report`): the
/// multi-section report built from the probes' results — the banner
/// and the aligned identity labels (`Version:` / `Python:  ` /
/// `OS:      ` / `Machine:`), the paths block with its own alignment
/// (`config:` through `crashes:`), the probe lines appended raw, the
/// empty-log placeholder (a falsy *recent_errors* — the reference's
/// `filtered if filtered else` — substitutes `(no warnings or errors
/// logged)`), the crash-file section (`None` is the no-crash-files
/// marker; a `(path, content)` pair unpacks into the `From:` line, the
/// content, and the staleness note when *crash_note* is not `None` —
/// an empty-string note still lands, `is not None` is not a truthiness
/// gate), the hooks section with its trailing note paragraph, and the
/// final `"\n"` join run through the redaction pass (the same `redact`
/// engine the Python side binds to).
///
/// Every interpolation runs through `str()` exactly like the
/// reference's f-strings — `Path` objects render as their path text,
/// the version triple as its numbers, and anything with a `__str__`
/// through its own. The inputs are never mutated.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the report's 22 slots — one parameter per line the reference appends
#[pyo3(signature = (version, py_major, py_minor, py_micro, system, release, machine, mpv_status, libmpv_status, config_path, theme_path, session_path, log_path, crash_dir, process_status, mpris_status, recent_errors, recent_mpv, faulthandler, crash, crash_note, hooks))]
pub(crate) fn doctor_report<'py>(
    version: &Bound<'py, PyAny>,
    py_major: &Bound<'py, PyAny>,
    py_minor: &Bound<'py, PyAny>,
    py_micro: &Bound<'py, PyAny>,
    system: &Bound<'py, PyAny>,
    release: &Bound<'py, PyAny>,
    machine: &Bound<'py, PyAny>,
    mpv_status: &Bound<'py, PyAny>,
    libmpv_status: &Bound<'py, PyAny>,
    config_path: &Bound<'py, PyAny>,
    theme_path: &Bound<'py, PyAny>,
    session_path: &Bound<'py, PyAny>,
    log_path: &Bound<'py, PyAny>,
    crash_dir: &Bound<'py, PyAny>,
    process_status: &Bound<'py, PyAny>,
    mpris_status: &Bound<'py, PyAny>,
    recent_errors: &Bound<'py, PyAny>,
    recent_mpv: &Bound<'py, PyAny>,
    faulthandler: &Bound<'py, PyAny>,
    crash: &Bound<'py, PyAny>,
    crash_note: &Bound<'py, PyAny>,
    hooks: &Bound<'py, PyAny>,
) -> PyResult<String> {
    let mut sections: Vec<String> = Vec::with_capacity(40);

    sections.push("=== ytm-player diagnostics ===".to_string());
    sections.push(format!("Version: {}", py_str(version)?));
    sections.push(format!(
        "Python:  {}.{}.{}",
        py_str(py_major)?,
        py_str(py_minor)?,
        py_str(py_micro)?
    ));
    sections.push(format!("OS:      {} {}", py_str(system)?, py_str(release)?));
    sections.push(format!("Machine: {}", py_str(machine)?));
    sections.push(py_str(mpv_status)?);
    sections.push(py_str(libmpv_status)?);

    sections.push(String::new());
    sections.push("=== Paths ===".to_string());
    sections.push(format!("config:   {}", py_str(config_path)?));
    sections.push(format!("theme:    {}", py_str(theme_path)?));
    sections.push(format!("session:  {}", py_str(session_path)?));
    sections.push(format!("log:      {}", py_str(log_path)?));
    sections.push(format!("crashes:  {}", py_str(crash_dir)?));

    sections.push(String::new());
    sections.push("=== Process status ===".to_string());
    sections.push(py_str(process_status)?);

    sections.push(String::new());
    sections.push("=== MPRIS / media keys ===".to_string());
    sections.push(py_str(mpris_status)?);

    sections.push(String::new());
    sections.push("=== Recent ERROR/WARNING (last 20) ===".to_string());
    // filtered if filtered else "(no warnings or errors logged)" —
    // truthiness, not a None check.
    let errors_line = if recent_errors.is_truthy()? {
        py_str(recent_errors)?
    } else {
        "(no warnings or errors logged)".to_string()
    };
    sections.push(errors_line);

    sections.push(String::new());
    sections.push("=== Recent mpv warnings/errors ===".to_string());
    sections.push(py_str(recent_mpv)?);

    sections.push(String::new());
    sections.push("=== Most recent faulthandler trace ===".to_string());
    sections.push(py_str(faulthandler)?);

    sections.push(String::new());
    sections.push("=== Most recent crash file ===".to_string());
    if crash.is_none() {
        sections.push("(no crash files found)".to_string());
    } else {
        // path, content = crash — the reference's tuple unpacking.
        let (path, content) = unpack_two(crash)?;
        sections.push(format!("From: {}", py_str(&path)?));
        sections.push(py_str(&content)?);
        // if note is not None — not a truthiness gate: an empty-string
        // note still lands.
        if !crash_note.is_none() {
            sections.push(py_str(crash_note)?);
        }
    }

    sections.push(String::new());
    sections.push("=== Active hooks ===".to_string());
    sections.push(py_str(hooks)?);
    sections.push(String::new());
    let note = "Note: this section reflects the doctor subcommand's own process. ".to_string()
        + "Hooks are installed inside the long-running ytm TUI process — "
        + "their effect is visible above (crashes/, faulthandler.log).";
    sections.push(note);

    // "\n".join(sections) — then the redaction pass, the same engine
    // the Python side's _redact binds to.
    Ok(redact(&sections.join("\n")))
}

// ---------------------------------------------------------------------------
// Doctor probe engines (mirror ytm_player/utils/doctor.py)
// ---------------------------------------------------------------------------

/// The last *n* lines containing `mpv[`, joined together. Mirrors the pure
/// filtering core of `ytm_player.utils.doctor._recent_mpv_lines` (the
/// existence check and the file read stay in Python): the `mpv[` match is
/// a substring test, not a prefix one; lines keep their trailing newlines
/// so the join reproduces them; the empty match list returns the
/// no-warnings marker before any slice is taken; and the tail follows
/// Python's `matches[-n:]` semantics (see `py_tail_start`).
#[pyfunction]
pub(crate) fn recent_mpv_lines(lines: Vec<String>, n: i64) -> String {
    let matches: Vec<&str> = lines
        .iter()
        .map(String::as_str)
        .filter(|ln| ln.contains("mpv["))
        .collect();
    if matches.is_empty() {
        return "(no mpv warnings/errors logged)".to_string();
    }
    matches[py_tail_start(matches.len(), n)..].concat()
}

/// The `Fatal Python error:` tail extraction behind the Most recent
/// faulthandler trace section of `ytm doctor`. Mirrors the pure core of
/// `ytm_player.utils.doctor._recent_faulthandler` (the existence check
/// and the file read stay in Python): a text that strips to nothing —
/// every character in CPython's `str.isspace` set, the empty string
/// included, so `\x1c`-`\x1f` count where Rust's own `is_whitespace`
/// would stop at White_Space — returns the no-signals marker; otherwise
/// the *last* occurrence of `Fatal Python error:` (`str.rfind`, so the
/// last fatal block wins when the log holds several) opens the returned
/// slice, which runs to the end of the text; and a text without the
/// marker falls back to its last 4000 *characters* — `text[-4000:]`,
/// which is the whole text at or below 4000 and cuts on a character
/// boundary above it (Python slices by code point, not byte, so the
/// 4000-character tail of a multi-byte text can span well over 4000
/// bytes).
#[pyfunction]
pub(crate) fn faulthandler_tail(text: &str) -> String {
    // not text.strip() — empty after stripping iff every character is
    // CPython whitespace (an empty text included, all() on nothing).
    if text.chars().all(py_isspace) {
        return "(faulthandler.log is empty — no fatal signals captured)".to_string();
    }
    match text.rfind("Fatal Python error:") {
        // text[last_idx:] — from the last marker occurrence to the end.
        Some(idx) => text[idx..].to_string(),
        None => {
            // text[-4000:] — the last 4000 characters, the whole text at
            // or below 4000 of them.
            let char_count = text.chars().count();
            if char_count <= 4000 {
                return text.to_string();
            }
            let start = text
                .char_indices()
                .nth(char_count - 4000)
                .map(|(i, _)| i)
                .unwrap_or(0);
            text[start..].to_string()
        }
    }
}

/// The metadata keys `write_crash_file()` stamps at the top of every crash
/// file (`logging._crash_metadata_header`). Only these count as header
/// lines — a traceback line that happens to be `Word: value` shaped (e.g.
/// `ValueError: x`) must not be mistaken for crash metadata. Mirrors
/// `doctor._CRASH_META_KEYS` (case-sensitive membership).
const CRASH_META_KEYS: [&str; 4] = ["version", "time", "python", "platform"];

/// The `(\w+):\s+(\S+)` line shape of the crash-metadata header, with
/// CPython's `\s`/`\S` set spelled out (`\s` is the str.isspace set —
/// White_Space plus `\x1c`–`\x1f`, which the crate's own `\s` disagrees
/// on; `\S` is its exact complement). The key class stays `\w`: where
/// Python's and the crate's word-char sets differ the divergence provably
/// cannot change the verdict — a key in `CRASH_META_KEYS` is pure ASCII,
/// where both sets agree, and any divergent word run is left without its
/// colon on one side, ending the header scan on both.
static CRASH_META_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(\w+):[\s\x{1c}-\x{1f}]+([^\s\x{1c}-\x{1f}]+)").expect("valid crash meta regex")
});

/// Parse a version string into its PEP 440 comparison key, or None when
/// the string is not a valid version — or one of its numbers exceeds
/// CPython's int-conversion digit limit (see `INT_MAX_STR_DIGITS`); both
/// failure kinds read the same to the crash-staleness verdict (no note).
/// *admit_unicode_folds* selects the grammar shape (see `PEP440_FOLD_RE`).
fn parse_pep440(text: &str, admit_unicode_folds: bool) -> Option<Pep440Key> {
    parse_pep440_err(text, admit_unicode_folds).ok()
}

/// Split exactly like CPython's `str.split()` with no separator: on
/// runs of the `isspace` set (see `py_isspace` — a wider set than the
/// crate's own `split_whitespace`, which stops at the Unicode
/// `White_Space` property and misses `\x1c`–`\x1f`), producing no
/// empty pieces — leading and trailing whitespace yields nothing, and
/// an empty string yields no pieces at all.
pub(crate) fn py_split_none(s: &str) -> Vec<&str> {
    s.split(py_isspace)
        .filter(|piece| !piece.is_empty())
        .collect()
}

/// Split exactly like CPython's `str.splitlines`: on every line boundary
/// it accepts (`\n`, `\r\n` as one, `\r`, `\v`, `\f`, `\x1c`, `\x1d`,
/// `\x1e`, `\x85`, `\u{2028}`, `\u{2029}`) — a wider set than the
/// crate's own `str::splitlines` — with a trailing boundary producing no
/// trailing empty line, and an empty string producing no lines at all.
pub(crate) fn py_splitlines(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\n' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}'
            | '\u{2029}' => lines.push(std::mem::take(&mut current)),
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                lines.push(std::mem::take(&mut current));
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Warn when the most recent crash predates the running build. Mirrors
/// `ytm_player.utils.doctor._crash_staleness_note`: only the contiguous
/// known `key: value` metadata header just after the `=== label ===`
/// banner is parsed (a blank line ends it; an unknown or non-key-shaped
/// line ends it — so a traceback's `ValueError: x` is never mistaken for
/// metadata, and a `version:` line buried in the body is never read); a
/// missing version line earns the predates-stamping note; the `unknown`
/// sentinel earns none; and otherwise a PEP 440 comparison decides
/// between the stale-build note and no note. Any version that fails to
/// parse — including a numeric component longer than CPython's
/// int-conversion digit limit — earns no note.
///
/// *admit_unicode_folds* steers the PEP 440 grammar to the shape the
/// installed `packaging` compiled — the Python side probes it once
/// (`Version("1.0+ſ")`: parseable through packaging 26.0, InvalidVersion
/// from 26.1's `(?a:...)` scoping on) and passes the verdict here, so
/// the engine's accept/reject decisions track the reference the
/// fallback itself calls, on every `packaging` the dependency range
/// (`>=21.0`, uncapped) admits.
#[pyfunction]
#[pyo3(signature = (content, current_version, admit_unicode_folds))]
pub(crate) fn crash_staleness_note(
    content: &str,
    current_version: &str,
    admit_unicode_folds: bool,
) -> Option<String> {
    let lines = py_splitlines(content);
    let start = usize::from(lines.first().is_some_and(|l| l.starts_with("===")));
    let mut crash_version: Option<&str> = None;
    for line in &lines[start..] {
        // A blank line (nothing but CPython whitespace) ends the header.
        if line.chars().all(py_isspace) {
            break;
        }
        let meta = CRASH_META_RE.captures(line).and_then(|caps| {
            let key = caps.get(1)?.as_str();
            let value = caps.get(2)?.as_str();
            Some((key, value))
        });
        let Some((key, value)) = meta else {
            break; // not a key: value line -> the header is over
        };
        if !CRASH_META_KEYS.contains(&key) {
            break; // not a known header key (e.g. a traceback line)
        }
        if key == "version" {
            crash_version = Some(value);
            break;
        }
    }

    let Some(crash_version) = crash_version else {
        return Some(
            "⚠ This crash file predates crash-version stamping (no version recorded). If it's \
             from an older build it may already be fixed — reproduce on the current version \
             before reporting."
                .to_string(),
        );
    };
    if crash_version == "unknown" {
        return None;
    }
    let (crash_key, current_key) = (
        parse_pep440(crash_version, admit_unicode_folds),
        parse_pep440(current_version, admit_unicode_folds),
    );
    if crash_key? < current_key? {
        return Some(format!(
            "⚠ This crash was recorded by ytm-player {crash_version}, older than the installed \
             {current_version} — it may already be fixed. Reproduce on the current version \
             before reporting."
        ));
    }
    None
}

/// The process-status text behind the Process status section of
/// `ytm doctor` — the pure core of `_running_status` in
/// `utils/doctor.py`.
///
/// Mirrors the pure-Python reference (`_running_status_text`): the
/// /proc walk, with every side-effecting probe passed in as a callable
/// the engine invokes exactly where the reference performs it — the
/// *proc_exists* thunk (the reference's `Path("/proc").exists()`), the
/// *proc_entries* thunk (the reference's `proc_root.iterdir()`), the
/// per-entry *read_cmdline* and *read_status* thunks (the reference's
/// `(entry / "cmdline").read_bytes()` and `(entry / "status")
/// .read_text(encoding="utf-8", errors="replace")`), and the *own_pid*
/// thunk (the reference's `os.getpid()`, evaluated once per collected
/// pid by the reference's list-comprehension filter — so zero times
/// when nothing was collected, once per match when something was).
///
/// The walk itself: a missing /proc is the cannot-determine marker;
/// every entry whose `.name` attribute's own `isdigit` accepts —
/// CPython's Numeric_Type=Digit-or-Decimal set, which the crate carries
/// no table for, so the verdict runs through the name object's own
/// method, str subclasses included — gets its cmdline read (an
/// `OSError` skips the entry, exactly the reference's swallow; anything
/// else propagates), NUL bytes replaced by spaces, the result decoded
/// like `decode("utf-8", errors="replace")` — one U+FFFD per maximal
/// invalid subpart, the same subpart boundaries CPython's replace
/// handler draws — and split like `str.split()` (the CPython whitespace
/// set, see `py_split_none`); an empty argv skips the entry, a
/// non-ytm argv skips it through the same `argv_is_ytm` engine the
/// Python side's module global resolves to (its own parity tests lock
/// the two spellings together). A match gets its status read (the
/// `OSError`-only swallow again), the first `VmRSS:` line's second
/// split word coerced through CPython's own `int()` — the acceptance
/// set (signed, underscored and non-ASCII digit spellings, the
/// 4300-digit limit) and the `ValueError` swallows matching the
/// reference's exactly, a missing or unparseable word leaving the
/// reference's 0 (see `vmrss_word_kb`) — and the pair `int(entry.name)`
/// builds appended, the pid conversion running through the builtin too
/// so an isdigit-accepted name int() rejects (a superscript digit, say)
/// raises exactly at the reference's append. The filter drops the
/// entry whose pid compares equal to the thunk's (CPython's own `!=`,
/// so an int own-pid compares numerically and anything else is never
/// equal); no survivors is `ytm not running`, otherwise the survivors
/// render as `PID <pid> (RSS <rss // 1024> MB)` parts — the floor
/// division through the int object's own `__floordiv__`, so an
/// unbounded VmRSS stays exact — joined after `ytm running: `.
///
/// The cmdline thunk's result is read as `bytes` and the status
/// thunk's as `str` — the shapes `read_bytes()` and `read_text()`
/// return — and both stay unmutated. The pid and RSS objects are the
/// objects `int()` made; everything else is fresh. The inputs are
/// never mutated.
#[pyfunction]
#[pyo3(signature = (proc_exists, proc_entries, read_cmdline, read_status, own_pid))]
pub(crate) fn running_status_text<'py>(
    py: Python<'py>,
    proc_exists: &Bound<'py, PyAny>,
    proc_entries: &Bound<'py, PyAny>,
    read_cmdline: &Bound<'py, PyAny>,
    read_status: &Bound<'py, PyAny>,
    own_pid: &Bound<'py, PyAny>,
) -> PyResult<String> {
    if !proc_exists.call0()?.is_truthy()? {
        return Ok("(cannot determine — /proc not available on this OS)".to_string());
    }
    // rss_kb = 0 — the reference's starting int, swapped in whenever the
    // VmRSS parse yields nothing.
    let zero_kb: Bound<'py, PyAny> = 0i64.into_pyobject(py)?.into_any();
    // (pid, rss) — both stay the Python objects int() made them, so a
    // pid or VmRSS beyond i64 renders exactly like the reference's
    // f-strings instead of overflowing a fixed-width coercion.
    let mut pids: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
    for entry in proc_entries.call0()?.try_iter()? {
        let entry = entry?;
        // entry.name — the attribute read the reference performs.
        let name = entry.getattr("name")?;
        // entry.name.isdigit() — through the name object's own method
        // (see the doc comment above).
        if !name.call_method0("isdigit")?.is_truthy()? {
            continue;
        }
        // cmdline_path.read_bytes() — the OSError-only swallow.
        let cmdline_raw = match read_cmdline.call1((&entry,)) {
            Ok(raw) => raw,
            Err(err) if err.is_instance_of::<PyOSError>(py) => continue,
            Err(err) => return Err(err),
        };
        // replace(b"\x00", b" ") then decode("utf-8", errors="replace").
        let raw_bytes: Vec<u8> = cmdline_raw.extract()?;
        let spaced: Vec<u8> = raw_bytes
            .iter()
            .map(|&b| if b == 0 { b' ' } else { b })
            .collect();
        let cmdline = String::from_utf8_lossy(&spaced);
        // cmdline.split() — str.split() with no separator.
        let argv: Vec<String> = py_split_none(&cmdline)
            .into_iter()
            .map(str::to_string)
            .collect();
        if argv.is_empty() {
            continue;
        }
        // _argv_is_ytm(argv) — the same engine the Python side's module
        // global resolves to.
        if !argv_is_ytm(argv) {
            continue;
        }
        // (entry / "status").read_text(...) — the OSError-only swallow
        // again.
        let status_text = match read_status.call1((&entry,)) {
            Ok(text) => text,
            Err(err) if err.is_instance_of::<PyOSError>(py) => continue,
            Err(err) => return Err(err),
        };
        let status: String = status_text.extract()?;
        // The VmRSS scan: the first VmRSS-prefixed line decides, and the
        // break is unconditional — a later line never gets a say.
        let mut rss_kb: Option<Bound<'py, PyAny>> = None;
        for line in py_splitlines(&status) {
            if line.starts_with("VmRSS:") {
                rss_kb = Some(vmrss_word_kb(py, &line)?);
                break;
            }
        }
        // int(entry.name) — through the builtin, at the reference's
        // append-time position.
        let pid = builtin_int(py, &name)?;
        let rss = rss_kb.unwrap_or_else(|| zero_kb.clone());
        pids.push((pid, rss));
    }
    // pids = [(pid, rss) for pid, rss in pids if pid != os.getpid()] —
    // the reference's comprehension evaluates the getpid call once per
    // collected pid, so the thunk is invoked per element here too.
    let mut kept: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
    for (pid, rss) in pids {
        let own = own_pid.call0()?;
        // pid != own — CPython's own != through PyObject_RichCompare.
        let differs = pid.rich_compare(&own, CompareOp::Ne)?.is_truthy()?;
        if differs {
            kept.push((pid, rss));
        }
    }
    if kept.is_empty() {
        return Ok("ytm not running".to_string());
    }
    // parts = [f"PID {pid} (RSS {rss // 1024} MB)" ...] — the floor
    // division through the int object's own __floordiv__, the
    // interpolation through str() exactly like the reference's f-strings.
    let mut parts: Vec<String> = Vec::with_capacity(kept.len());
    for (pid, rss) in &kept {
        let mb = rss.call_method1("__floordiv__", (1024,))?;
        parts.push(format!("PID {} (RSS {} MB)", py_str(pid)?, py_str(&mb)?));
    }
    Ok("ytm running: ".to_string() + &parts.join(", "))
}

/// `int(line.split()[1])` — the VmRSS value word — with the reference's
/// `(IndexError, ValueError)` swallow back to 0: a line whose split
/// yields fewer than two words is the IndexError arm, a word CPython's
/// `int()` rejects is the ValueError one, and anything `int()` accepts
/// is the value — the object itself, unbounded, never a fixed-width
/// coercion (see `builtin_int`).
fn vmrss_word_kb<'py>(py: Python<'py>, line: &str) -> PyResult<Bound<'py, PyAny>> {
    let words = py_split_none(line);
    let Some(word) = words.get(1) else {
        // IndexError — line.split()[1] on a one-word (or empty) line.
        return Ok(0i64.into_pyobject(py)?.into_any());
    };
    let word_obj = PyString::new(py, word);
    match builtin_int(py, &word_obj) {
        Ok(value) => Ok(value),
        // ValueError — a word int() cannot parse ("5,368", "N/A", a run
        // beyond the 4300-digit limit, ...).
        Err(err) if err.is_instance_of::<PyValueError>(py) => {
            Ok(0i64.into_pyobject(py)?.into_any())
        }
        Err(err) => Err(err),
    }
}

/// The `mpv:` version line behind the mpv status probe of `ytm doctor`
/// — the pure core of `_mpv_version` in `utils/doctor.py`.
///
/// Mirrors the pure-Python reference (`_mpv_version_text`): the
/// *which_mpv* thunk — the reference's `shutil.which("mpv")` PATH
/// probe, invoked exactly once, exactly up front — and a falsy result
/// (an unset `None` or an empty string alike, the reference's
/// `if not mpv_bin:`) is the NOT FOUND marker, the *run_version* thunk
/// never invoked; otherwise the *run_version* thunk — the reference's
/// `subprocess.run([mpv_bin, "--version"], ...)` — runs with the
/// resolved binary, and the version line comes from its result: the
/// `stdout or stderr or ""` or-chain picks the first truthy output —
/// `stderr` read only when `stdout` is falsy, exactly like the
/// reference's or-chain, so a result whose `stderr` attribute raises
/// never raises when `stdout` already satisfied it — the chosen
/// output's own `splitlines()` supplies the lines (CPython's line
/// boundaries, `\x85` and `\u{2028}` included, subclasses included),
/// the `[0:1]` slice the first of them, the `unknown` stand-in when no
/// line made it, and the f-string the `mpv: ` prefix. An `OSError` or
/// `subprocess.SubprocessError` raised anywhere in the run or the
/// parse — the whole span the reference's try block covers — is the
/// failed-to-execute marker (see `is_subprocess_error`); anything else
/// propagates. The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (which_mpv, run_version))]
pub(crate) fn mpv_version_text<'py>(
    py: Python<'py>,
    which_mpv: &Bound<'py, PyAny>,
    run_version: &Bound<'py, PyAny>,
) -> PyResult<String> {
    let mpv_bin = which_mpv.call0()?;
    if !mpv_bin.is_truthy()? {
        return Ok("mpv: NOT FOUND in PATH".to_string());
    }
    match mpv_version_attempt(py, run_version, &mpv_bin) {
        Ok(line) => Ok(line),
        // except (OSError, subprocess.SubprocessError) — the reference's
        // clause, both arms through CPython's own exception matching.
        Err(err) if err.is_instance_of::<PyOSError>(py) || is_subprocess_error(py, &err)? => {
            Ok("mpv: failed to execute".to_string())
        }
        Err(err) => Err(err),
    }
}

/// The run-and-parse half of the mpv version probe — everything the
/// reference's try block covers, so an `OSError` or
/// `subprocess.SubprocessError` from any of it lands in the
/// failed-to-execute arm of the caller.
fn mpv_version_attempt<'py>(
    py: Python<'py>,
    run_version: &Bound<'py, PyAny>,
    mpv_bin: &Bound<'py, PyAny>,
) -> PyResult<String> {
    let out = run_version.call1((mpv_bin,))?;
    // out.stdout — always read; out.stderr — only when stdout is falsy,
    // exactly like the reference's or-chain.
    let stdout = out.getattr("stdout")?;
    let chosen = if stdout.is_truthy()? {
        stdout
    } else {
        let stderr = out.getattr("stderr")?;
        if stderr.is_truthy()? {
            stderr
        } else {
            // "" — the or-chain's literal; its splitlines() is always
            // [], so the [0:1] slice is empty and the marker is unknown.
            return Ok("mpv: unknown".to_string());
        }
    };
    // .splitlines() — the chosen output's own method.
    let lines = chosen.call_method0("splitlines")?;
    // [0:1] — the slice, then its truthiness, then first_line[0].
    let first_line = lines.get_item(PySlice::new(py, 0, 1, 1))?;
    if first_line.is_truthy()? {
        let first = first_line.get_item(0)?;
        Ok(format!("mpv: {}", py_str(&first)?))
    } else {
        Ok("mpv: unknown".to_string())
    }
}

/// `isinstance(err, subprocess.SubprocessError)` — the second arm of
/// the mpv probe's except clause, through the module's own exception
/// root (`subprocess.TimeoutExpired` and `CalledProcessError` inherit
/// it, so the reference's clause catches them too).
fn is_subprocess_error(py: Python<'_>, err: &PyErr) -> PyResult<bool> {
    let subprocess_error = py.import("subprocess")?.getattr("SubprocessError")?;
    Ok(err.is_instance(py, &subprocess_error))
}

/// The libmpv status line behind the libmpv loadability probe of
/// `ytm doctor` — the pure core of `_libmpv_status` in
/// `utils/doctor.py`.
///
/// Mirrors the pure-Python reference (`_libmpv_status_text`): the
/// *import_player* thunk — the reference's
/// `from ytm_player.services.player import _IMPORT_ERROR_MSG` — is
/// invoked exactly once, exactly up front, and an `ImportError` it
/// raises (whatever arm produced it: the module's own import failing,
/// or the name being absent because the mpv import succeeded) is the
/// OK marker, the library loadable; otherwise the message the thunk
/// returns is rendered through `str()` exactly like the reference's
/// `str(_IMPORT_ERROR_MSG)`, its own `splitlines()` supplies the lines
/// (CPython's line boundaries, `\x85` and `\u{2028}` included), the
/// `[-1]` index the last of them (an `IndexError` when none made it,
/// propagating exactly like the reference's — the except clause covers
/// `ImportError` only), and the `strip()`-ed line (Python's whitespace
/// set, see `py_strip`) interpolates into the NOT LOADABLE message.
/// Everything the reference's try block covered stays under the same
/// single `except ImportError` here — a message object whose own
/// `__str__` raises `ImportError` is the OK marker too (see
/// `libmpv_status_attempt`), exactly like the reference's. The inputs
/// are never mutated.
#[pyfunction]
#[pyo3(signature = (import_player))]
pub(crate) fn libmpv_status_text<'py>(
    py: Python<'py>,
    import_player: &Bound<'py, PyAny>,
) -> PyResult<String> {
    match libmpv_status_attempt(import_player) {
        Ok(text) => Ok(text),
        // except ImportError — the reference's only clause, matched
        // through the exception type itself.
        Err(err) if err.is_instance_of::<PyImportError>(py) => Ok("libmpv: OK".to_string()),
        Err(err) => Err(err),
    }
}

/// The import-and-render half of the libmpv probe — everything the
/// reference's try block covers, so an `ImportError` from any of it
/// (the thunk itself, or a message object's own `__str__`) lands in the
/// OK arm of the caller, and anything else propagates.
fn libmpv_status_attempt(import_player: &Bound<'_, PyAny>) -> PyResult<String> {
    // from ytm_player.services.player import _IMPORT_ERROR_MSG — the
    // thunk, exactly once, exactly up front.
    let import_error_msg = import_player.call0()?;
    // str(_IMPORT_ERROR_MSG) — the reference's conversion, inside its
    // try block, so an ImportError it raises is the OK marker.
    let text = py_str(&import_error_msg)?;
    // .splitlines() — CPython's line boundaries (see py_splitlines).
    let lines = py_splitlines(&text);
    // [-1] — the last line; an empty list is the reference's own
    // IndexError (the except clause covers ImportError only), reproduced
    // through CPython's own message.
    let last = lines
        .last()
        .ok_or_else(|| PyIndexError::new_err("list index out of range"))?;
    // first.strip() — Python's whitespace set (see py_strip).
    Ok(format!("libmpv: NOT LOADABLE ({})", py_strip(last)))
}

/// The MPRIS status line behind the MPRIS availability probe of
/// `ytm doctor` — the pure core of `_mpris_status` in
/// `utils/doctor.py`.
///
/// Mirrors the pure-Python reference (`_mpris_status_text`): a falsy
/// *is_linux* (the reference's `sys.platform != "linux"`, computed by
/// the caller) is the n/a marker, the *system_name* thunk — the
/// reference's `platform.system()` — invoked exactly here, exactly
/// once, only on this branch, so the platform probe never runs on
/// Linux — rendering into the message through `str()` exactly like the
/// reference's f-string; otherwise the *import_mpris* thunk — the
/// reference's `from ytm_player.services.mpris import BUS_NAME,
/// DBUS_AVAILABLE` — supplies the pair, unpacked with CPython's own
/// `UNPACK_SEQUENCE(2)` semantics (see `unpack_pair`), and a truthy
/// `DBUS_AVAILABLE` element is the available line interpolating the
/// `BUS_NAME` element through `str()` exactly like the reference's
/// f-string, a falsy one the UNAVAILABLE fix text. An error from
/// either thunk propagates exactly where the reference's own operation
/// would raise it — the reference has no try around the import, so an
/// `ImportError` surfaces instead of becoming the unavailable marker.
/// The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (is_linux, system_name, import_mpris))]
pub(crate) fn mpris_status_text<'py>(
    py: Python<'py>,
    is_linux: &Bound<'py, PyAny>,
    system_name: &Bound<'py, PyAny>,
    import_mpris: &Bound<'py, PyAny>,
) -> PyResult<String> {
    if !is_linux.is_truthy()? {
        // platform.system() — invoked exactly here, exactly once, only
        // on this branch, exactly like the reference's f-string.
        let system = system_name.call0()?;
        return Ok(format!(
            "MPRIS: n/a ({} uses its native media integration)",
            py_str(&system)?
        ));
    }
    // from ytm_player.services.mpris import BUS_NAME, DBUS_AVAILABLE —
    // the thunk supplies the pair; the unpack is CPython's own (see
    // unpack_pair).
    let pair = import_mpris.call0()?;
    let (bus_name, dbus_available) = unpack_pair(py, &pair)?;
    if dbus_available.is_truthy()? {
        Ok(format!(
            "MPRIS: available (dbus-fast OK; registers as {})",
            py_str(&bus_name)?
        ))
    } else {
        Ok(
            "MPRIS: UNAVAILABLE — dbus-fast is missing or failed to load (it's a \
             Linux core dependency, so this install is broken/partial; an \
             incompatible dbus-fast build logs the failure to ytm.log — see the \
             recent-errors section below). Fix: reinstall ytm-player \
             (pip install --force-reinstall ytm-player), or pip install dbus-fast."
                .to_string(),
        )
    }
}

/// `a, b = obj` — CPython's `UNPACK_SEQUENCE(2)` exactly: the
/// exact-tuple and exact-list fast paths (a *subclass* of either does
/// not take them — its own `__iter__` runs, exactly like the opcode's
/// slow path), otherwise the iterator protocol with the opcode's own
/// error shaping — `ValueError: not enough values to unpack (expected
/// 2, got N)` when the iterator ends early, `ValueError: too many
/// values to unpack (expected 2)` when it keeps going (on 3.14 the
/// exact types carry the actual count — see `unpack_too_many_message`),
/// and a non-iterable reporting `TypeError: cannot unpack non-iterable <type>
/// object`, the opcode's reformat of `GetIter`'s own TypeError.
fn unpack_pair<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    // PyTuple_CheckExact(seq) && PyTuple_GET_SIZE(seq) == 2 — the fast
    // path reads the items directly, no __iter__ dispatch.
    if let Ok(tuple) = obj.downcast_exact::<PyTuple>() {
        if tuple.len() == 2 {
            return Ok((tuple.get_item(0)?, tuple.get_item(1)?));
        }
    }
    // PyList_CheckExact(seq) && PyList_GET_SIZE(seq) == 2 — the same.
    if let Ok(list) = obj.downcast_exact::<PyList>() {
        if list.len() == 2 {
            return Ok((list.get_item(0)?, list.get_item(1)?));
        }
    }
    // The slow path — PyObject_GetIter, with the opcode's reformat of a
    // non-iterable's TypeError.
    let mut iter = match obj.try_iter() {
        Ok(iter) => iter,
        Err(err) => {
            if err.is_instance_of::<PyTypeError>(py) {
                let type_name = obj.get_type().name()?;
                return Err(PyTypeError::new_err(format!(
                    "cannot unpack non-iterable {type_name} object"
                )));
            }
            return Err(err);
        }
    };
    // Collect exactly two items, counting the successes for the
    // not-enough message.
    let mut items: [Option<Bound<'py, PyAny>>; 2] = [None, None];
    let mut got = 0usize;
    for slot in &mut items {
        match iter.next() {
            Some(Ok(item)) => {
                *slot = Some(item);
                got += 1;
            }
            Some(Err(err)) => return Err(err),
            None => break, // StopIteration before the pair was filled
        }
    }
    let [first, second] = items;
    match (first, second) {
        (Some(a), Some(b)) => {
            // The extra-item probe: a third item is too many, the
            // iterator ending is the reference's success.
            match iter.next() {
                Some(Ok(_)) => Err(PyValueError::new_err(unpack_too_many_message(py, obj, 2))),
                Some(Err(err)) => Err(err),
                None => Ok((a, b)),
            }
        }
        _ => Err(PyValueError::new_err(format!(
            "not enough values to unpack (expected 2, got {got})"
        ))),
    }
}
