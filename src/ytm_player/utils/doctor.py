"""Diagnostic gathering for the `ytm doctor` command.

Produces a single-string report users can paste directly into a GitHub
issue. It covers every failure class: version, paths, process status,
MPRIS / media keys, recent ERROR/WARNING, recent mpv warnings,
faulthandler trace, last crash file, active hooks. All output passes through
redaction for recognized auth headers/tokens and HTTP(S)/TCP URLs. This is
not a guarantee that arbitrary secrets embedded in diagnostic text are removed.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from packaging.version import Version

# Try to load the native Rust implementation via PyO3.  If the compiled
# extension is unavailable (e.g. during development without a Rust toolchain),
# fall back to the pure-Python implementations defined below.
try:
    from ._ytm_player_native import (  # type: ignore[import-not-found]
        argv_is_ytm as _native_argv_is_ytm,
    )
    from ._ytm_player_native import (
        redact as _native_redact,
    )
except ImportError:
    _native_argv_is_ytm = None
    _native_redact = None

# Stream URLs may contain the public IP and signed playback parameters.
# Remove the whole URL rather than guessing which query fields are sensitive.
_URL_PATTERN = re.compile(r"\b(?:https?|tcp)://[^\s<>\"']+", re.IGNORECASE)

_REDACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Match header-value pairs — consume the rest of the line to catch multi-word tokens
    re.compile(r"(authorization\s*[:=]\s*)(.+)", re.IGNORECASE),
    re.compile(r"(cookie\s*[:=]\s*)(.+)", re.IGNORECASE),
    re.compile(r"(bearer\s+)(\S+)", re.IGNORECASE),
    re.compile(r"(token\s*[:=]\s*)(\S+)", re.IGNORECASE),
    re.compile(r"(x-goog-pageid\s*[:=]\s*)(\S+)", re.IGNORECASE),
    re.compile(r"(SAPISID\s*=\s*)(\S+)"),
)


def _redact(text: str) -> str:
    if _native_redact is not None:
        return _native_redact(text)
    for pat in _REDACT_PATTERNS:
        text = pat.sub(r"\1[REDACTED]", text)
    return _URL_PATTERN.sub("[URL REDACTED]", text)


def _mpv_version_text(which_mpv: Any, run_version: Any) -> str:
    """The mpv version probe text behind ``_mpv_version``.

    The pure half of the ``mpv:`` status line: the *which_mpv* thunk —
    the reference's ``shutil.which("mpv")`` PATH probe — is invoked
    exactly once, exactly up front, and a falsy result (an unset
    ``None`` or an empty string alike, the reference's ``if not
    mpv_bin:``) is the NOT FOUND marker, the *run_version* thunk never
    invoked; otherwise the *run_version* thunk — the reference's
    ``subprocess.run([mpv_bin, "--version"], ...)`` — runs with the
    resolved binary, and the version line comes from its result: the
    ``stdout or stderr or ""`` or-chain picks the first truthy output
    (``stderr`` is read only when ``stdout`` is falsy, exactly like
    the reference's or-chain, so a result whose ``stderr`` attribute
    raises never raises when ``stdout`` already satisfied it), the
    chosen output's own ``splitlines()`` supplies the lines, the
    ``[0:1]`` slice the first of them, and the ``unknown`` stand-in
    when no line made it. An ``OSError`` or
    ``subprocess.SubprocessError`` raised anywhere in the run or the
    parse is the failed-to-execute marker; anything else propagates.

    Implemented by the Rust extension
    (``_ytm_player_native.mpv_version_text``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    mpv_bin = which_mpv()
    if not mpv_bin:
        return "mpv: NOT FOUND in PATH"
    try:
        out = run_version(mpv_bin)
        first_line = (out.stdout or out.stderr or "").splitlines()[0:1]
        return f"mpv: {first_line[0] if first_line else 'unknown'}"
    except (OSError, subprocess.SubprocessError):
        return "mpv: failed to execute"


def _mpv_version() -> str:
    """Return mpv version string, or a clear missing marker.

    The version probe — the PATH lookup, the ``--version`` run, the
    first-line extraction and the failure markers — is resolved by
    the ``_mpv_version_text`` engine above (native when the Rust
    extension is built, the pure-Python transcription otherwise);
    this helper supplies the probes as thunks.
    """
    return _mpv_version_text(
        lambda: shutil.which("mpv"),
        lambda mpv_bin: subprocess.run(
            [mpv_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ),
    )


def _libmpv_status_text(import_player: Any) -> str:
    """The libmpv status text behind ``_libmpv_status``.

    The pure half of the libmpv loadability probe: the *import_player*
    thunk — the reference's ``from ytm_player.services.player import
    _IMPORT_ERROR_MSG`` — is invoked exactly once, exactly up front,
    and an ``ImportError`` it raises (whatever arm produced it: the
    module's own import failing, or the name being absent because the
    mpv import succeeded) is the OK marker, the library loadable;
    otherwise the message the thunk returns is rendered through
    ``str()`` exactly like the reference's ``str(_IMPORT_ERROR_MSG)``,
    its own ``splitlines()`` supplies the lines (CPython's line
    boundaries, ``\\x85`` and ``\\u2028`` included), the ``[-1]``
    index the last of them (an ``IndexError`` when none made it,
    propagating exactly like the reference's — the except clause
    covers ``ImportError`` only), and the ``strip()``-ed line
    interpolates into the NOT LOADABLE message. Everything the
    reference's try block covered stays under the same single
    ``except ImportError`` here — a message object whose ``__str__``
    itself raises ``ImportError`` is the OK marker too, exactly like
    the reference's.

    Implemented by the Rust extension
    (``_ytm_player_native.libmpv_status_text``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    try:
        import_error_msg = import_player()
        first = str(import_error_msg).splitlines()[-1]
        return f"libmpv: NOT LOADABLE ({first.strip()})"
    except ImportError:
        return "libmpv: OK"


def _libmpv_status() -> str:
    """Report whether the libmpv shared library is loadable.

    The mpv CLI being on PATH does not imply libmpv is discoverable —
    Homebrew installs (macOS and Linux) put the library where
    ctypes.util.find_library can't see it from a non-brew Python
    (#90, #101, #104). Importing our player module exercises the full
    discovery chain including the brew-prefix fallback.

    The probe — the player-module import and the last-line render — is
    resolved by the ``_libmpv_status_text`` engine above (native when
    the Rust extension is built, the pure-Python transcription
    otherwise); this helper supplies the import as a thunk.
    """

    def _import_player() -> Any:
        from ytm_player.services.player import _IMPORT_ERROR_MSG  # type: ignore[attr-defined]

        return _IMPORT_ERROR_MSG

    return _libmpv_status_text(_import_player)


def _mpris_status_text(is_linux: Any, system_name: Any, import_mpris: Any) -> str:
    """The MPRIS status text behind ``_mpris_status``.

    The pure half of the MPRIS availability probe: a falsy *is_linux*
    (the reference's ``sys.platform != "linux"``, computed by the
    caller) is the n/a marker, the *system_name* thunk — the
    reference's ``platform.system()`` — invoked exactly here, exactly
    once, only on this branch, so the platform probe never runs on
    Linux — rendering into the message through ``str()`` exactly like
    the reference's f-string; otherwise the *import_mpris* thunk — the
    reference's ``from ytm_player.services.mpris import BUS_NAME,
    DBUS_AVAILABLE`` — supplies the pair, and a truthy
    ``DBUS_AVAILABLE`` element is the available line interpolating the
    ``BUS_NAME`` element through ``str()`` exactly like the reference's
    f-string, a falsy one the UNAVAILABLE fix text. An error from
    either thunk propagates exactly where the reference's own operation
    would raise it — the reference has no try around the import, so an
    ``ImportError`` surfaces instead of becoming the unavailable
    marker. The inputs are never mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.mpris_status_text``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if not is_linux:
        return f"MPRIS: n/a ({system_name()} uses its native media integration)"
    bus_name, dbus_available = import_mpris()
    if dbus_available:
        return f"MPRIS: available (dbus-fast OK; registers as {bus_name})"
    return (
        "MPRIS: UNAVAILABLE — dbus-fast is missing or failed to load (it's a "
        "Linux core dependency, so this install is broken/partial; an "
        "incompatible dbus-fast build logs the failure to ytm.log — see the "
        "recent-errors section below). Fix: reinstall ytm-player "
        "(pip install --force-reinstall ytm-player), or pip install dbus-fast."
    )


def _mpris_status() -> str:
    """Report whether MPRIS (playerctl / media keys) can run on this install.

    MPRIS needs dbus-fast, a Linux core dependency. If it's missing — or its
    build is incompatible and blew up while our interfaces were defined (#113) —
    the install is broken/partial, the TUI runs without D-Bus controls, and
    playerctl finds no player (#110) — so surface the gap and the fix here.
    doctor runs in its own process, so this reflects dependency availability —
    not whether the live TUI currently holds the bus name.

    The verdict — the platform gate, the system render, the import and
    the three spellings — is resolved by the ``_mpris_status_text``
    engine above (native when the Rust extension is built, the
    pure-Python transcription otherwise); this helper supplies the
    platform facts and the import as thunks.
    """

    def _import_mpris() -> tuple[Any, Any]:
        from ytm_player.services.mpris import BUS_NAME, DBUS_AVAILABLE

        return BUS_NAME, DBUS_AVAILABLE

    return _mpris_status_text(
        sys.platform == "linux",
        platform.system,
        _import_mpris,
    )


def _argv_is_ytm(argv: list[str]) -> bool:
    """True when *argv* is a real ytm-player entry point.

    Matches the `ytm` console script (which the kernel launches as
    ``python /…/bin/ytm`` — interpreter in argv[0], script path in argv[1])
    and ``python -m ytm_player``. Deliberately does NOT match unrelated
    processes whose cmdline merely contains a path like /ytm-player or
    …/ytm_player/… (e.g. an editor with the repo open).
    """
    if _native_argv_is_ytm is not None:
        return _native_argv_is_ytm(argv)
    head_names = [Path(a).name for a in argv[:2]]
    return "ytm" in head_names or "ytm_player" in argv


def _running_status_text(
    proc_exists: Any,
    proc_entries: Any,
    read_cmdline: Any,
    read_status: Any,
    own_pid: Any,
) -> str:
    """The process-status text behind ``_running_status``.

    The pure half of the Process status section: the /proc walk with
    every side-effecting probe passed in as a callable invoked exactly
    where the reference performs it — the *proc_exists* thunk (the
    reference's ``Path("/proc").exists()``), the *proc_entries* thunk
    (the reference's ``proc_root.iterdir()``), the per-entry
    *read_cmdline* and *read_status* thunks (the reference's
    ``(entry / "cmdline").read_bytes()`` and ``(entry / "status")
    ``.read_text(encoding="utf-8", errors="replace")``), and the
    *own_pid* thunk (the reference's ``os.getpid()``, evaluated once
    per collected pid by the reference's list-comprehension filter —
    so zero times when nothing was collected, once per match when
    something was).

    The walk itself: a missing /proc is the cannot-determine marker;
    every entry whose ``name`` attribute's own ``isdigit`` accepts gets
    its cmdline read (an ``OSError`` skips the entry, exactly the
    reference's swallow; anything else propagates), the NUL bytes
    replaced by spaces, the result decoded with ``errors="replace"``
    and split like ``str.split()``; an empty argv skips the entry, a
    non-ytm argv skips it through ``_argv_is_ytm``. A match gets its
    status read (the ``OSError``-only swallow again), the first
    ``VmRSS:`` line's second split word coerced through ``int()``
    (the ``(IndexError, ValueError)`` swallow back to 0 included),
    and the pair ``int(entry.name)`` builds appended. The filter drops
    the entry whose pid compares equal to the thunk's; no survivors
    is ``ytm not running``, otherwise the survivors render as
    ``PID <pid> (RSS <rss // 1024> MB)`` parts joined after
    ``ytm running: ``.

    Implemented by the Rust extension
    (``_ytm_player_native.running_status_text``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if not proc_exists():
        return "(cannot determine — /proc not available on this OS)"
    pids: list[tuple[Any, Any]] = []
    for entry in proc_entries():
        if not entry.name.isdigit():
            continue
        try:
            cmdline_raw = read_cmdline(entry)
        except OSError:
            continue
        cmdline = cmdline_raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
        argv = cmdline.split()
        if not argv:
            continue
        if _argv_is_ytm(argv):
            try:
                status = read_status(entry)
            except OSError:
                continue
            rss_kb: Any = 0
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    try:
                        rss_kb = int(line.split()[1])
                    except (IndexError, ValueError):
                        rss_kb = 0
                    break
            pids.append((int(entry.name), rss_kb))
    pids = [(pid, rss) for pid, rss in pids if pid != own_pid()]
    if not pids:
        return "ytm not running"
    parts = [f"PID {pid} (RSS {rss // 1024} MB)" for pid, rss in pids]
    return "ytm running: " + ", ".join(parts)


def _running_status() -> str:
    """Best-effort check whether a ytm TUI process is currently running.

    Uses /proc on Linux (no psutil dependency); on other platforms or if
    /proc is unavailable, returns a 'cannot determine' marker.

    The /proc walk — the digit-name filter, the cmdline decode and
    argv match, the VmRSS parse, the own-pid exclusion and the
    running/not-running verdicts — is resolved by the
    ``_running_status_text`` engine above (native when the Rust
    extension is built, the pure-Python transcription otherwise);
    this helper supplies the /proc probes as thunks.
    """
    proc_root = Path("/proc")
    return _running_status_text(
        proc_root.exists,
        proc_root.iterdir,
        lambda entry: (entry / "cmdline").read_bytes(),
        lambda entry: (entry / "status").read_text(encoding="utf-8", errors="replace"),
        os.getpid,
    )


def _mpv_lines(lines: list[str], n: int) -> str:
    """The mpv-warning tail behind ``_recent_mpv_lines``.

    The pure half of the Recent mpv warnings/errors probe: the log
    lines containing ``mpv[`` (our log_handler prefix — a substring
    test, not a prefix one), the empty-match substitution of the
    no-warnings marker (decided before any slice is taken), the last
    *n* of the matches under Python's ``[-n:]`` tail semantics (zero
    keeps everything — ``[-0:]`` is ``[0:]`` — and a negative n drops
    the leading entries), and the separator-less join, the line endings
    coming along inside the lines themselves. The input lines are
    never mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.recent_mpv_lines``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    matches = [ln for ln in lines if "mpv[" in ln]
    if not matches:
        return "(no mpv warnings/errors logged)"
    return "".join(matches[-n:])


def _recent_mpv_lines(log_file: Path, n: int = 20) -> str:
    """Return the last *n* log lines that contain 'mpv[' (our log_handler prefix).

    The mpv-warning tail — the ``mpv[`` filter, the last-n slice and
    the join — is resolved by the ``_mpv_lines`` engine above (native
    when the Rust extension is built, the pure-Python transcription
    otherwise); this helper performs the reads.
    """
    if not log_file.exists():
        return "(log file is empty or missing)"
    try:
        with open(log_file, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return "(could not read log file)"
    return _mpv_lines(lines, n)


def _faulthandler_tail(text: str) -> str:
    """The ``Fatal Python error:`` tail extraction behind
    ``_recent_faulthandler``.

    The pure half of the Most recent faulthandler trace probe: the
    empty-after-strip check (a text that is nothing but Python
    whitespace — the empty string included — is the no-signals
    marker), the ``rfind`` of the ``Fatal Python error:`` marker, the
    marker-to-end slice of the *last* fatal block when it is found,
    and the last-4000-characters tail (``text[-4000:]`` — the whole
    text when shorter, character- not byte-wise so a multi-byte tail
    cuts on a code-point boundary) when it is not. The input is
    never mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.faulthandler_tail``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if not text.strip():
        return "(faulthandler.log is empty — no fatal signals captured)"
    marker = "Fatal Python error:"
    last_idx = text.rfind(marker)
    if last_idx == -1:
        return text[-4000:]
    return text[last_idx:]


def _recent_faulthandler(crash_dir: Path) -> str:
    """Return the last 'Fatal Python error' block from crash_dir/faulthandler.log,
    or a marker if the file is missing/empty.

    The tail extraction — the empty-text check, the ``rfind`` of the
    marker and the two slices — is resolved by the
    ``_faulthandler_tail`` engine above (native when the Rust
    extension is built, the pure-Python transcription otherwise);
    this helper performs the reads.
    """
    fh_log = crash_dir / "faulthandler.log"
    if not fh_log.exists():
        return "(no faulthandler trace — file does not exist)"
    try:
        text = fh_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(could not read faulthandler.log)"
    return _faulthandler_tail(text)


# The metadata keys write_crash_file() stamps at the top of every crash file
# (logging._crash_metadata_header). Only these count as header lines — a
# traceback line that happens to be "Word: value" shaped (e.g. "ValueError: x")
# must not be mistaken for crash metadata.
_CRASH_META_KEYS = frozenset({"version", "time", "python", "platform"})


def _crash_staleness_note(content: str, current_version: str) -> str | None:
    """Warn when the most recent crash predates the running build.

    Crash files record a ``version:`` line in a metadata header (see
    ``write_crash_file``). If it names a version older than the one installed,
    the crash may already be fixed — flag it so a stale log isn't mistaken for
    a live bug. Crash files written before stamping existed have no version
    line and get a softer note.

    Only the metadata header — the contiguous known ``key: value`` lines just
    after the ``=== label ===`` banner — is parsed. Scanning the whole body, or
    accepting any key-shaped line, would let an exception line (``ValueError:
    x``) or a ``version:``-shaped traceback line produce a bogus verdict.

    The PEP 440 comparison is whatever the installed ``packaging`` says it
    is: through 26.0 its grammar admitted the simple Unicode case folds
    (``1.0+ſ``, ``1.0prevıew`` parse), 26.1+ scopes them out — so the
    verdict for such spellings follows the installed release. The native
    engine probes the same behaviour once and mirrors it (see
    ``_packaging_admits_unicode_folds``).

    Implemented by the Rust extension
    (``_ytm_player_native.crash_staleness_note``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    lines = content.splitlines()
    start = 1 if lines and lines[0].startswith("===") else 0
    crash_version: str | None = None
    for line in lines[start:]:
        if not line.strip():
            break  # blank line ends the metadata header
        meta = re.match(r"(\w+):\s+(\S+)", line)
        if meta is None or meta.group(1) not in _CRASH_META_KEYS:
            break  # not a known header key -> header is over (e.g. a traceback line)
        if meta.group(1) == "version":
            crash_version = meta.group(2)
            break
    if crash_version is None:
        return (
            "⚠ This crash file predates crash-version stamping (no version "
            "recorded). If it's from an older build it may already be fixed — "
            "reproduce on the current version before reporting."
        )
    if crash_version == "unknown":
        return None
    # InvalidVersion is a ValueError subclass; the wider catch also covers
    # the plain ValueError CPython's int(str) raises for numeric components
    # beyond the ~4300-digit conversion limit, so an absurd version in a
    # corrupted crash file is a no-verdict instead of a doctor crash (the
    # native engine enforces the same limit and never raises either).
    try:
        if Version(crash_version) < Version(current_version):
            return (
                f"⚠ This crash was recorded by ytm-player {crash_version}, "
                f"older than the installed {current_version} — it may already be "
                "fixed. Reproduce on the current version before reporting."
            )
    except ValueError:
        return None
    return None


def _doctor_report(
    version: Any,
    py_major: Any,
    py_minor: Any,
    py_micro: Any,
    system: Any,
    release: Any,
    machine: Any,
    mpv_status: Any,
    libmpv_status: Any,
    config_path: Any,
    theme_path: Any,
    session_path: Any,
    log_path: Any,
    crash_dir: Any,
    process_status: Any,
    mpris_status: Any,
    recent_errors: Any,
    recent_mpv: Any,
    faulthandler: Any,
    crash: Any,
    crash_note: Any,
    hooks: Any,
) -> str:
    """The report assembly behind ``gather_diagnostics``.

    The pure half of ``ytm doctor``: the multi-section report text
    from the probes' results — the section banners in their order,
    the aligned identity labels (``Version:``/``Python:``/``OS:``/
    ``Machine:``) and the paths block, the probes' own lines appended
    raw, the empty-log placeholder, the crash-file section (``None``
    is the no-crash-files marker; a ``(path, content)`` pair unpacks
    into the ``From:`` line, the content and the optional staleness
    note), the trailing hooks note, the final ``"\\n"`` join, and
    the redaction pass.

    Implemented by the Rust extension
    (``_ytm_player_native.doctor_report``) when it is built; this
    pure-Python transcription is the fallback otherwise. The
    interpolations run through ``str()`` exactly like the reference's
    f-strings (``Path`` objects render as their path text, the
    version triple as its numbers). Never mutates its inputs.
    """
    sections: list[str] = []

    sections.append("=== ytm-player diagnostics ===")
    sections.append(f"Version: {version}")
    sections.append(f"Python:  {py_major}.{py_minor}.{py_micro}")
    sections.append(f"OS:      {system} {release}")
    sections.append(f"Machine: {machine}")
    sections.append(mpv_status)
    sections.append(libmpv_status)

    sections.append("")
    sections.append("=== Paths ===")
    sections.append(f"config:   {config_path}")
    sections.append(f"theme:    {theme_path}")
    sections.append(f"session:  {session_path}")
    sections.append(f"log:      {log_path}")
    sections.append(f"crashes:  {crash_dir}")

    sections.append("")
    sections.append("=== Process status ===")
    sections.append(process_status)

    sections.append("")
    sections.append("=== MPRIS / media keys ===")
    sections.append(mpris_status)

    sections.append("")
    sections.append("=== Recent ERROR/WARNING (last 20) ===")
    sections.append(recent_errors if recent_errors else "(no warnings or errors logged)")

    sections.append("")
    sections.append("=== Recent mpv warnings/errors ===")
    sections.append(recent_mpv)

    sections.append("")
    sections.append("=== Most recent faulthandler trace ===")
    sections.append(faulthandler)

    sections.append("")
    sections.append("=== Most recent crash file ===")
    if crash is None:
        sections.append("(no crash files found)")
    else:
        path, content = crash
        sections.append(f"From: {path}")
        sections.append(content)
        if crash_note is not None:
            sections.append(crash_note)

    sections.append("")
    sections.append("=== Active hooks ===")
    sections.append(hooks)
    sections.append("")
    sections.append(
        "Note: this section reflects the doctor subcommand's own process. "
        "Hooks are installed inside the long-running ytm TUI process — "
        "their effect is visible above (crashes/, faulthandler.log)."
    )

    return _redact("\n".join(sections))


def gather_diagnostics() -> str:
    """Return a multi-section text report describing the install.

    The report assembly — the banners, the aligned labels, the
    crash-file section, the final join and the redaction pass — is
    resolved by the ``_doctor_report`` engine above (native when the
    Rust extension is built, the pure-Python transcription otherwise);
    this function gathers the probes' results, in the reference's
    order, and hands them over.
    """
    from ytm_player import __version__
    from ytm_player.config.paths import (
        CONFIG_FILE,
        CRASH_DIR,
        LOG_FILE,
        SESSION_STATE_FILE,
        THEME_FILE,
    )
    from ytm_player.utils.logging import (
        get_recent_crash,
        get_recent_log_lines,
        list_active_hooks,
    )

    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    mpv_status = _mpv_version()
    libmpv_status = _libmpv_status()
    process_status = _running_status()
    mpris_status = _mpris_status()
    recent_errors = get_recent_log_lines(LOG_FILE, n=20, min_level="WARNING")
    recent_mpv = _recent_mpv_lines(LOG_FILE)
    faulthandler = _recent_faulthandler(CRASH_DIR)
    crash = get_recent_crash(CRASH_DIR)
    crash_note = None
    if crash is not None:
        crash_note = _crash_staleness_note(crash[1], __version__)
    hooks = list_active_hooks()

    return _doctor_report(
        __version__,
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
        system,
        release,
        machine,
        mpv_status,
        libmpv_status,
        CONFIG_FILE,
        THEME_FILE,
        SESSION_STATE_FILE,
        LOG_FILE,
        CRASH_DIR,
        process_status,
        mpris_status,
        recent_errors,
        recent_mpv,
        faulthandler,
        crash,
        crash_note,
        hooks,
    )


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same engines; these bindings replace the
# pure-Python transcriptions above when the extension is importable.  The
# rebinding must happen AFTER the definitions above — assigning the native
# functions any earlier would be clobbered by the ``def`` statements (the
# lesson that produced formatting.py's dead native bindings).

# Cache for the installed-packaging fold probe (see
# _packaging_admits_unicode_folds below): None = not probed yet.
_PACKAGING_ADMITS_UNICODE_FOLDS: bool | None = None

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        crash_staleness_note as _native_crash_staleness_note,
    )
except ImportError:
    pass
else:

    def _packaging_admits_unicode_folds() -> bool:
        """True when the installed ``packaging`` PEP 440 grammar admits the
        simple Unicode case folds — the long s ``ſ`` (U+017F), the Kelvin
        sign ``K`` (U+212A), the dotted and dotless ``İ``/``ı``
        (U+0130/U+0131) — wherever a letter can appear.

        ``packaging`` compiled its version grammar with plain
        ``re.IGNORECASE`` through 26.0, so those folds parsed as valid
        versions there (``Version("1.0+ſ")`` works); 26.1 scoped the
        grammar's groups ``(?a:...)`` and the local alphabet ASCII, so the
        same spellings are ``InvalidVersion`` now. The pure-Python
        fallback follows the installed ``packaging`` by construction — it
        calls ``Version`` directly — so the native engine must follow it
        too: this probe (one parse attempt, cached) is the verdict both
        sides agree on, whichever ``packaging`` the uncapped
        ``>=21.0`` dependency range admits.
        """
        global _PACKAGING_ADMITS_UNICODE_FOLDS
        if _PACKAGING_ADMITS_UNICODE_FOLDS is None:
            try:
                Version("1.0+\u017f")
            except ValueError:
                _PACKAGING_ADMITS_UNICODE_FOLDS = False
            else:
                _PACKAGING_ADMITS_UNICODE_FOLDS = True
        return _PACKAGING_ADMITS_UNICODE_FOLDS

    def _crash_staleness_note(content: str, current_version: str) -> str | None:  # noqa: F811
        """Crash-staleness verdict (native), on the installed packaging's grammar."""
        return _native_crash_staleness_note(
            content, current_version, _packaging_admits_unicode_folds()
        )


try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        recent_mpv_lines as _native_recent_mpv_lines,
    )
except ImportError:
    pass
else:
    _mpv_lines = _native_recent_mpv_lines  # type: ignore[assignment]

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        faulthandler_tail as _native_faulthandler_tail,
    )
except ImportError:
    pass
else:
    _faulthandler_tail = _native_faulthandler_tail  # type: ignore[assignment]

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        doctor_report as _native_doctor_report,
    )
except ImportError:
    pass
else:
    _doctor_report = _native_doctor_report  # type: ignore[assignment]

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        running_status_text as _native_running_status_text,
    )
except ImportError:
    pass
else:
    _running_status_text = _native_running_status_text  # type: ignore[assignment]

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        mpv_version_text as _native_mpv_version_text,
    )
except ImportError:
    pass
else:
    _mpv_version_text = _native_mpv_version_text  # type: ignore[assignment]

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        libmpv_status_text as _native_libmpv_status_text,
    )
except ImportError:
    pass
else:
    _libmpv_status_text = _native_libmpv_status_text  # type: ignore[assignment]

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        mpris_status_text as _native_mpris_status_text,
    )
except ImportError:
    pass
else:
    _mpris_status_text = _native_mpris_status_text  # type: ignore[assignment]
