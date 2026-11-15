//! Native cores for `src/ytm_player/services/download.py` — download
//! base opts and candidate lists.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

// ── Download service (services/download.py) ──────────────────────────────

/// The base yt-dlp options dict behind `DownloadService._build_opts` —
/// the pure half of a download's yt-dlp configuration.
///
/// Mirrors the pure-Python reference (`_base_opts`): the nine-key dict
/// in the reference's insertion order — the audio format selector
/// (`bestaudio[ext=webm]/bestaudio/best`, the module's
/// `_DOWNLOAD_FORMAT`), the staged `outtmpl` passed through as the very
/// object the caller assembled, the quiet/no-warning/noplaylist
/// `True` flags, the `extract_flat`/`writethumbnail`/`writeinfojson`
/// `False` flags, and the one-entry `postprocessors` list (the
/// `FFmpegExtractAudio` postprocessor with its opus codec and 128
/// quality, fresh containers every call). The configured extras stay
/// in the caller's `apply_configured_yt_dlp_options` pass; the
/// staging, the yt-dlp invocation and the atomic publication stay
/// Python-side in the service.
#[pyfunction]
#[pyo3(signature = (output_path))]
pub(crate) fn download_base_opts<'py>(
    py: Python<'py>,
    output_path: &Bound<'py, PyAny>,
) -> Bound<'py, PyDict> {
    let opts = PyDict::new(py);
    // Only str keys are inserted and only str/bool/object values —
    // always storable, so the insertions cannot fail.
    opts.set_item("format", "bestaudio[ext=webm]/bestaudio/best")
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("outtmpl", output_path)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("quiet", true)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("no_warnings", true)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("noplaylist", true)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("extract_flat", false)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("writethumbnail", false)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("writeinfojson", false)
        .expect("dict insertion with a str key cannot fail");
    let postprocessor = PyDict::new(py);
    postprocessor
        .set_item("key", "FFmpegExtractAudio")
        .expect("dict insertion with a str key cannot fail");
    postprocessor
        .set_item("preferredcodec", "opus")
        .expect("dict insertion with a str key cannot fail");
    postprocessor
        .set_item("preferredquality", "128")
        .expect("dict insertion with a str key cannot fail");
    let postprocessors = PyList::new(py, [postprocessor]).expect("list of one dict cannot fail");
    opts.set_item("postprocessors", postprocessors)
        .expect("dict insertion with a str key cannot fail");
    opts
}

/// The five probe filenames a completed download is looked up under —
/// the extension order behind `DownloadService._download_sync`'s
/// publication walk and `get_path`'s lookup.
///
/// Mirrors the pure-Python reference (`_candidate_names`): the
/// `<video_id>.<ext>` names built over the reference's own probe order
/// — `opus` first (the postprocessor's target), then `webm`, `m4a`,
/// `mp3` and `ogg` — answering as a fresh tuple of fresh strings. The
/// video id is typed `str` end to end in the service; a non-str id
/// raises the argument extraction's own `TypeError` where the
/// reference's f-string would have stringified it.
#[pyfunction]
#[pyo3(signature = (video_id))]
pub(crate) fn download_candidates<'py>(py: Python<'py>, video_id: &str) -> Bound<'py, PyTuple> {
    let names = ["opus", "webm", "m4a", "mp3", "ogg"].map(|ext| format!("{video_id}.{ext}"));
    PyTuple::new(py, names).expect("tuple of five strings cannot fail")
}
